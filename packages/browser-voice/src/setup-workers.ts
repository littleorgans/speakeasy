import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { endianness } from "node:os";
import { fileURLToPath } from "node:url";
import {
  MAX_HEADER_BYTES as MAX_WORKER_HEADER_BYTES, MAX_PCM_BYTES as MAX_WORKER_PCM_BYTES,
  MAX_TEXT_BYTES as MAX_WORKER_TEXT_BYTES, SETUP_PROTOCOL_VERSION, sha256 as hash,
  type PcmIdentity, type SetupId, type SetupExecution, type SetupExecutionContext,
  type SetupExecutionResult, type SetupExecutor, type SetupLoadResult,
} from "./setup-protocol.ts";
import type { ChildTranscriber, CpuReady } from "./transcriber.ts";

export { MAX_WORKER_HEADER_BYTES, MAX_WORKER_PCM_BYTES, MAX_WORKER_TEXT_BYTES };

const WORKER_PATH = fileURLToPath(new URL("../runtime/worker.py", import.meta.url));
const PYTHON_SETUPS = new Set([
  "mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy",
  "mlx-cleanup/qwen-targeted", "qwen/context",
]);

export type PythonWorkerResult = {
  text: string; processingMs: number; decodeMs?: number | null; rawText?: string; diagnostics?: Record<string, unknown>;
  metadata: Record<string, unknown>; firstUse: boolean;
  startupLoadWallMs: number; workerWallMs: number;
};
export type PythonWorkerOptions = {
  setupId: Exclude<SetupId, "sweep/score-2">;
  pythonPath: string;
  workerPath?: string;
  backendOptions?: Record<string, unknown>;
  expectedIdentity?: Record<string, unknown>;
  loadTimeoutMs?: number;
  operationTimeoutMs?: number;
};

export class SetupWorkerError extends Error {
  readonly rawText: string | undefined;
  constructor(message: string, rawText?: string) {
    super(message);
    this.name = "SetupWorkerError";
    this.rawText = rawText;
  }
}

type Header = Record<string, unknown>;
type Pending = {
  id: number; operation: string; type: string;
  resolve(header: Header): void; reject(error: Error): void; timer: NodeJS.Timeout;
};
type Session = {
  child: ChildProcessWithoutNullStreams; generation: string; buffer: Buffer;
  pending?: Pending; failure?: Error; metadata?: Record<string, unknown>; used: boolean;
  closed: Promise<void>; finish(): void; exited: boolean; killTimer?: NodeJS.Timeout;
};

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function finiteTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
function validateIdentity(value: PcmIdentity): void {
  if (!record(value) || value.sampleRate !== 16000 || value.channels !== 1
    || !Number.isSafeInteger(value.sampleCount) || value.sampleCount < 0
    || value.pcmBytes !== value.sampleCount * 4 || value.pcmBytes > MAX_WORKER_PCM_BYTES
    || typeof value.pcmSha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.pcmSha256)) {
    throw new SetupWorkerError("invalid PCM identity");
  }
}
function wireIdentity(value: PcmIdentity): Header {
  return { sampleRate: value.sampleRate, channels: value.channels, sampleCount: value.sampleCount,
    pcmBytes: value.pcmBytes, sha256: value.pcmSha256 };
}
function canonicalPcm(samples: Float32Array): { body: Buffer; identity: PcmIdentity } {
  if (!(samples instanceof Float32Array) || samples.byteLength > MAX_WORKER_PCM_BYTES) {
    throw new SetupWorkerError("invalid PCM length");
  }
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new SetupWorkerError("PCM must be finite");
  }
  // Own exactly this view before any asynchronous model load. Preserve signed zero.
  const body = Buffer.from(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
  if (endianness() === "BE") body.swap32();
  return { body, identity: { sampleRate: 16000, channels: 1, sampleCount: samples.length,
    pcmBytes: body.length, pcmSha256: hash(body) } };
}
function matches(actual: unknown, expected: unknown): boolean {
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length
    && expected.every((value, index) => matches(actual[index], value));
  if (record(expected)) return record(actual)
    && Object.entries(expected).every(([key, value]) => Object.hasOwn(actual, key) && matches(actual[key], value));
  return actual === expected;
}
function positiveOption(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value <= 0) throw new Error("worker deadlines must be positive");
  return value;
}

/** One backend, one in flight operation. Failed generations settle only after process close. */
export class PythonSetupWorker {
  readonly #options: PythonWorkerOptions;
  readonly #loadTimeoutMs: number;
  readonly #operationTimeoutMs: number;
  #session: Session | undefined;
  #nextId = 0;
  #busy = false;
  #closed = false;
  #stderr = Buffer.alloc(0);

  constructor(options: PythonWorkerOptions) {
    if (!PYTHON_SETUPS.has(options.setupId)) throw new Error("unsupported Python setup");
    this.#options = options;
    this.#loadTimeoutMs = positiveOption(options.loadTimeoutMs, 120_000);
    this.#operationTimeoutMs = positiveOption(options.operationTimeoutMs, 120_000);
  }

  /** Bounded diagnostics, deliberately separate from protocol stdout. */
  get stderr(): string { return this.#stderr.toString("utf8"); }

  transcribe(samples: Float32Array, signal?: AbortSignal): Promise<PythonWorkerResult> {
    if (this.#options.setupId === "mlx-cleanup/qwen-targeted") {
      return Promise.reject(new SetupWorkerError("cleanup worker does not transcribe"));
    }
    try {
      const { body, identity } = canonicalPcm(samples);
      return this.#run("transcribe", wireIdentity(identity), body, signal);
    } catch (error) { return Promise.reject(error); }
  }

  cleanup(text: string, basePcm: PcmIdentity, signal?: AbortSignal): Promise<PythonWorkerResult> {
    try {
      if (this.#options.setupId !== "mlx-cleanup/qwen-targeted") throw new Error("not a cleanup worker");
      validateIdentity(basePcm);
      if (typeof text !== "string" || Buffer.byteLength(text, "utf8") > MAX_WORKER_TEXT_BYTES) {
        throw new Error("cleanup text exceeds limit");
      }
      return this.#run("cleanup", { text, basePcm: wireIdentity(basePcm), B: null,
        transcriptSha256: hash(text) }, Buffer.alloc(0), signal);
    } catch (error) { return Promise.reject(error); }
  }

  load(signal?: AbortSignal): Promise<SetupLoadResult> {
    return this.#run("load", {}, Buffer.alloc(0), signal);
  }

  /** Scheduler path: execute cannot load or replace a generation after load failed. */
  execute(request: SetupExecution, { signal }: SetupExecutionContext): Promise<PythonWorkerResult> {
    try {
      if (request.setupId !== this.#options.setupId || typeof request.requestId !== "string"
        || request.requestId.length === 0 || Buffer.byteLength(request.requestId) > 128) {
        throw new SetupWorkerError("execution identity mismatch");
      }
      if (request.operation === "transcribe") {
        const { body, identity } = canonicalPcm(request.pcm.samples);
        validateIdentity(request.pcm);
        if (!matches(identity, { sampleRate: request.pcm.sampleRate, channels: request.pcm.channels,
          sampleCount: request.pcm.sampleCount, pcmBytes: request.pcm.pcmBytes, pcmSha256: request.pcm.pcmSha256 })
          || hash(request.pcm.bytes) !== identity.pcmSha256) throw new Error("canonical PCM identity mismatch");
        return this.#run("transcribe", wireIdentity(identity), body, signal, true);
      }
      validateIdentity(request.base);
      if (request.B !== null || typeof request.A !== "string"
        || Buffer.byteLength(request.A) > MAX_WORKER_TEXT_BYTES || hash(request.A) !== request.baseTextSha256) {
        throw new Error("cleanup transcript identity mismatch");
      }
      return this.#run("cleanup", { text: request.A,
        basePcm: wireIdentity(request.base), B: null,
        transcriptSha256: request.baseTextSha256 }, Buffer.alloc(0), signal, true);
    } catch (error) { return Promise.reject(error); }
  }

  async close(): Promise<void> {
    this.#closed = true;
    const session = this.#session;
    if (session) {
      this.#retire(session, new SetupWorkerError("worker closed"));
      await session.closed;
    }
  }

  async release(): Promise<void> {
    if (this.#busy) throw new SetupWorkerError("worker is busy");
    const session = this.#session;
    if (session) {
      this.#retire(session, new SetupWorkerError("worker released"));
      await session.closed;
    }
  }

  #run(operation: "load", fields: Header, body: Buffer, signal?: AbortSignal): Promise<SetupLoadResult>;
  #run(operation: "transcribe" | "cleanup", fields: Header, body: Buffer, signal?: AbortSignal,
    requireReady?: boolean): Promise<PythonWorkerResult>;
  async #run(operation: string, fields: Header, body: Buffer, signal?: AbortSignal,
    requireReady = false): Promise<PythonWorkerResult | SetupLoadResult> {
    if (this.#closed) throw new SetupWorkerError("worker closed");
    if (this.#busy) throw new SetupWorkerError("worker is busy");
    if (signal?.aborted) throw new SetupWorkerError("worker cancelled");
    this.#busy = true;
    const started = performance.now();
    let session: Session | undefined;
    const cancel = () => { if (session) this.#retire(session, new SetupWorkerError("worker cancelled")); };
    signal?.addEventListener("abort", cancel, { once: true });
    try {
      // An idle child may already be retiring. Keep this slot until it exits.
      if (requireReady && (!this.#session?.metadata || this.#session.failure)) {
        throw new SetupWorkerError("worker is not ready; explicit load required");
      }
      if (this.#session?.failure) await this.#session.closed;
      if (this.#closed) throw new SetupWorkerError("worker closed");
      if (signal?.aborted) throw new SetupWorkerError("worker cancelled");
      session = this.#session ?? this.#spawn();
      let startupMs = 0;
      if (!session.metadata) {
        const ready = await this.#request(session, "load", "ready", {
          options: this.#options.backendOptions ?? {},
        }, Buffer.alloc(0), this.#loadTimeoutMs);
        if (!record(ready.metadata) || !matches(ready.metadata, this.#options.expectedIdentity ?? {})) {
          throw new SetupWorkerError("worker ready identity mismatch");
        }
        session.metadata = ready.metadata;
        startupMs = performance.now() - started;
      }
      if (signal?.aborted) cancel();
      if (session.failure) throw session.failure;
      // Ready includes backend load. Independent process startup is unavailable.
      if (operation === "load") return { startupMs: startupMs > 0 ? null : 0,
        loadMs: startupMs > 0 && finiteTime(session.metadata.loadMs) ? session.metadata.loadMs : startupMs > 0 ? null : 0,
        firstUse: !session.used };
      const reply = await this.#request(session, operation, "result", fields, body, this.#operationTimeoutMs);
      const result = reply.result;
      if (!record(result) || typeof result.text !== "string" || !finiteTime(result.processingMs)
        || (result.decodeMs !== undefined && result.decodeMs !== null && !finiteTime(result.decodeMs))
        || (result.rawText !== undefined && typeof result.rawText !== "string")
        || (result.diagnostics !== undefined && !record(result.diagnostics))) {
        throw new SetupWorkerError("invalid worker result");
      }
      if (session.failure) throw session.failure;
      const firstUse = !session.used;
      session.used = true;
      return { text: result.text, processingMs: result.processingMs,
        ...(result.decodeMs !== undefined ? { decodeMs: result.decodeMs as number | null } : {}),
        ...(typeof result.rawText === "string" ? { rawText: result.rawText } : {}),
        ...(result.diagnostics !== undefined ? { diagnostics: result.diagnostics } : {}),
        metadata: session.metadata, firstUse, startupLoadWallMs: startupMs, workerWallMs: performance.now() - started };
    } catch (error) {
      const failure = error instanceof Error ? error : new Error(String(error));
      if (session) {
        this.#retire(session, failure);
        await session.closed;
      }
      throw failure;
    } finally {
      signal?.removeEventListener("abort", cancel);
      this.#busy = false;
    }
  }

  #spawn(): Session {
    const generation = randomUUID();
    const child = spawn(this.#options.pythonPath, ["-B", "-u", this.#options.workerPath ?? WORKER_PATH,
      "--setup", this.#options.setupId, "--generation", generation], { stdio: ["pipe", "pipe", "pipe"] });
    let finish!: () => void;
    const closed = new Promise<void>((resolve) => { finish = resolve; });
    const session: Session = { child, generation, buffer: Buffer.alloc(0), used: false,
      closed, finish, exited: false };
    this.#session = session;
    this.#stderr = Buffer.alloc(0);
    child.stdout.on("data", (chunk: Buffer) => this.#receive(session, chunk));
    child.stderr.on("data", (chunk: Buffer) => {
      if (this.#session === session) this.#stderr = Buffer.from(Buffer.concat([this.#stderr, chunk]).subarray(-16_384));
    });
    child.stdin.on("error", (error) => this.#retire(session, error));
    child.stdout.on("error", (error) => this.#retire(session, error));
    child.stdout.on("end", () => this.#retire(session, new SetupWorkerError("worker protocol pipe closed")));
    child.on("error", (error) => this.#retire(session, error));
    child.on("exit", () => { session.failure ??= new SetupWorkerError("worker exited"); });
    child.on("close", (code, signal) => {
      session.exited = true;
      clearTimeout(session.killTimer);
      const pending = session.pending;
      session.pending = undefined;
      if (pending) {
        clearTimeout(pending.timer);
        pending.reject(session.failure ?? new SetupWorkerError(`worker exited (${signal ?? code})`));
      }
      if (this.#session === session) this.#session = undefined;
      session.finish();
    });
    return session;
  }

  #request(session: Session, operation: string, type: string, fields: Header, body: Buffer, timeoutMs: number): Promise<Header> {
    if (session.failure) return Promise.reject(session.failure);
    const id = ++this.#nextId;
    const header = Buffer.from(JSON.stringify({ ...fields, version: SETUP_PROTOCOL_VERSION, generation: session.generation,
      id, setupId: this.#options.setupId, operation, pcmBytes: body.length }), "utf8");
    if (header.length > MAX_WORKER_HEADER_BYTES) throw new SetupWorkerError("request exceeds header limit");
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32LE(header.length);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.#retire(session,
        new SetupWorkerError(`${operation} timed out after ${timeoutMs} ms`)), timeoutMs);
      session.pending = { id, operation, type, resolve, reject, timer };
      const frame = Buffer.concat([prefix, header, body]);
      session.child.stdin.write(frame, (error) => { if (error) this.#retire(session, error); });
    });
  }

  #receive(session: Session, chunk: Buffer): void {
    if (session.failure) return;
    try {
      if (!session.pending) throw new Error("unsolicited reply");
      session.buffer = Buffer.concat([session.buffer, chunk]);
      while (session.buffer.length >= 4) {
        const length = session.buffer.readUInt32LE();
        if (length === 0 || length > MAX_WORKER_HEADER_BYTES) throw new Error("invalid reply length");
        if (session.buffer.length < length + 4) return;
        const json = new TextDecoder("utf-8", { fatal: true }).decode(session.buffer.subarray(4, length + 4));
        const reply: unknown = JSON.parse(json);
        session.buffer = session.buffer.subarray(length + 4);
        const pending = session.pending;
        if (!record(reply) || !pending || reply.version !== SETUP_PROTOCOL_VERSION || reply.generation !== session.generation
          || reply.id !== pending.id || reply.setupId !== this.#options.setupId
          || reply.operation !== pending.operation || reply.pcmBytes !== 0) throw new Error("reply envelope mismatch");
        if (reply.type === "error") {
          if (typeof reply.error !== "string" || (reply.rawText !== undefined && typeof reply.rawText !== "string")) {
            throw new Error("invalid worker error");
          }
          this.#retire(session, new SetupWorkerError(reply.error, reply.rawText as string | undefined));
          return;
        }
        if (reply.type !== pending.type) throw new Error("unexpected reply type");
        session.pending = undefined;
        clearTimeout(pending.timer);
        pending.resolve(reply);
        if (session.buffer.length) throw new Error("unsolicited reply bytes");
      }
    } catch (error) {
      this.#retire(session, new SetupWorkerError(error instanceof Error ? error.message : String(error)));
    }
  }

  #retire(session: Session, error: Error): void {
    session.failure ??= error;
    if (session.exited || session.killTimer) return;
    session.child.kill("SIGTERM");
    session.killTimer = setTimeout(() => { session.child.kill("SIGKILL"); }, 250);
    session.killTimer.unref();
  }
}

export type SetupWorkersOptions = {
  python?: Partial<Record<Exclude<SetupId, "sweep/score-2">, Omit<PythonWorkerOptions, "setupId">>>;
  /** Supply one dedicated ChildTranscriber constructed with mode: "sweep/score-2". */
  cpu?: Pick<ChildTranscriber, "prepare" | "transcribe" | "closeAndWait">;
};

/** Scheduler executor. CPU child ownership and advanced IPC remain in ChildTranscriber. */
export class SetupWorkers implements SetupExecutor {
  readonly #python = new Map<SetupId, PythonSetupWorker>();
  readonly #cpu: SetupWorkersOptions["cpu"];
  #cpuReady: CpuReady | undefined;
  #cpuUsed = false;
  #cpuBusy = false;
  #cpuFinished: Promise<void> | undefined;
  #closed = false;

  constructor(options: SetupWorkersOptions = {}) {
    this.#cpu = options.cpu;
    for (const [setupId, config] of Object.entries(options.python ?? {})) {
      this.#python.set(setupId as SetupId, new PythonSetupWorker({ ...config,
        setupId: setupId as PythonWorkerOptions["setupId"] }));
    }
  }

  async load(setupId: SetupId, { signal }: SetupExecutionContext): Promise<SetupLoadResult> {
    this.#assertOpen();
    if (setupId !== "sweep/score-2") return this.#worker(setupId).load(signal);
    return this.#withCpu(signal, async () => {
      const ready = await this.#cpu!.prepare();
      const cold = this.#cpuReady !== ready;
      if (cold) this.#cpuUsed = false;
      this.#cpuReady = ready;
      return { startupMs: cold ? null : 0, loadMs: cold ? ready.load.loadMs : 0, firstUse: !this.#cpuUsed };
    });
  }

  async execute(request: SetupExecution, context: SetupExecutionContext): Promise<SetupExecutionResult> {
    this.#assertOpen();
    if (request.setupId !== "sweep/score-2") {
      const result = await this.#worker(request.setupId).execute(request, context);
      return { text: result.text, processingMs: result.processingMs, workerWallMs: result.workerWallMs,
        ...(result.decodeMs !== undefined ? { decodeMs: result.decodeMs } : {}),
        ...(result.rawText !== undefined ? { rawText: result.rawText } : {}),
        ...(result.diagnostics ? { diagnostics: result.diagnostics } : {}) };
    }
    return this.#withCpu(context.signal, async () => {
      if (!this.#cpuReady || request.operation !== "transcribe") throw new Error("CPU requires explicit load");
      const result = await this.#cpu!.transcribe(request.pcm.samples);
      if (!result.comparison) throw new Error("CPU executor requires fixed ChildTranscriber mode");
      this.#cpuUsed = true;
      return { text: result.text, decodeMs: result.decodeMs, processingMs: result.comparison.processingMs,
        workerWallMs: result.comparison.workerWallMs,
        diagnostics: { modelConstructionMs: result.comparison.modelConstructionMs } };
    });
  }

  async release(setupId: SetupId): Promise<void> {
    if (setupId !== "sweep/score-2") return this.#python.get(setupId)?.release();
    if (this.#cpuBusy) throw new SetupWorkerError("CPU worker is busy");
    this.#cpuReady = undefined;
    await this.#cpu?.closeAndWait();
  }

  async close(): Promise<void> {
    this.#closed = true;
    this.#cpuReady = undefined;
    // Await every child even if a sibling close rejects.
    const activeCpu = this.#cpuFinished;
    const results = await Promise.allSettled([...this.#python.values()].map((worker) => worker.close())
      .concat(this.#cpu ? [this.#cpu.closeAndWait()] : []));
    if (activeCpu) {
      await activeCpu;
      await this.#cpu?.closeAndWait();
    }
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  #assertOpen(): void { if (this.#closed) throw new SetupWorkerError("workers closed"); }
  #worker(setupId: SetupId): PythonSetupWorker {
    const worker = this.#python.get(setupId);
    if (!worker) throw new SetupWorkerError(`local runtime is not configured for ${setupId}`);
    return worker;
  }

  async #withCpu<T>(signal: AbortSignal, operation: () => Promise<T>): Promise<T> {
    if (!this.#cpu) throw new SetupWorkerError("CPU runtime is not configured");
    if (this.#cpuBusy) throw new SetupWorkerError("CPU worker is busy");
    if (signal.aborted) throw new SetupWorkerError("worker cancelled");
    this.#cpuBusy = true;
    let finish!: () => void;
    this.#cpuFinished = new Promise<void>((resolve) => { finish = resolve; });
    let retirement: Promise<void> | undefined;
    const cancel = () => {
      this.#cpuReady = undefined;
      retirement = this.#cpu!.closeAndWait();
      // The operation catch below awaits retirement; attach a handler immediately.
      void retirement.catch(() => {});
    };
    signal.addEventListener("abort", cancel, { once: true });
    try {
      const result = await operation();
      if (signal.aborted) throw new SetupWorkerError("worker cancelled");
      this.#assertOpen();
      return result;
    } catch (error) {
      this.#cpuReady = undefined;
      await this.#cpu.closeAndWait();
      throw signal.aborted ? new SetupWorkerError("worker cancelled") : error;
    } finally {
      try { await retirement; }
      finally {
        signal.removeEventListener("abort", cancel);
        this.#cpuBusy = false;
        this.#cpuFinished = undefined;
        finish();
      }
    }
  }
}
