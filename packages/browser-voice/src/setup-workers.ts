import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { endianness } from "node:os";
import { fileURLToPath } from "node:url";
import type { NamesSnapshot } from "./names.ts";
import {
  assertLivePartial, MAX_HEADER_BYTES as MAX_WORKER_HEADER_BYTES, MAX_PCM_BYTES as MAX_WORKER_PCM_BYTES,
  MAX_TEXT_BYTES as MAX_WORKER_TEXT_BYTES, SETUP_PROTOCOL_VERSION, sha256 as hash,
  type LivePartial, type LiveSettings, type PcmIdentity, type SetupId, type SetupExecution, type SetupExecutionContext,
  type SetupExecutionResult, type SetupExecutor, type SetupLoadContext, type SetupLoadResult,
} from "./setup-protocol.ts";
import { isCpuSetupId, type ChildTranscriber, type CpuReady, type CpuSetupId } from "./transcriber.ts";
import { isWorkerOutput, LOCAL_OUTPUT_MODELS, type LocalOutputModelId } from "../../speech-io/src/tts/catalogue.ts";
import type { SynthRequest } from "../../speech-io/src/tts/synth.ts";
import { CLEANUP_MODEL_IDS, isCleanupModelId, type CleanupModelId } from "./text-cleanup.ts";

export { MAX_WORKER_HEADER_BYTES, MAX_WORKER_PCM_BYTES, MAX_WORKER_TEXT_BYTES };

const WORKER_PATH = fileURLToPath(new URL("../runtime/worker.py", import.meta.url));
/** Speech in setups, speech out worker voices and text cleanup models share this one Python worker. */
const PYTHON_WORKERS = new Set<string>([
  "mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy",
  "mlx-cleanup/qwen-targeted", "qwen/context", "hex/unified", "hex/vocabulary", "cohere/transcribe",
  "nemotron/streaming", "apple/speech", "apple/dictation",
  ...LOCAL_OUTPUT_MODELS.map((model) => model.id).filter(isWorkerOutput),
  ...CLEANUP_MODEL_IDS,
]);

export type PythonWorkerResult = {
  text: string; processingMs: number; decodeMs?: number | null; rawText?: string; diagnostics?: Record<string, unknown>;
  metadata: Record<string, unknown>; firstUse: boolean;
  startupLoadWallMs: number; workerWallMs: number;
};
export type PythonVoiceResult = {
  samples: Float32Array; sampleRate: number; processingMs: number; firstUse: boolean;
  startupLoadWallMs: number; workerWallMs: number;
};
/** Every speech in setup that runs in the Python worker: all but the sherpa CPU setups. */
export type PythonSetupId = Exclude<SetupId, CpuSetupId>;
/** One final's text for a text cleanup model, with the context only a model that takes it receives. */
export type TextCleanupInput = { text: string; instruction?: string; names?: readonly string[]; earlier?: readonly string[] };
export type PythonWorkerOptions = {
  /** A Python speech in setup, a worker voice or a text cleanup model; the wire field keeps its setupId name. */
  setupId: PythonSetupId | LocalOutputModelId | CleanupModelId;
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
  /** The names snapshot this generation loaded with; none for a setup that takes no names. */
  names?: NamesSnapshot;
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
/** PCM identity at any rate: speech in is 16 kHz, a voice copy reference keeps its capture rate. */
type RatedPcmIdentity = Omit<PcmIdentity, "sampleRate"> & { sampleRate: number };
function wireIdentity(value: RatedPcmIdentity): Header {
  return { sampleRate: value.sampleRate, channels: value.channels, sampleCount: value.sampleCount,
    pcmBytes: value.pcmBytes, sha256: value.pcmSha256 };
}
function canonicalPcm(samples: Float32Array, sampleRate = 16000): { body: Buffer; identity: RatedPcmIdentity } {
  if (!(samples instanceof Float32Array) || samples.byteLength > MAX_WORKER_PCM_BYTES) {
    throw new SetupWorkerError("invalid PCM length");
  }
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new SetupWorkerError("PCM must be finite");
  }
  // Own exactly this view before any asynchronous model load. Preserve signed zero.
  const body = Buffer.from(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
  if (endianness() === "BE") body.swap32();
  return { body, identity: { sampleRate, channels: 1, sampleCount: samples.length,
    pcmBytes: body.length, pcmSha256: hash(body) } };
}
/** The SHA256 of samples as the worker receives them: little endian Float32. */
export function pcmDigest(samples: Float32Array): string {
  return canonicalPcm(samples).identity.pcmSha256;
}
function matches(actual: unknown, expected: unknown): boolean {
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length
    && expected.every((value, index) => matches(actual[index], value));
  if (record(expected)) return record(actual)
    && Object.entries(expected).every(([key, value]) => Object.hasOwn(actual, key) && matches(actual[key], value));
  return actual === expected;
}
/** The audio reply's PCM, checked against its stated identity. */
function voicePcm(reply: Header): Float32Array {
  const audio = reply.audio;
  const pcm = reply.pcm;
  if (!record(audio) || !(pcm instanceof Buffer) || !finiteTime(reply.processingMs)
    || !Number.isInteger(audio.sampleRate) || (audio.sampleRate as number) < 8_000 || (audio.sampleRate as number) > 192_000
    || audio.sampleCount !== pcm.length / 4 || audio.sha256 !== hash(pcm)) throw new SetupWorkerError("invalid voice audio");
  // A fresh, aligned copy: the caller owns it, and pooled buffers may sit at any offset.
  const bytes = new Uint8Array(pcm.length);
  bytes.set(pcm);
  if (endianness() === "BE") Buffer.from(bytes.buffer).swap32();
  const samples = new Float32Array(bytes.buffer);
  if (!samples.every(Number.isFinite)) throw new SetupWorkerError("voice audio must be finite");
  return samples;
}
function positiveOption(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value <= 0) throw new Error("worker deadlines must be positive");
  return value;
}
/** Signals the worker's process group, so native runners it started end with it. */
function signalGroup(child: ChildProcessWithoutNullStreams, signal: NodeJS.Signals): void {
  if (child.pid === undefined) return void child.kill(signal);
  try { process.kill(-child.pid, signal); } catch { /* The group already ended. */ }
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
    if (!PYTHON_WORKERS.has(options.setupId)) throw new Error("unsupported Python setup");
    this.#options = options;
    this.#loadTimeoutMs = positiveOption(options.loadTimeoutMs, 120_000);
    this.#operationTimeoutMs = positiveOption(options.operationTimeoutMs, 120_000);
  }

  /** Bounded diagnostics, deliberately separate from protocol stdout. */
  get stderr(): string { return this.#stderr.toString("utf8"); }

  transcribe(samples: Float32Array, signal?: AbortSignal): Promise<PythonWorkerResult> {
    if (this.#options.setupId === "mlx-cleanup/qwen-targeted" || isCleanupModelId(this.#options.setupId)) {
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

  /** A text cleanup model's pass over one final's own text; the worker loads on first use. */
  cleanText({ text, ...context }: TextCleanupInput, signal?: AbortSignal): Promise<PythonWorkerResult> {
    try {
      if (!isCleanupModelId(this.#options.setupId)) throw new SetupWorkerError("not a text cleanup worker");
      if (typeof text !== "string" || Buffer.byteLength(text, "utf8") > MAX_WORKER_TEXT_BYTES) {
        throw new SetupWorkerError("cleanup text exceeds limit");
      }
      return this.#run("cleanup", { text, ...context }, Buffer.alloc(0), signal);
    } catch (error) { return Promise.reject(error); }
  }

  /**
   * One planned segment of speech out; the worker loads on first use. A
   * reading's settings ride in the header and a voice copy reference as the
   * body, its identity hashed here; the worker refuses either for a voice
   * that does not take it.
   */
  speak(text: string, voice: string, speed: number, signal?: AbortSignal,
    { settings, reference }: Pick<SynthRequest, "settings" | "reference"> = {}): Promise<PythonVoiceResult> {
    try {
      if (typeof text !== "string" || !text.trim() || Buffer.byteLength(text, "utf8") > MAX_WORKER_TEXT_BYTES) {
        throw new SetupWorkerError("speak text is empty or exceeds limit");
      }
      const pcm = reference ? canonicalPcm(reference.samples, reference.sampleRate) : undefined;
      if (pcm && pcm.identity.pcmSha256 !== reference!.sha256) throw new SetupWorkerError("reference identity mismatch");
      return this.#run("speak", { text, voice, speed, ...(settings ? { settings } : {}),
        ...(pcm ? { reference: { ...wireIdentity(pcm.identity), transcript: reference!.transcript } } : {}) },
      pcm?.body ?? Buffer.alloc(0), signal);
    } catch (error) { return Promise.reject(error); }
  }

  /** A setup that takes names loads with them as a backend option; another list retires this generation first. */
  load(signal?: AbortSignal, names?: NamesSnapshot): Promise<SetupLoadResult> {
    return this.#run("load", names ? { names } : {}, Buffer.alloc(0), signal);
  }

  /** Live words after an explicit load; start opens a fresh stream in the worker, with its settings. */
  feed(samples: Float32Array, start: boolean, signal?: AbortSignal, settings?: LiveSettings): Promise<LivePartial> {
    try {
      const { body, identity } = canonicalPcm(samples);
      return this.#run("feed", { ...wireIdentity(identity), start, ...(start && settings ? { settings } : {}) },
        body, signal, true);
    } catch (error) { return Promise.reject(error); }
  }

  /** Scheduler path: execute cannot load or replace a generation after load failed. */
  execute(request: SetupExecution, { signal }: SetupExecutionContext): Promise<PythonWorkerResult> {
    try {
      if (request.setupId !== this.#options.setupId || typeof request.requestId !== "string"
        || request.requestId.length === 0 || Buffer.byteLength(request.requestId) > 128) {
        throw new SetupWorkerError("execution identity mismatch");
      }
      if (this.#session && this.#session.names?.digest !== request.names?.digest) {
        throw new SetupWorkerError("names changed since load; explicit load required");
      }
      if (request.operation === "transcribe") {
        const { body, identity } = canonicalPcm(request.pcm.samples);
        validateIdentity(request.pcm);
        if (!matches(identity, { sampleRate: request.pcm.sampleRate, channels: request.pcm.channels,
          sampleCount: request.pcm.sampleCount, pcmBytes: request.pcm.pcmBytes, pcmSha256: request.pcm.pcmSha256 })
          || hash(request.pcm.bytes) !== identity.pcmSha256) throw new Error("canonical PCM identity mismatch");
        const settings = request.settings ? { settings: request.settings } : {};
        return this.#run("transcribe", { ...wireIdentity(identity), ...settings }, body, signal, true);
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

  #run(operation: "load", fields: { names?: NamesSnapshot }, body: Buffer, signal?: AbortSignal): Promise<SetupLoadResult>;
  #run(operation: "transcribe" | "cleanup", fields: Header, body: Buffer, signal?: AbortSignal,
    requireReady?: boolean): Promise<PythonWorkerResult>;
  #run(operation: "speak", fields: Header, body: Buffer, signal?: AbortSignal): Promise<PythonVoiceResult>;
  #run(operation: "feed", fields: Header, body: Buffer, signal: AbortSignal | undefined,
    requireReady: true): Promise<LivePartial>;
  async #run(operation: string, fields: Header, body: Buffer, signal?: AbortSignal,
    requireReady = false): Promise<PythonWorkerResult | PythonVoiceResult | SetupLoadResult | LivePartial> {
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
      const names = operation === "load" ? fields.names as NamesSnapshot | undefined : this.#session?.names;
      if (this.#session && this.#session.names?.digest !== names?.digest) {
        // The list changed: this generation exits before the next one loads.
        const stale = this.#session;
        this.#retire(stale, new SetupWorkerError("names changed"));
        await stale.closed;
      }
      if (this.#closed) throw new SetupWorkerError("worker closed");
      if (signal?.aborted) throw new SetupWorkerError("worker cancelled");
      session = this.#session ?? this.#spawn();
      let startupMs = 0;
      if (!session.metadata) {
        session.names = names;
        const ready = await this.#request(session, "load", "ready", {
          options: { ...this.#options.backendOptions, ...(names ? { names: names.names } : {}) },
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
      if (operation === "feed") {
        const reply = await this.#request(session, operation, "partial", fields, body, this.#operationTimeoutMs);
        const partial = reply.partial;
        try { assertLivePartial(partial); }
        catch { throw new SetupWorkerError("invalid worker partial"); }
        if (session.failure) throw session.failure;
        // The final pass after live words is no longer this generation's first use.
        session.used = true;
        return { stable: partial.stable, tentative: partial.tentative, processingMs: partial.processingMs };
      }
      if (operation === "speak") {
        const reply = await this.#request(session, operation, "audio", fields, body, this.#operationTimeoutMs);
        const samples = voicePcm(reply);
        if (session.failure) throw session.failure;
        const firstUse = !session.used;
        session.used = true;
        return { samples, sampleRate: (reply.audio as Header).sampleRate as number, processingMs: reply.processingMs as number,
          firstUse, startupLoadWallMs: startupMs, workerWallMs: performance.now() - started };
      }
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
    // A separate lifetime pipe stays observable while Python is blocked in a backend call.
    const child = spawn(this.#options.pythonPath, ["-B", "-u", this.#options.workerPath ?? WORKER_PATH,
      "--setup", this.#options.setupId, "--generation", generation, "--parent-fd", "3"],
      { stdio: ["pipe", "pipe", "pipe", "pipe"],
      // Its own process group: retirement reaches every runner it starts, busy ones included.
      detached: true });
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
        // Only a voice's audio reply carries PCM after its header.
        const pcmBytes = record(reply) && reply.type === "audio" ? reply.pcmBytes : 0;
        if (!Number.isSafeInteger(pcmBytes) || (pcmBytes as number) < 0 || (pcmBytes as number) > MAX_WORKER_PCM_BYTES) {
          throw new Error("invalid reply PCM length");
        }
        if (session.buffer.length < length + 4 + (pcmBytes as number)) return;
        const pcm = Buffer.from(session.buffer.subarray(length + 4, length + 4 + (pcmBytes as number)));
        session.buffer = session.buffer.subarray(length + 4 + (pcmBytes as number));
        const pending = session.pending;
        if (!record(reply) || !pending || reply.version !== SETUP_PROTOCOL_VERSION || reply.generation !== session.generation
          || reply.id !== pending.id || reply.setupId !== this.#options.setupId
          || reply.operation !== pending.operation || reply.pcmBytes !== pcmBytes) throw new Error("reply envelope mismatch");
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
        pending.resolve(pending.type === "audio" ? { ...reply, pcm } : reply);
        if (session.buffer.length) throw new Error("unsolicited reply bytes");
      }
    } catch (error) {
      this.#retire(session, new SetupWorkerError(error instanceof Error ? error.message : String(error)));
    }
  }

  #retire(session: Session, error: Error): void {
    session.failure ??= error;
    if (session.exited || session.killTimer) return;
    signalGroup(session.child, "SIGTERM");
    session.killTimer = setTimeout(() => signalGroup(session.child, "SIGKILL"), 250);
    session.killTimer.unref();
  }
}

type CpuTranscriber = Pick<ChildTranscriber, "prepare" | "transcribe" | "closeAndWait">;
export type SetupWorkersOptions = {
  python?: Partial<Record<PythonSetupId, Omit<PythonWorkerOptions, "setupId">>>;
  /** One dedicated ChildTranscriber per CPU setup, constructed with that setup id as its mode. */
  cpu?: Partial<Record<CpuSetupId, CpuTranscriber>>;
};

/** Scheduler executor. CPU child ownership and advanced IPC remain in ChildTranscriber. */
export class SetupWorkers implements SetupExecutor {
  readonly #python = new Map<SetupId, PythonSetupWorker>();
  readonly #cpu = new Map<CpuSetupId, CpuSetup>();
  #closed = false;

  constructor(options: SetupWorkersOptions = {}) {
    for (const [setupId, config] of Object.entries(options.python ?? {})) {
      this.#python.set(setupId as SetupId, new PythonSetupWorker({ ...config,
        setupId: setupId as PythonWorkerOptions["setupId"] }));
    }
    for (const [setupId, transcriber] of Object.entries(options.cpu ?? {})) {
      this.#cpu.set(setupId as CpuSetupId, new CpuSetup(transcriber, () => this.#assertOpen()));
    }
  }

  async load(setupId: SetupId, { signal, names }: SetupLoadContext): Promise<SetupLoadResult> {
    this.#assertOpen();
    return isCpuSetupId(setupId) ? this.#cpuSetup(setupId).load(signal, names) : this.#worker(setupId).load(signal, names);
  }

  async execute(request: SetupExecution, context: SetupExecutionContext): Promise<SetupExecutionResult> {
    this.#assertOpen();
    if (isCpuSetupId(request.setupId)) return this.#cpuSetup(request.setupId).execute(request, context.signal);
    const result = await this.#worker(request.setupId).execute(request, context);
    return { text: result.text, processingMs: result.processingMs, workerWallMs: result.workerWallMs,
      ...(result.decodeMs !== undefined ? { decodeMs: result.decodeMs } : {}),
      ...(result.rawText !== undefined ? { rawText: result.rawText } : {}),
      ...(result.diagnostics ? { diagnostics: result.diagnostics } : {}) };
  }

  /** Live words: only the Python setups whose backend streams answer; others reject in the worker. */
  async feed(setupId: SetupId, samples: Float32Array, start: boolean, { signal }: SetupExecutionContext,
    settings?: LiveSettings): Promise<LivePartial> {
    this.#assertOpen();
    if (isCpuSetupId(setupId)) throw new SetupWorkerError(`${setupId} gives final text only`);
    return this.#worker(setupId).feed(samples, start, signal, settings);
  }

  async release(setupId: SetupId): Promise<void> {
    if (isCpuSetupId(setupId)) return this.#cpu.get(setupId)?.release();
    return this.#python.get(setupId)?.release();
  }

  async close(): Promise<void> {
    this.#closed = true;
    // Await every child even if a sibling close rejects.
    const results = await Promise.allSettled([...this.#python.values(), ...this.#cpu.values()].map((worker) => worker.close()));
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  #assertOpen(): void { if (this.#closed) throw new SetupWorkerError("workers closed"); }
  #worker(setupId: PythonSetupId): PythonSetupWorker {
    const worker = this.#python.get(setupId);
    if (!worker) throw new SetupWorkerError(`local runtime is not configured for ${setupId}`);
    return worker;
  }
  #cpuSetup(setupId: CpuSetupId): CpuSetup {
    const setup = this.#cpu.get(setupId);
    if (!setup) throw new SetupWorkerError("CPU runtime is not configured");
    return setup;
  }
}

/** One CPU setup: its child, readiness, the names it prepared and the one operation it runs at a time. */
class CpuSetup {
  readonly #cpu: CpuTranscriber;
  readonly #assertOpen: () => void;
  #ready: CpuReady | undefined;
  #names: NamesSnapshot | undefined;
  #used = false;
  #busy = false;
  #finished: Promise<void> | undefined;

  constructor(cpu: CpuTranscriber, assertOpen: () => void) {
    this.#cpu = cpu;
    this.#assertOpen = assertOpen;
  }

  /** A names change rebuilds the recognizer inside prepare, so it counts as a cold load. */
  load(signal: AbortSignal, names?: NamesSnapshot): Promise<SetupLoadResult> {
    return this.#with(signal, async () => {
      const ready = await this.#cpu.prepare(names?.names ?? []);
      const cold = this.#ready !== ready || this.#names?.digest !== names?.digest;
      if (cold) this.#used = false;
      this.#ready = ready;
      this.#names = names;
      return { startupMs: cold ? null : 0, loadMs: cold ? ready.load.loadMs : 0, firstUse: !this.#used };
    });
  }

  execute(request: SetupExecution, signal: AbortSignal): Promise<SetupExecutionResult> {
    return this.#with(signal, async () => {
      if (!this.#ready || request.operation !== "transcribe") throw new Error("CPU requires explicit load");
      if (this.#names?.digest !== request.names?.digest) throw new Error("names changed since load; explicit load required");
      const result = await this.#cpu.transcribe(request.pcm.samples, request.names?.names ?? []);
      if (!result.comparison) throw new Error("CPU executor requires fixed ChildTranscriber mode");
      this.#used = true;
      return { text: result.text, decodeMs: result.decodeMs, processingMs: result.comparison.processingMs,
        workerWallMs: result.comparison.workerWallMs,
        // The names count and list revision the child applied, and the decoding it constructed.
        diagnostics: { modelConstructionMs: result.comparison.modelConstructionMs,
          ...(result.hotwords ? { names: result.hotwords.count, namesRevision: result.hotwords.revision } : {}),
          ...(result.comparison.decodingMethod ? { decodingMethod: result.comparison.decodingMethod } : {}) } };
    });
  }

  async release(): Promise<void> {
    if (this.#busy) throw new SetupWorkerError("CPU worker is busy");
    this.#ready = undefined;
    await this.#cpu.closeAndWait();
  }

  async close(): Promise<void> {
    this.#ready = undefined;
    const active = this.#finished;
    await this.#cpu.closeAndWait();
    if (active) {
      await active;
      await this.#cpu.closeAndWait();
    }
  }

  async #with<T>(signal: AbortSignal, operation: () => Promise<T>): Promise<T> {
    if (this.#busy) throw new SetupWorkerError("CPU worker is busy");
    if (signal.aborted) throw new SetupWorkerError("worker cancelled");
    this.#busy = true;
    let finish!: () => void;
    this.#finished = new Promise<void>((resolve) => { finish = resolve; });
    let retirement: Promise<void> | undefined;
    const cancel = () => {
      this.#ready = undefined;
      retirement = this.#cpu.closeAndWait();
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
      this.#ready = undefined;
      await this.#cpu.closeAndWait();
      throw signal.aborted ? new SetupWorkerError("worker cancelled") : error;
    } finally {
      try { await retirement; }
      finally {
        signal.removeEventListener("abort", cancel);
        this.#busy = false;
        this.#finished = undefined;
        finish();
      }
    }
  }
}
