import { fork, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { serialize } from "node:v8";
import { SHERPA_MODELS } from "@speakeasy/speech-io/stt";

export const TRANSCRIBE_MODEL_ID = "en-parakeet-ultra-int8";
export const CPU_PRESET = Object.freeze({
  mode: "sweep/score-2", model: TRANSCRIBE_MODEL_ID,
  revision: "b86784a3a5356cbd126f710b61cd90c5fe5898f1",
  runtimeVersion: "2.0.1", runtimeGitSha1: "af2d8d81", onnxruntimeVersion: "1.28.2",
  hotwordsSha256: "0f5d821e73171e540a688d50c481007f58662609e1e7892b20829ba7092ea79b",
  vocabularySha256: "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17",
  provider: "cpu", threads: 2, featureDim: 80, sampleRate: 16_000,
  decodingMethod: "modified_beam_search", maxActivePaths: 4, hotwordsScore: 2,
  lengthReward: 0, maxSamples: 960_000, gate: "off", rewrite: "off", padding: false,
} as const);

export type TranscriptionMode = "legacy" | typeof CPU_PRESET.mode;
export type TakeHotwords = { path: string; count: number; revision: string };
export type CpuLoadMetadata = { loadMs: number; modelConstructionMs: number | null };
export type CpuComparisonTiming = CpuLoadMetadata & {
  processingMs: number | null; workerWallMs: number; firstUse: boolean; processStartupMs?: number;
};
export type Transcript = {
  text: string; decodeMs: number; model: string; hotwords?: TakeHotwords; comparison?: CpuComparisonTiming;
};
export type CpuReady = {
  type: "ready"; version: 1; preset: typeof CPU_PRESET;
  load: CpuLoadMetadata; assets: Record<string, string>;
};

export class TranscriptionError extends Error {
  readonly model: string;
  readonly hotwords: TakeHotwords | undefined;
  constructor(message: string, model: string, hotwords?: TakeHotwords) {
    super(message);
    this.model = model;
    this.hotwords = hotwords;
  }
}

export type Transcriber = {
  transcribe(samples: Float32Array): Promise<Transcript>;
  close(): void;
};
export type TranscribeRequest = { id: number; samples: Float32Array };
export type TranscribeReply =
  | { id: number; text: string; decodeMs: number; hotwords?: TakeHotwords; comparison?: CpuComparisonTiming }
  | { id: number; error: string; hotwords?: TakeHotwords };

const WORKER_PATH = fileURLToPath(new URL("./transcribe-worker.ts", import.meta.url));
const TRANSCRIBE_TIMEOUT_MS = 30_000;
type Pending = {
  resolve(transcript: Transcript): void; reject(error: Error): void; timer: NodeJS.Timeout;
};
type ReadyWait = {
  promise: Promise<CpuReady>; resolve(ready: CpuReady): void; reject(error: Error): void;
  timer: NodeJS.Timeout; started: number;
};

/** Each instance owns its isolated generation. Fixed mode requires ready before audio. */
export class ChildTranscriber implements Transcriber {
  readonly #workerPath: string;
  readonly #label: string;
  readonly #timeoutMs: number;
  readonly #startupTimeoutMs: number;
  readonly #hotwordsPath: string | undefined;
  readonly #modelRoot: string | undefined;
  readonly #mode: TranscriptionMode;
  readonly #pending = new Map<number, Pending>();
  #child: ChildProcess | undefined;
  #nextId = 0;
  #ready: ReadyWait | undefined;
  #readyValue: CpuReady | undefined;
  #startupMs = 0;
  #retiring: Promise<void> = Promise.resolve();
  #exit: Promise<void> = Promise.resolve();
  #lifecycle = 0;

  constructor(options: {
    workerPath?: string; label?: string; timeoutMs?: number; hotwordsPath?: string;
    mode?: TranscriptionMode; modelRoot?: string; startupTimeoutMs?: number;
  } = {}) {
    this.#workerPath = options.workerPath ?? WORKER_PATH;
    this.#label = options.label ?? TRANSCRIBE_MODEL_ID;
    this.#timeoutMs = options.timeoutMs ?? TRANSCRIBE_TIMEOUT_MS;
    this.#startupTimeoutMs = options.startupTimeoutMs ?? TRANSCRIBE_TIMEOUT_MS;
    this.#mode = options.mode ?? "legacy";
    if (this.#mode !== "legacy" && this.#mode !== CPU_PRESET.mode) throw new Error("Unknown CPU transcription mode");
    if (this.#mode === CPU_PRESET.mode && options.hotwordsPath !== undefined) {
      throw new Error("Fixed CPU mode cannot use an editable hotword file");
    }
    this.#hotwordsPath = options.hotwordsPath === undefined ? undefined : resolve(options.hotwordsPath);
    this.#modelRoot = options.modelRoot === undefined ? undefined : resolve(options.modelRoot);
    for (const duration of [this.#timeoutMs, this.#startupTimeoutMs]) {
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("CPU timeouts must be positive finite durations");
    }
  }

  async prepare(): Promise<CpuReady> {
    if (this.#mode !== CPU_PRESET.mode) throw new Error("Explicit prepare requires fixed CPU mode");
    const lifecycle = this.#lifecycle;
    await this.#retiring;
    if (lifecycle !== this.#lifecycle) throw new TranscriptionError("transcriber closed", this.#label);
    this.#spawn();
    return this.#ready!.promise;
  }

  async transcribe(samples: Float32Array): Promise<Transcript> {
    const lifecycle = this.#lifecycle;
    if (this.#mode === CPU_PRESET.mode) {
      validateCpuSamples(samples);
      // Own this exact view before cold preparation or retirement can yield.
      samples = samples.slice();
      await this.prepare();
    } else await this.#retiring;
    if (lifecycle !== this.#lifecycle) throw new TranscriptionError("transcriber closed", this.#label);
    const child = this.#spawn();
    const id = ++this.#nextId;
    return new Promise<Transcript>((resolve, reject) => {
      const timer = setTimeout(() => this.#retire(child,
        new Error(`transcription timed out after ${this.#timeoutMs} ms`)), this.#timeoutMs);
      this.#pending.set(id, { resolve, reject, timer });
      try {
        child.send({ id, samples } satisfies TranscribeRequest, (error) => {
          if (error) this.#retire(child, error);
        });
      } catch (error) { this.#retire(child, error instanceof Error ? error : new Error(String(error))); }
    });
  }

  close(): void {
    this.#lifecycle += 1;
    if (this.#child) this.#retire(this.#child, new Error("transcriber closed"));
  }

  async closeAndWait(): Promise<void> {
    this.close();
    await this.#retiring;
  }

  #spawn(): ChildProcess {
    if (this.#child) return this.#child;
    const started = performance.now();
    const args = this.#mode === CPU_PRESET.mode
      ? ["--cpu-preset=sweep/score-2", ...(this.#modelRoot ? [this.#modelRoot] : [])]
      : this.#hotwordsPath === undefined ? [] : [this.#hotwordsPath];
    const child = fork(this.#workerPath, args, {
      serialization: "advanced", stdio: ["ignore", "inherit", "inherit", "ipc"],
    });
    this.#child = child;
    this.#readyValue = undefined;
    this.#exit = new Promise<void>((resolve) => {
      child.once("exit", () => resolve());
      child.once("close", () => resolve());
    });
    if (this.#mode === CPU_PRESET.mode) {
      let accept!: (ready: CpuReady) => void;
      let reject!: (error: Error) => void;
      const promise = new Promise<CpuReady>((resolve, fail) => { accept = resolve; reject = fail; });
      this.#ready = { promise, resolve: accept, reject, started,
        timer: setTimeout(() => this.#retire(child, new Error("CPU worker ready timed out")), this.#startupTimeoutMs) };
    }
    child.on("message", (reply: unknown) => this.#message(child, reply));
    child.on("error", (error) => this.#retire(child, error));
    child.on("exit", (code, signal) => this.#retire(child, new Error(`transcriber exited (${signal ?? code})`)));
    return child;
  }

  #message(child: ChildProcess, value: unknown): void {
    if (this.#child !== child) return;
    try {
      if (serialize(value).byteLength > 65_536 || !record(value)) throw new Error("Invalid CPU worker reply");
      if (this.#mode === CPU_PRESET.mode && !this.#readyValue) {
        if (value.type === "startup-error" && typeof value.error === "string") throw new Error(value.error);
        validateCpuReady(value);
        this.#readyValue = value;
        this.#startupMs = performance.now() - this.#ready!.started;
        clearTimeout(this.#ready!.timer);
        this.#ready!.resolve(value);
        return;
      }
      if (!Number.isSafeInteger(value.id) || !this.#pending.has(value.id as number)) {
        throw new Error("CPU worker reply has an unknown request id");
      }
      if (value.hotwords !== undefined && !validHotwords(value.hotwords)) throw new Error("Invalid CPU hotword metadata");
      if ("error" in value) {
        if (typeof value.error !== "string") throw new Error("Invalid CPU worker error");
        this.#settle(value.id as number, new TranscriptionError(value.error, this.#label, value.hotwords as TakeHotwords | undefined));
      } else {
        if (typeof value.text !== "string" || !duration(value.decodeMs)) throw new Error("Invalid CPU transcript");
        if (this.#mode === CPU_PRESET.mode && (!validTiming(value.comparison) ||
            !validHotwords(value.hotwords) || value.hotwords.revision !== CPU_PRESET.hotwordsSha256 || value.hotwords.count !== 3)) {
          throw new Error("Invalid fixed CPU result metadata");
        }
        this.#settle(value.id as number, { text: value.text, decodeMs: value.decodeMs,
          ...(value.hotwords ? { hotwords: value.hotwords as TakeHotwords } : {}),
          ...(this.#mode === CPU_PRESET.mode ? {
            comparison: { ...value.comparison as CpuComparisonTiming, processStartupMs: this.#startupMs },
          } : {}) });
      }
    } catch (error) { this.#retire(child, error instanceof Error ? error : new Error(String(error))); }
  }

  #retire(child: ChildProcess, error: Error): void {
    if (this.#child !== child) return;
    this.#child = undefined;
    this.#lifecycle += 1;
    this.#retiring = this.#exit;
    if (this.#ready) {
      clearTimeout(this.#ready.timer);
      this.#ready.reject(error);
      this.#ready = undefined;
    }
    for (const id of [...this.#pending.keys()]) this.#settle(id, error);
    child.kill("SIGKILL");
  }

  #settle(id: number, outcome: Error | Omit<Transcript, "model">): void {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    if (outcome instanceof Error) pending.reject(outcome instanceof TranscriptionError
      ? outcome : new TranscriptionError(outcome.message, this.#label));
    else pending.resolve({ ...outcome, model: this.#label });
  }
}

export function validateCpuSamples(samples: Float32Array): void {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TranscriptionError("CPU comparison requires Float32 samples", TRANSCRIBE_MODEL_ID);
  }
  if (samples.length > CPU_PRESET.maxSamples) {
    throw new TranscriptionError("CPU comparison exceeds 60 seconds; audio is never truncated", TRANSCRIBE_MODEL_ID);
  }
  if (!samples.every(Number.isFinite)) {
    throw new TranscriptionError("CPU comparison samples must be finite", TRANSCRIBE_MODEL_ID);
  }
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function duration(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
function validLoad(value: unknown): value is CpuLoadMetadata & Record<string, unknown> {
  return record(value) && duration(value.loadMs) && (value.modelConstructionMs === null || duration(value.modelConstructionMs));
}
function validTiming(value: unknown): value is CpuComparisonTiming {
  return validLoad(value) && record(value) && duration(value.workerWallMs) && typeof value.firstUse === "boolean" &&
    (value.processingMs === null || duration(value.processingMs));
}
function validHotwords(value: unknown): value is TakeHotwords {
  return record(value) && typeof value.path === "string" && Number.isSafeInteger(value.count) &&
    (value.count as number) >= 0 && typeof value.revision === "string";
}

export function validateCpuReady(value: unknown): asserts value is CpuReady {
  if (!record(value) || value.type !== "ready" || value.version !== 1 || !record(value.preset) ||
      !validLoad(value.load) || value.load.modelConstructionMs === null || !record(value.assets)) {
    throw new Error("Invalid CPU worker ready metadata");
  }
  for (const [key, expected] of Object.entries(CPU_PRESET)) {
    if (value.preset[key] !== expected) throw new Error(`CPU ready preset mismatch: ${key}`);
  }
  for (const file of SHERPA_MODELS[TRANSCRIBE_MODEL_ID].files) {
    if (value.assets[file.name] !== file.sha256) throw new Error(`CPU ready asset mismatch: ${file.name}`);
  }
}
