import { fork, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { serialize } from "node:v8";
import { SHERPA_MODELS } from "@speakeasy/speech-io/stt";

export const TRANSCRIBE_MODEL_ID = "en-parakeet-ultra-int8";
export const CPU_DECODING = {
  model: TRANSCRIBE_MODEL_ID, revision: "b86784a3a5356cbd126f710b61cd90c5fe5898f1",
  runtimeVersion: "2.0.1", runtimeGitSha1: "af2d8d81", onnxruntimeVersion: "1.28.2",
  // The BPE vocabulary derives from the token table alone, so every list shares it.
  vocabularySha256: "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17",
  provider: "cpu", threads: 2, featureDim: 80, sampleRate: 16_000,
  decodingMethod: "modified_beam_search", maxActivePaths: 4, hotwordsScore: 2,
  lengthReward: 0, maxSamples: 960_000, gate: "off", rewrite: "off", padding: false,
} as const;
/**
 * One sherpa CPU setup per road test row; each runs in its own child and biases
 * toward the Names list each take supplies, so the two differ only in id. Every
 * list, the empty one included, decodes with the pinned modified beam search.
 */
export const CPU_PRESETS = Object.freeze({
  "sweep/score-2": Object.freeze({ ...CPU_DECODING, mode: "sweep/score-2" }),
  "tm/today": Object.freeze({ ...CPU_DECODING, mode: "tm/today" }),
});
export type CpuSetupId = keyof typeof CPU_PRESETS;
export type CpuPreset = typeof CPU_PRESETS[CpuSetupId];
export const CPU_SETUP_IDS = Object.freeze(Object.keys(CPU_PRESETS) as CpuSetupId[]);
export function isCpuSetupId(value: unknown): value is CpuSetupId {
  return typeof value === "string" && Object.hasOwn(CPU_PRESETS, value);
}

export type TranscriptionMode = "legacy" | CpuSetupId;
export type TakeHotwords = { path: string; count: number; revision: string };
export type CpuLoadMetadata = { loadMs: number; modelConstructionMs: number | null };
export type CpuComparisonTiming = CpuLoadMetadata & {
  processingMs: number | null; workerWallMs: number; firstUse: boolean; processStartupMs?: number;
  /** The decoding the native recognizer was constructed with; a CPU child always reports it. */
  decodingMethod?: string;
};
export type Transcript = {
  text: string; decodeMs: number; model: string; hotwords?: TakeHotwords; comparison?: CpuComparisonTiming;
};
export type CpuReady = {
  type: "ready"; version: 1; preset: CpuPreset;
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
  transcribe(samples: Float32Array, names: readonly string[]): Promise<Transcript>;
  close(): void;
};
/** Every request carries its names; prepare builds the recognizer for them before a take. */
export type TranscribeRequest = { id: number; samples: Float32Array; names: readonly string[] };
export type PrepareRequest = { id: number; type: "prepare"; names: readonly string[] };
export type TranscribeReply =
  | { id: number; text: string; decodeMs: number; hotwords?: TakeHotwords; comparison?: CpuComparisonTiming }
  | { id: number; error: string; hotwords?: TakeHotwords }
  | CpuReady & { id: number };

const WORKER_PATH = fileURLToPath(new URL("./transcribe-worker.ts", import.meta.url));
const TRANSCRIBE_TIMEOUT_MS = 30_000;
/** A prepare waits for ready; a take waits for a transcript that matches the names it sent. */
type Wait = { reject(error: Error): void } & (
  | { kind: "prepare"; resolve(ready: CpuReady): void }
  | { kind: "transcribe"; resolve(transcript: Transcript): void; names: Pick<TakeHotwords, "count" | "revision"> });
type Pending = Wait & { timer: NodeJS.Timeout };

/**
 * Each instance owns its isolated generation. Fixed mode requires ready for a
 * take's names before its audio; ready for the same names is the same object.
 */
export class ChildTranscriber implements Transcriber {
  readonly #workerPath: string;
  readonly #label: string;
  readonly #timeoutMs: number;
  readonly #startupTimeoutMs: number;
  readonly #namesSource: string;
  readonly #modelRoot: string | undefined;
  readonly #preset: CpuPreset | undefined;
  readonly #pending = new Map<number, Pending>();
  #child: ChildProcess | undefined;
  #nextId = 0;
  /** The names this child last prepared, or is preparing, and their ready. */
  #prepared: { content: string; ready: Promise<CpuReady> } | undefined;
  #spawnedAt = 0;
  #startupMs: number | undefined;
  #retiring: Promise<void> = Promise.resolve();
  #exit: Promise<void> = Promise.resolve();
  #lifecycle = 0;

  constructor(options: {
    workerPath?: string; label?: string; timeoutMs?: number;
    /** Labels TakeHotwords.path: where the names came from. */
    namesSource?: string;
    mode?: TranscriptionMode; modelRoot?: string; startupTimeoutMs?: number;
  } = {}) {
    this.#workerPath = options.workerPath ?? WORKER_PATH;
    this.#label = options.label ?? TRANSCRIBE_MODEL_ID;
    this.#timeoutMs = options.timeoutMs ?? TRANSCRIBE_TIMEOUT_MS;
    this.#startupTimeoutMs = options.startupTimeoutMs ?? TRANSCRIBE_TIMEOUT_MS;
    const mode = options.mode ?? "legacy";
    if (mode !== "legacy" && !isCpuSetupId(mode)) throw new Error("Unknown CPU transcription mode");
    this.#preset = mode === "legacy" ? undefined : CPU_PRESETS[mode];
    this.#namesSource = options.namesSource ?? NAMES_SOURCE;
    this.#modelRoot = options.modelRoot === undefined ? undefined : resolve(options.modelRoot);
    for (const duration of [this.#timeoutMs, this.#startupTimeoutMs]) {
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("CPU timeouts must be positive finite durations");
    }
  }

  /** Builds the recognizer for these names; a names change rebuilds it in the same child. */
  async prepare(names: readonly string[]): Promise<CpuReady> {
    if (!this.#preset) throw new Error("Explicit prepare requires fixed CPU mode");
    const list = cpuNames(names);
    const lifecycle = this.#lifecycle;
    await this.#retiring;
    if (lifecycle !== this.#lifecycle) throw new TranscriptionError("transcriber closed", this.#label);
    const child = this.#spawn();
    const content = namesContent(list);
    if (this.#prepared?.content !== content) {
      const ready = new Promise<CpuReady>((resolve, reject) => {
        this.#send(child, { type: "prepare", names: list }, this.#startupTimeoutMs, "CPU worker ready timed out",
          { kind: "prepare", resolve, reject });
      });
      this.#prepared = { content, ready };
    }
    return this.#prepared.ready;
  }

  async transcribe(samples: Float32Array, names: readonly string[]): Promise<Transcript> {
    const lifecycle = this.#lifecycle;
    const list = cpuNames(names);
    if (this.#preset) {
      validateCpuSamples(samples);
      // Own this exact view before cold preparation or retirement can yield.
      samples = samples.slice();
      await this.prepare(list);
    } else await this.#retiring;
    if (lifecycle !== this.#lifecycle) throw new TranscriptionError("transcriber closed", this.#label);
    const child = this.#spawn();
    const expected = { count: list.length, revision: namesRevision(list) };
    return new Promise<Transcript>((resolve, reject) => {
      this.#send(child, { samples, names: list }, this.#timeoutMs, `transcription timed out after ${this.#timeoutMs} ms`,
        { kind: "transcribe", resolve, reject, names: expected });
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

  #send(child: ChildProcess, request: Omit<TranscribeRequest, "id"> | Omit<PrepareRequest, "id">, timeoutMs: number,
    timeout: string, wait: Wait): void {
    const id = ++this.#nextId;
    const timer = setTimeout(() => this.#retire(child, new Error(timeout)), timeoutMs);
    this.#pending.set(id, { ...wait, timer });
    try {
      child.send({ id, ...request }, (error) => {
        if (error) this.#retire(child, error);
      });
    } catch (error) { this.#retire(child, error instanceof Error ? error : new Error(String(error))); }
  }

  #spawn(): ChildProcess {
    if (this.#child) return this.#child;
    this.#spawnedAt = performance.now();
    this.#startupMs = undefined;
    const args = this.#preset ? [`--cpu-preset=${this.#preset.mode}`, ...(this.#modelRoot ? [this.#modelRoot] : [])] : [];
    const child = fork(this.#workerPath, args, {
      serialization: "advanced", stdio: ["ignore", "inherit", "inherit", "ipc"],
    });
    this.#child = child;
    this.#exit = new Promise<void>((resolve) => {
      child.once("exit", () => resolve());
      child.once("close", () => resolve());
    });
    child.on("message", (reply: unknown) => this.#message(child, reply));
    child.on("error", (error) => this.#retire(child, error));
    child.on("exit", (code, signal) => this.#retire(child, new Error(`transcriber exited (${signal ?? code})`)));
    return child;
  }

  #message(child: ChildProcess, value: unknown): void {
    if (this.#child !== child) return;
    try {
      if (serialize(value).byteLength > 65_536 || !record(value)) throw new Error("Invalid CPU worker reply");
      const { id, ...reply } = value;
      const pending = Number.isSafeInteger(id) ? this.#pending.get(id as number) : undefined;
      if (!pending) throw new Error("CPU worker reply has an unknown request id");
      if (reply.hotwords !== undefined && !validHotwords(reply.hotwords)) throw new Error("Invalid CPU hotword metadata");
      // The child labels nothing: the names' source is the parent's to name.
      const hotwords = reply.hotwords === undefined ? undefined : { ...reply.hotwords, path: this.#namesSource };
      if ("error" in reply) {
        if (typeof reply.error !== "string") throw new Error("Invalid CPU worker error");
        // A recognizer that cannot build for these names fails its child.
        if (pending.kind === "prepare") throw new Error(reply.error);
        this.#settle(id as number, new TranscriptionError(reply.error, this.#label, hotwords));
      } else if (pending.kind === "prepare") {
        validateCpuReady(reply, this.#preset!);
        this.#startupMs ??= performance.now() - this.#spawnedAt;
        this.#settle(id as number, reply);
      } else {
        if (typeof reply.text !== "string" || !duration(reply.decodeMs)) throw new Error("Invalid CPU transcript");
        if (this.#preset && (!validTiming(reply.comparison) || !hotwords ||
            reply.comparison.decodingMethod !== this.#preset.decodingMethod ||
            hotwords.revision !== pending.names.revision || hotwords.count !== pending.names.count)) {
          throw new Error("Invalid fixed CPU result metadata");
        }
        this.#settle(id as number, { text: reply.text, decodeMs: reply.decodeMs,
          ...(hotwords ? { hotwords } : {}),
          ...(this.#preset ? {
            comparison: { ...reply.comparison as CpuComparisonTiming, processStartupMs: this.#startupMs },
          } : {}) });
      }
    } catch (error) { this.#retire(child, error instanceof Error ? error : new Error(String(error))); }
  }

  #retire(child: ChildProcess, error: Error): void {
    if (this.#child !== child) return;
    this.#child = undefined;
    this.#lifecycle += 1;
    this.#retiring = this.#exit;
    this.#prepared = undefined;
    for (const id of [...this.#pending.keys()]) this.#settle(id, error);
    child.kill("SIGKILL");
  }

  #settle(id: number, outcome: Error | CpuReady | Omit<Transcript, "model">): void {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    if (outcome instanceof Error) pending.reject(outcome instanceof TranscriptionError
      ? outcome : new TranscriptionError(outcome.message, this.#label));
    else if (pending.kind === "prepare") pending.resolve(outcome as CpuReady);
    else pending.resolve({ ...outcome as Omit<Transcript, "model">, model: this.#label });
  }
}

export function validateCpuSamples(samples: Float32Array): void {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TranscriptionError("CPU comparison requires Float32 samples", TRANSCRIBE_MODEL_ID);
  }
  if (samples.length > CPU_DECODING.maxSamples) {
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
    (value.processingMs === null || duration(value.processingMs)) && typeof value.decodingMethod === "string";
}
function validHotwords(value: unknown): value is TakeHotwords {
  return record(value) && typeof value.path === "string" && Number.isSafeInteger(value.count) &&
    (value.count as number) >= 0 && typeof value.revision === "string";
}

/** TakeHotwords.path when the host names no source. */
export const NAMES_SOURCE = "names";

/**
 * The names a CPU child biases toward, one hotword line each, as both sides of
 * its IPC check them. The host has already canonicalized them (src/names.ts);
 * the tokenizer validates each when its recognizer is built.
 */
export function cpuNames(value: unknown): readonly string[] {
  if (!Array.isArray(value) || !value.every((name) => typeof name === "string" && name.length > 0 &&
      name === name.trim() && !/[\u0000-\u001f\u007f-\u009f]/.test(name))) {
    throw new TranscriptionError("CPU names must be trimmed, non blank strings without control characters",
      TRANSCRIBE_MODEL_ID);
  }
  return Object.freeze([...value]);
}
/** The hotword lines for these names; their sha256 is TakeHotwords.revision. */
export function namesContent(names: readonly string[]): string { return names.map((name) => `${name}\n`).join(""); }
export function namesRevision(names: readonly string[]): string {
  return createHash("sha256").update(namesContent(names)).digest("hex");
}

export function validateCpuReady(value: unknown, preset: CpuPreset): asserts value is CpuReady {
  if (!record(value) || value.type !== "ready" || value.version !== 1 || !record(value.preset) ||
      !validLoad(value.load) || value.load.modelConstructionMs === null || !record(value.assets)) {
    throw new Error("Invalid CPU worker ready metadata");
  }
  for (const [key, expected] of Object.entries(preset)) {
    if (value.preset[key] !== expected) throw new Error(`CPU ready preset mismatch: ${key}`);
  }
  for (const file of SHERPA_MODELS[TRANSCRIBE_MODEL_ID].files) {
    if (value.assets[file.name] !== file.sha256) throw new Error(`CPU ready asset mismatch: ${file.name}`);
  }
}
