import { createHash } from "node:crypto";

export const SETUP_IDS = [
  "mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy", "qwen/context",
  "sweep/score-2", "mlx-cleanup/qwen-targeted",
] as const;
export type SetupId = typeof SETUP_IDS[number];
export const DEFAULT_SETUP_ID: SetupId = SETUP_IDS[0];
export const ULTRA_SETUP_ID: SetupId = "mlx-parakeet/ultra-greedy";
export const CLEANUP_SETUP_ID = "mlx-cleanup/qwen-targeted" as const;
export const SETUP_PROTOCOL_VERSION = 1;
export const SETUP_SAMPLE_RATE = 16_000;
export const MAX_PCM_BYTES = 7_680_000;
export const MAX_HEADER_BYTES = 65_536;
// JSON escaping can expand each input byte sixfold; leave room for metadata.
export const MAX_TEXT_BYTES = 8_192;
export const IDLE_RELEASE_MS = 5 * 60_000;

export type PcmIdentity = Readonly<{
  sampleRate: 16000; channels: 1; sampleCount: number; pcmBytes: number; pcmSha256: string;
}>;
/** Owned by the scheduler. Adapters must treat these views as read only. */
export type CanonicalPcm = PcmIdentity & Readonly<{ samples: Float32Array; bytes: Uint8Array }>;

export type SetupExecution =
  | Readonly<{ operation: "transcribe"; requestId: string; setupId: Exclude<SetupId, typeof CLEANUP_SETUP_ID>;
      pcm: CanonicalPcm }>
  | Readonly<{ operation: "cleanup"; requestId: string; setupId: typeof CLEANUP_SETUP_ID;
      base: PcmIdentity; baseTextSha256: string; A: string; B: null }>;
export type SetupLoadResult = { startupMs: number | null; loadMs: number | null; firstUse: boolean };
export type SetupExecutionResult = {
  text: string; processingMs: number | null; workerWallMs: number | null;
  rawText?: string; decodeMs?: number | null; diagnostics?: Record<string, unknown>;
};
export type SetupExecutionContext = Readonly<{ signal: AbortSignal }>;

/**
 * load and execute must settle only after a cancelled/failed generation exits.
 * Separate startup/load and operation deadlines belong to the transport.
 * release closes only an idle setup and settles after exit, including rejection.
 * close awaits every owned child exiting, including when another close fails.
 * No method may warm up, retry, or restart a cancelled request.
 */
export interface SetupExecutor {
  load(setupId: SetupId, context: SetupExecutionContext): Promise<SetupLoadResult>;
  execute(request: SetupExecution, context: SetupExecutionContext): Promise<SetupExecutionResult>;
  release(setupId: SetupId): Promise<void>;
  close(): Promise<void>;
}

export type SetupFailureCode = "limit" | "unavailable" | "load" | "execution" | "protocol" | "cancelled";
export type SetupFailure = { code: SetupFailureCode; message: string; rawText?: string; rawTextTruncated?: true };
export type SetupTimings = {
  queueMs: number | null; startupMs: number | null; loadMs: number | null;
  firstUse: boolean | null; processingMs: number | null; workerWallMs: number | null;
  requestWallMs: number | null; serviceWallMs: number | null;
  dependencyProcessingMs: number | null; deliveryMs: null;
};
export type SetupRowStatus = "queued" | "loading" | "processing" | "complete" | "error" | "skipped" | "cancelled";
export type SetupRow = {
  setupId: SetupId; status: SetupRowStatus; text: string | null; rawText?: string; decodeMs?: number | null;
  pcm: PcmIdentity; timings: SetupTimings; error?: SetupFailure; dependency?: typeof ULTRA_SETUP_ID;
  activeSetupId?: SetupId;
};
export type ComparisonEvent =
  | { type: "accepted"; comparisonId: string; queued: boolean; setupIds: readonly SetupId[]; pcm: PcmIdentity }
  | { type: "started"; comparisonId: string; queueMs: number }
  | { type: "row"; comparisonId: string; row: SetupRow }
  | { type: "done"; comparisonId: string; status: "complete" | "cancelled"; wallMs: number };
export type ComparisonResult = {
  comparisonId: string; status: "complete" | "cancelled"; rows: readonly SetupRow[]; wallMs: number;
};
export type ComparisonHandle = { comparisonId: string; done: Promise<ComparisonResult>; cancel(): void };

/** Binary protocol: uint32 LE header length, UTF8 JSON, then pcmBytes bytes. */
export type SetupWorkerRequest = {
  version: 1; generation: string; requestId: string; setupId: SetupId;
  operation: "transcribe" | "cleanup"; sampleRate: 16000; channels: 1;
  sampleCount: number; pcmBytes: number; pcmSha256: string;
  A?: string; B?: null; base?: PcmIdentity; baseTextSha256?: string;
};
export type SetupWorkerReply =
  | { type: "ready"; version: 1; generation: string; setupId: SetupId;
      identity: Record<string, unknown>; loadMs: number | null }
  | { type: "result"; version: 1; generation: string; requestId: string; setupId: SetupId;
      text: string; processingMs: number | null; workerWallMs: number | null;
      rawText?: string; decodeMs?: number | null; diagnostics?: Record<string, unknown> }
  | { type: "error"; version: 1; generation: string; requestId: string; setupId: SetupId;
      error: string; rawText?: string }
  | { type: "closed"; version: 1; generation: string; setupId: SetupId };

export function isSetupId(value: unknown): value is SetupId {
  return typeof value === "string" && (SETUP_IDS as readonly string[]).includes(value);
}

export function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function pcmIdentity(pcm: PcmIdentity): PcmIdentity {
  return Object.freeze({ sampleRate: pcm.sampleRate, channels: pcm.channels, sampleCount: pcm.sampleCount,
    pcmBytes: pcm.pcmBytes, pcmSha256: pcm.pcmSha256 });
}

/** Copies exactly one take, preserving finite Float32 bits, offset views and signed zero. */
export function createCanonicalPcm(input: Float32Array | Uint8Array): CanonicalPcm {
  if (!(input instanceof Float32Array) && !(input instanceof Uint8Array)) throw new Error("expected Float32 PCM");
  if (!input.byteLength || input.byteLength > MAX_PCM_BYTES || input.byteLength % 4 !== 0) {
    throw new Error("PCM must contain 1 to 1920000 whole Float32 samples");
  }
  const bytes = new Uint8Array(input.byteLength);
  const view = new DataView(bytes.buffer);
  if (input instanceof Float32Array) {
    input.forEach((sample, index) => view.setFloat32(index * 4, sample, true));
  } else bytes.set(input);
  for (let offset = 0; offset < bytes.length; offset += 4) {
    if (!Number.isFinite(view.getFloat32(offset, true))) throw new Error("PCM samples must be finite");
  }
  return Object.freeze({ samples: new Float32Array(bytes.buffer), bytes, sampleRate: SETUP_SAMPLE_RATE,
    channels: 1, sampleCount: bytes.length / 4, pcmBytes: bytes.length, pcmSha256: sha256(bytes) });
}

export function assertBoundedText(value: unknown, name = "text"): asserts value is string {
  if (typeof value !== "string" || Buffer.byteLength(value, "utf8") > MAX_TEXT_BYTES) {
    throw new Error(`${name} exceeds the UTF8 text bound or is not a string`);
  }
}

export function isDuration(value: unknown): value is number | null {
  return value === null || typeof value === "number" && Number.isFinite(value) && value >= 0;
}

export function assertSetupLoadResult(value: unknown): asserts value is SetupLoadResult {
  if (!isRecord(value) || !isDuration(value.startupMs) || !isDuration(value.loadMs)
    || typeof value.firstUse !== "boolean") throw new Error("invalid setup load metadata");
}

export function assertSetupExecutionResult(value: unknown): asserts value is SetupExecutionResult {
  if (!isRecord(value) || !isDuration(value.processingMs) || !isDuration(value.workerWallMs)
    || value.decodeMs !== undefined && !isDuration(value.decodeMs)
    || value.diagnostics !== undefined && !isRecord(value.diagnostics)) throw new Error("invalid setup result metadata");
  assertBoundedText(value.text);
  if (value.rawText !== undefined) assertBoundedText(value.rawText, "rawText");
  if (Buffer.byteLength(JSON.stringify(value), "utf8") > MAX_HEADER_BYTES) throw new Error("setup result exceeds bound");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
