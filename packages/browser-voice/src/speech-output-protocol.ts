import { parseSpeechOutputConfig, safeProviderError } from "@speakeasy/speech-io/providers";
import type { ProviderErrorCode, SpeechOutputConfig, SpeechOutputRegistry, SpeechTiming } from "@speakeasy/speech-io/providers";

export const MAX_SAMPLE_TEXT_LENGTH = 2_000;
export const DEFAULT_SAMPLE_TEXT = "The room is ready. Take your time, then tell me what you would like to try.";
const MAX_COMMAND_BYTES = 20_000;

export type SpeechOutputCommand =
  | { type: "select-output"; requestId: string; catalogueRevision: string; selection: SpeechOutputConfig }
  | { type: "speak-sample"; requestId: string; catalogueRevision: string; selectionRevision?: number;
      selection: SpeechOutputConfig; text: string }
  | { type: "cancel-sample"; requestId: string }
  | { type: "playback-drained"; playbackId: number; audioEndMs?: number };

export type SampleSegment = {
  index: number;
  sentence: string;
  sampleRate: number;
  audioDurationMs: number;
  timing: SpeechTiming;
};

/** Per provider request, with the adapter's boundaries preserved verbatim. */
export type SampleProviderObservation = {
  model: string;
  voice: string | number;
  text: string;
  returnedTranscript?: string | null;
  timing: Pick<SpeechTiming, "providerRequestWallMs" | "providerProcessingMs" | "providerNetworkMs">;
  sessionSetupWallMs?: SpeechTiming["providerRequestWallMs"];
  providerFirstAudioMs?: SpeechTiming["providerRequestWallMs"];
};

export type SpeechOutputEvent =
  | { type: "output-selected"; requestId: string; catalogueRevision: string;
      selectionRevision: number; selection: SpeechOutputConfig }
  | { type: "sample-accepted"; requestId: string; playbackId: number; generation: number;
      selection: SpeechOutputConfig; text: string; textId: string }
  | { type: "sample-phase"; requestId: string; playbackId: number; phase: "opening" | "producing" }
  | { type: "sample-complete"; requestId: string; playbackId: number;
      outcome: "produced" | "drained" | "cancelled"; selection: SpeechOutputConfig;
      textId: string; timing: SpeechTiming; segments: SampleSegment[]; requests: SampleProviderObservation[] }
  | { type: "sample-error"; requestId: string | null; playbackId?: number; code: SpeechOutputErrorCode; message: string }
  | { type: "playback"; requestId: string; action: "start" | "end" | "clear";
      playbackId: number; sampleRate?: number };

const MESSAGES = {
  COMMAND_INVALID: "Speech trial command is invalid.",
  TEXT_INVALID: "Enter nonblank trial text of at most 2000 characters.",
  REQUEST_REUSED: "Use a new request ID for each speech trial.",
  BUSY: "Audio work is active. Stop it before trying a voice.",
  TIMEOUT: "Speech trial timed out before playback completed.",
  AUDIO_INVALID: "Speech provider returned invalid audio.",
  SESSION_CLOSED: "Speech trial session is closed.",
} as const;

export type SpeechOutputErrorCode = keyof typeof MESSAGES | ProviderErrorCode;

export class SpeechOutputError extends Error {
  readonly code: keyof typeof MESSAGES;
  constructor(code: keyof typeof MESSAGES) {
    super(MESSAGES[code]);
    this.code = code;
    this.name = "SpeechOutputError";
  }
}

export function safeSpeechOutputError(error: unknown): { code: SpeechOutputErrorCode; message: string } {
  if (error instanceof SpeechOutputError) return { code: error.code, message: MESSAGES[error.code] };
  return safeProviderError(error).toJSON();
}

/** Only typed trial text has a command. No client source, session or grant fields. */
export function parseSpeechOutputCommand(raw: string, registry: SpeechOutputRegistry): SpeechOutputCommand {
  if (typeof raw !== "string" || Buffer.byteLength(raw) > MAX_COMMAND_BYTES) return invalid();
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return invalid(); }
  if (!record(value)) return invalid();
  if (value.type === "playback-drained") {
    if (!fields(value, ["type", "playbackId"], ["audioEndMs"]) ||
        !integer(value.playbackId, 0, 0xffff_ffff) ||
        (value.audioEndMs !== undefined && !integer(value.audioEndMs, 0))) return invalid();
    return { type: value.type, playbackId: value.playbackId,
      ...(value.audioEndMs === undefined ? {} : { audioEndMs: value.audioEndMs as number }) };
  }
  if (typeof value.requestId !== "string" || !/^[a-zA-Z0-9_-]{1,96}$/.test(value.requestId)) return invalid();
  if (value.type === "cancel-sample") {
    if (!fields(value, ["type", "requestId"])) return invalid();
    return { type: value.type, requestId: value.requestId };
  }
  if (value.type !== "select-output" && value.type !== "speak-sample") return invalid();
  const required = ["type", "requestId", "catalogueRevision", "selection"];
  if (value.type === "speak-sample") required.push("text");
  if (!fields(value, required, value.type === "speak-sample" ? ["selectionRevision"] : []) ||
      typeof value.catalogueRevision !== "string" || value.catalogueRevision.length > 128 ||
      !value.catalogueRevision.trim()) return invalid();
  const selection = parseSpeechOutputConfig(value.selection, registry);
  if (value.type === "select-output") return {
    type: value.type, requestId: value.requestId, catalogueRevision: value.catalogueRevision, selection,
  };
  if (typeof value.text !== "string" || !value.text.trim() || value.text.length > MAX_SAMPLE_TEXT_LENGTH) {
    throw new SpeechOutputError("TEXT_INVALID");
  }
  if (value.selectionRevision !== undefined && !integer(value.selectionRevision, 1)) return invalid();
  return { type: value.type, requestId: value.requestId, catalogueRevision: value.catalogueRevision,
    selection, text: value.text,
    ...(value.selectionRevision === undefined ? {} : { selectionRevision: value.selectionRevision as number }) };
}

function invalid(): never { throw new SpeechOutputError("COMMAND_INVALID"); }
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function fields(value: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  return required.every((key) => Object.hasOwn(value, key)) &&
    Object.keys(value).every((key) => required.includes(key) || optional.includes(key));
}
function integer(value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= min && value <= max;
}
