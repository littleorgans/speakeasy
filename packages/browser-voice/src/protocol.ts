import {
  parseRealtimeVoice,
  type AudioSegment,
  type ConversationEvent,
  type RealtimeVoice,
} from "@speakeasy/convo-engine";
import { parseSpeechOutputConfig, type SpeechOutputConfig, type SpeechOutputRegistry } from "@speakeasy/speech-io/providers";
import { parseSpeechOutputCommand, type SpeechOutputCommand, type SpeechOutputErrorCode,
  type SpeechOutputEvent } from "./speech-output-protocol.ts";

export const INPUT_SAMPLE_RATE = 16_000;
export const AUDIO_PACKET_KIND = 1;
const AUDIO_HEADER_BYTES = 9;
const MAX_MIC_FRAME_SAMPLES = INPUT_SAMPLE_RATE;
export const MIN_PAUSE_MS = 200;
export const MAX_PAUSE_MS = 3_000;

export type ConversationMode = "natural" | "hold";
export const RESPONSE_ENGINES = [
  "realtime",
  "mercury-instant",
  "mercury-low",
] as const;
export type ResponseEngine = (typeof RESPONSE_ENGINES)[number];

export type ClientCommand =
  | {
      type: "start";
      engine: ResponseEngine;
      mode: ConversationMode;
      pauseMs: number;
      voice: RealtimeVoice;
      barge: boolean;
      systemPrompt?: string;
      speechOutput?: SpeechOutputConfig;
      catalogueRevision?: string;
      selectionRevision?: number;
    }
  | Extract<SpeechOutputCommand, { type: "select-output" }>
  | { type: "stop" }
  | { type: "interrupt" }
  | { type: "commit-input" }
  | { type: "playback-drained"; playbackId: number; audioEndMs?: number };

export type BrowserHostEvent =
  | ConversationEvent
  | Extract<SpeechOutputEvent, { type: "output-selected" }>
  | { type: "output-selection-error"; requestId: string | null; code: SpeechOutputErrorCode; message: string }
  | { type: "session"; phase: "loading" | "ready" | "stopped"; label?: string }
  | {
      type: "playback";
      action: "start" | "end" | "clear";
      playbackId: number;
      sampleRate?: number;
    };

export function parseClientCommand(raw: string, outputRegistry: SpeechOutputRegistry = []): ClientCommand {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("invalid JSON command");
  }
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new Error("command must have a type");
  }
  switch (value.type) {
    case "select-output": {
      const command = parseSpeechOutputCommand(raw, outputRegistry);
      if (command.type === "select-output") return command;
      throw new Error("room requires an output selection");
    }
    case "start": {
      if (Object.keys(value).some((key) => !["type", "engine", "mode", "pauseMs", "voice", "barge",
        "systemPrompt", "speechOutput", "catalogueRevision", "selectionRevision"].includes(key))) {
        throw new Error("start contains unsupported fields");
      }
      const engine = RESPONSE_ENGINES.find((candidate) => candidate === value.engine);
      if (engine === undefined) {
        throw new Error(`start requires engine: ${RESPONSE_ENGINES.join(", ")}`);
      }
      if (value.mode !== "natural" && value.mode !== "hold") {
        throw new Error('start requires mode "natural" or "hold"');
      }
      if (
        !Number.isInteger(value.pauseMs) ||
        Number(value.pauseMs) < MIN_PAUSE_MS ||
        Number(value.pauseMs) > MAX_PAUSE_MS
      ) {
        throw new Error(`start pauseMs must be ${MIN_PAUSE_MS} to ${MAX_PAUSE_MS}`);
      }
      if (typeof value.voice !== "string") {
        throw new Error("start requires a Realtime voice");
      }
      const speechOutput = value.speechOutput === undefined ? undefined :
        parseSpeechOutputConfig(value.speechOutput, outputRegistry);
      const hasRevision = value.catalogueRevision !== undefined || value.selectionRevision !== undefined;
      if (hasRevision && (typeof value.catalogueRevision !== "string" || !value.catalogueRevision.trim() ||
          value.catalogueRevision.length > 128 || !Number.isSafeInteger(value.selectionRevision) ||
          Number(value.selectionRevision) < 1)) {
        throw new Error("start requires paired catalogue and selection revisions");
      }
      if (engine !== "realtime" && speechOutput && speechOutput.provider.kind !== "local" && !hasRevision) {
        throw new Error("hosted output requires an explicit room selection");
      }
      return {
        type: "start",
        engine,
        mode: value.mode,
        pauseMs: Number(value.pauseMs),
        voice: parseRealtimeVoice(value.voice),
        barge: value.barge === true,
        ...(speechOutput === undefined ? {} : { speechOutput }),
        ...(hasRevision ? { catalogueRevision: value.catalogueRevision as string,
          selectionRevision: Number(value.selectionRevision) } : {}),
        ...(typeof value.systemPrompt === "string" && value.systemPrompt.trim()
          ? { systemPrompt: value.systemPrompt.trim() }
          : {}),
      };
    }
    case "stop":
    case "interrupt":
    case "commit-input":
      return { type: value.type };
    case "playback-drained":
      if (!Number.isSafeInteger(value.playbackId) || Number(value.playbackId) < 0) {
        throw new Error("playback-drained requires a non-negative playbackId");
      }
      if (
        value.audioEndMs !== undefined &&
        (!Number.isSafeInteger(value.audioEndMs) || Number(value.audioEndMs) < 0)
      ) {
        throw new Error("playback-drained audioEndMs must be a non-negative integer");
      }
      return {
        type: value.type,
        playbackId: Number(value.playbackId),
        ...(value.audioEndMs === undefined
          ? {}
          : { audioEndMs: Number(value.audioEndMs) }),
      };
    default:
      throw new Error(`unknown command "${value.type}"`);
  }
}

/** Raw browser Float32 PCM becomes one owned 16 kHz frame. */
export function decodeMicFrame(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength === 0 || bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("microphone frame must contain Float32 PCM");
  }
  const samples = bytes.byteLength / Float32Array.BYTES_PER_ELEMENT;
  if (samples > MAX_MIC_FRAME_SAMPLES) {
    throw new Error("microphone frame exceeds one second");
  }
  return decodePcmSamples(bytes);
}

/** Decode an exact byte view as finite little endian Float32 without clamping. */
export function decodePcmSamples(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength === 0 || bytes.byteLength % 4 !== 0) {
    throw new Error("body must be a non-empty whole number of Float32 samples");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frame = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < frame.length; index++) frame[index] = view.getFloat32(index * 4, true);
  for (const sample of frame) {
    if (!Number.isFinite(sample)) {
      throw new Error("microphone frame contains a non-finite sample");
    }
  }
  return frame;
}

/** Binary server packet: kind, playback id, sample rate, then Float32 PCM. */
export function encodeAudioPacket(playbackId: number, segment: AudioSegment): Uint8Array {
  const packet = new Uint8Array(
    AUDIO_HEADER_BYTES + segment.samples.byteLength,
  );
  const header = new DataView(packet.buffer);
  header.setUint8(0, AUDIO_PACKET_KIND);
  header.setUint32(1, playbackId, true);
  header.setUint32(5, segment.sampleRate, true);
  packet.set(
    new Uint8Array(
      segment.samples.buffer,
      segment.samples.byteOffset,
      segment.samples.byteLength,
    ),
    AUDIO_HEADER_BYTES,
  );
  return packet;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
