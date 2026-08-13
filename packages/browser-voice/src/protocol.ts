import {
  parseRealtimeVoice,
  type AudioSegment,
  type ConversationEvent,
  type RealtimeVoice,
} from "@speakeasy/convo-engine";

export const INPUT_SAMPLE_RATE = 16_000;
export const AUDIO_PACKET_KIND = 1;
const AUDIO_HEADER_BYTES = 9;
const MAX_MIC_FRAME_SAMPLES = INPUT_SAMPLE_RATE;
export const MIN_PAUSE_MS = 200;
export const MAX_PAUSE_MS = 3_000;

export type ConversationMode = "natural" | "hold";

export type ClientCommand =
  | {
      type: "start";
      mode: ConversationMode;
      pauseMs: number;
      voice: RealtimeVoice;
      barge: boolean;
      systemPrompt?: string;
    }
  | { type: "stop" }
  | { type: "interrupt" }
  | { type: "commit-input" }
  | { type: "playback-drained"; playbackId: number; audioEndMs?: number };

export type BrowserHostEvent =
  | ConversationEvent
  | { type: "session"; phase: "loading" | "ready" | "stopped"; label?: string }
  | {
      type: "playback";
      action: "start" | "end" | "clear";
      playbackId: number;
      sampleRate?: number;
    };

export function parseClientCommand(raw: string): ClientCommand {
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
    case "start":
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
      return {
        type: "start",
        mode: value.mode,
        pauseMs: Number(value.pauseMs),
        voice: parseRealtimeVoice(value.voice),
        barge: value.barge === true,
        ...(typeof value.systemPrompt === "string" && value.systemPrompt.trim()
          ? { systemPrompt: value.systemPrompt.trim() }
          : {}),
      };
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
  const owned = bytes.slice();
  const frame = new Float32Array(owned.buffer);
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
  return typeof value === "object" && value !== null;
}
