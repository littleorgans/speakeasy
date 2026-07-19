/// <reference path="./types/sherpa-onnx-node.d.ts" />
/**
 * @speakeasy/speech-io public surface.
 *
 * The contracts are the boundary other packages (convo-engine) build on; the
 * concrete engines, capture, player, and bench helpers are exported so callers
 * reuse one implementation instead of deep-importing or re-implementing. Nothing
 * here reaches past this file.
 */

// STT contract + rewrite types.
export type {
  STTConfig,
  STTSession,
  VoiceToText,
  EndpointConfig,
  EndpointMode,
  PartialEvent,
  FinalEvent,
  EndpointEvent,
  STTErrorEvent,
  RewriteRule,
  RewriteConfig,
  NumbersMode,
} from "./contract.ts";

// TTS contract.
export type {
  TTSConfig,
  TTSSession,
  TextToSpeech,
  AudioSegment,
} from "./tts/contract.ts";

// Concrete STT engine + the rewrite decorator.
export { SherpaEngine } from "./engines/sherpa.ts";
export { withRewrite } from "./rewrite/decorator.ts";
export { DEFAULT_RULES } from "./rewrite/rules.ts";
export {
  DEFAULT_SHERPA_MODEL,
  parseSherpaModelId,
  SHERPA_MODELS,
  type SherpaModelId,
} from "./engines/sherpa-models.ts";

// Concrete TTS engines + the streaming player.
export { SherpaTextToSpeech } from "./tts/sherpa-tts.ts";
export {
  CartesiaTextToSpeech,
  CARTESIA_VOICES,
  DEFAULT_CARTESIA_MODEL,
  DEFAULT_CARTESIA_VOICE,
  resolveCartesiaVoice,
} from "./tts/cartesia.ts";
export { createSegmentPlayer, type SegmentPlayer } from "./tts/player.ts";
export {
  parseTtsModelId,
  TTS_MODELS,
  type TtsModelId,
} from "./tts/models.ts";

// Microphone capture seam.
export {
  startMicCapture,
  resolveDefaultMicDevice,
  listAudioDevices,
  CAPTURE_SAMPLE_RATE,
  CAPTURE_FRAME_MS,
  DEFAULT_MIC_DEVICE,
  type MicCapture,
  type MicCaptureOptions,
  type AudioDevice,
} from "./capture/ffmpeg.ts";

// Recorded-audio helper (drives the mic-free live smoke).
export { readWavFrames, type WavAudio } from "./bench/wav.ts";

// Bench formatting + stats helpers, reused for latency reporting.
export { formatMs, formatOptionalMs } from "./bench/format.ts";
export { median, medianOptional } from "./bench/stats.ts";
