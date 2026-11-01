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
  FinalToken,
  EmptyEvent,
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

// Concrete STT engines + the rewrite decorator.
export { SherpaEngine } from "./engines/sherpa.ts";
export { SherpaOfflineEngine } from "./engines/sherpa-offline.ts";
export { loadHotwords, HOTWORDS_FILENAME, type HotwordFileSnapshot } from "./engines/hotwords.ts";
export {
  createLocalSherpaEngine,
  loadSherpaBinding,
  SHERPA_ROOT,
} from "./engines/sherpa-binding.ts";
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
  type CartesiaOptions,
  type FetchLike,
} from "./tts/cartesia.ts";
export {
  OpenAITextToSpeech,
  DEFAULT_OPENAI_OUTPUT_CONFIG,
  DEFAULT_OPENAI_MODEL,
  DEFAULT_OPENAI_VOICE,
  OPENAI_SAMPLE_RATE,
  OPENAI_VOICES,
  type OpenAIOptions,
  type OpenAISessionOptions,
  type OpenAISocket,
  type OpenAISocketFactory,
  type OpenAILimits,
  type OpenAIAudioSegment,
  type OpenAISegmentObservation,
  type OpenAISpeechStream,
} from "./tts/openai.ts";
export { LocalSpeechOutputFactory, type LocalSpeechOutputOptions } from "./tts/factory.ts";
export {
  SpeechOutputFactory,
  type SpeechOutputContext,
  type SpeechOutputFactoryOptions,
} from "./tts/provider-factory.ts";
export {
  DEFAULT_LOCAL_OUTPUT,
  LOCAL_OUTPUT_MODELS,
  LOCAL_OUTPUT_REGISTRY,
  LOCAL_OUTPUT_SPEED,
  getLocalOutputCatalogue,
  getLocalOutputReadiness,
  resolveLocalOutput,
  type LocalOutputModel,
  type LocalOutputModelId,
  type LocalOutputReadiness,
} from "./tts/catalogue.ts";
export {
  CartesiaBatchTranscriber,
  CARTESIA_STT_MODEL,
  type CartesiaSttOptions,
  type CartesiaSttResult,
  type CartesiaSttMetadata,
  type CartesiaSttFetch,
} from "./engines/cartesia-stt.ts";
export {
  OpenAIFileTranscriber,
  OPENAI_FILE_MODEL,
  type OpenAIFileOptions,
  type OpenAIFileResult,
} from "./engines/openai-file.ts";
export type {
  SpeechInputConfig,
  SpeechOutputConfig,
  SpeechInputRegistry,
  SpeechOutputRegistry,
  SpeechTiming,
  TimingObservation,
} from "./provider-config.ts";
export { decodePcm16 } from "./pcm.ts";
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
export {
  pacedFrames, readWavFrames, decodeWavSamples, decodeWavFrames, encodeWavFloat32, type WavAudio,
} from "./bench/wav.ts";
export { upsertReportSection } from "./bench/report.ts";
export {
  DEFAULT_CORPUS_DIR,
  PROMPT_ID,
  readCorpusEntries,
  saveCorpusPair,
  type CorpusEntry,
  type CorpusSidecar,
  type CorpusSidecarMeta,
} from "./corpus/store.ts";

// Bench formatting + stats helpers, reused for latency reporting.
export { formatMs, formatOptionalMs } from "./bench/format.ts";
export { median, medianOptional } from "./bench/stats.ts";
