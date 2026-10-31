/**
 * @speakeasy/speech-io/stt: the narrow speech to text entry for hosts that
 * vendor or bundle the engine (transport-matters).
 *
 * Its import closure stays free of TTS, capture, the onnxruntime Moonshine
 * engine and every native package: the sherpa binding is injected by the
 * host. stt.test.ts enforces the closure; keep additions inside engines/ and
 * the contract.
 */

export type {
  EmptyEvent,
  EndpointConfig,
  EndpointEvent,
  EndpointMode,
  FinalEvent,
  FinalToken,
  PartialEvent,
  STTConfig,
  STTErrorEvent,
  STTSession,
  VoiceToText,
} from "./contract.ts";

export {
  SherpaEngine,
  SHERPA_SAMPLE_RATE,
  type SherpaEngineOptions,
} from "./engines/sherpa.ts";
export { SherpaOfflineEngine } from "./engines/sherpa-offline.ts";
export {
  createSherpaEngine,
  type AnySherpaEngine,
} from "./engines/sherpa-factory.ts";
export type {
  Hotwords,
  RecognizerResult,
  SherpaBinding,
} from "./engines/sherpa-api.ts";
export { resultDetail, resultText, type ResultDetail } from "./engines/result.ts";

export {
  isSherpaModelId,
  resolveModelPaths,
  resolveSherpaModel,
  SHERPA_MODELS,
  type CanaryModel,
  type ModelArchive,
  type ModelFile,
  type ModelLicence,
  type MoonshineModel,
  type NemoTransducerModel,
  type OfflineSherpaModel,
  type Qwen3AsrModel,
  type SherpaModel,
  type SherpaModelId,
  type StreamingSherpaModel,
} from "./engines/sherpa-models.ts";

export {
  COMPLETE_MARKER,
  completeMarkerValue,
  ensureAsset,
  isAssetComplete,
  sha256File,
  type AssetFile,
  type AssetSpec,
  type FetchArchive,
  type FileSetSpec,
} from "./engines/assets.ts";
