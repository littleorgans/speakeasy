/**
 * @speakeasy/speech-io/ort: every speech to text engine on onnxruntime-node,
 * for the bench. Kept outside `./stt` (which vendors only sherpa models)
 * because this closure loads the onnxruntime native package.
 */

export {
  createLocalOrtEngine,
  ORT_ROOT,
  type AnyOrtEngine,
} from "./engines/ort-local.ts";
export {
  isOrtModelId,
  ORT_MODELS,
  resolveOrtModel,
  type OrtModel,
  type OrtModelId,
} from "./engines/ort-models.ts";

export {
  OrtWhistleEngine,
  whistleDetail,
  type OrtWhistleEngineOptions,
  type WhistleFiles,
  type WhistleRecognizer,
  type WhistleRuntime,
} from "./engines/ort-whistle.ts";
export {
  isWhistleModelId,
  ORT_WHISTLE_MODELS,
  resolveWhistleModel,
  type WhistleModel,
  type WhistleModelId,
} from "./engines/ort-whistle-models.ts";
export { loadWhistleRuntime } from "./engines/ort-whistle-runtime.ts";

export { OrtGraniteEngine, type OrtGraniteEngineOptions } from "./engines/ort-granite.ts";
export {
  GRANITE_MODELS,
  isGraniteModelId,
  resolveGraniteFiles,
  resolveGraniteModel,
  type GraniteCtcModel,
  type GraniteModelId,
} from "./engines/ort-granite-models.ts";
export { ortCtcRuntime, type CtcGraph, type CtcRuntime } from "./engines/ort-ctc-runtime.ts";

export { OrtTdtEngine, type OrtTdtEngineOptions } from "./engines/ort-tdt.ts";
export {
  downloadBytes,
  isOrtTdtModelId,
  ORT_TDT_MODELS,
  resolveOrtTdtModel,
  type FusedTdtModel,
  type OrtTdtModel,
  type OrtTdtModelId,
  type SplitTdtModel,
} from "./engines/ort-tdt-models.ts";
export type { OrtBinding, OrtGraph, OrtTensor } from "./engines/ort-tdt-graphs.ts";
export { loadOrtBinding } from "./engines/ort-tdt-runtime.ts";

export { MoonshineEngine } from "./engines/moonshine.ts";
