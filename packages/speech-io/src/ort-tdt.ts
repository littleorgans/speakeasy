/**
 * @speakeasy/speech-io/ort-tdt: Parakeet TDT engines on onnxruntime-node, for
 * the bench. Kept outside `./stt` (which vendors only sherpa models) because
 * this closure loads the onnxruntime native package.
 */

export {
  OrtTdtEngine,
  type OrtTdtEngineOptions,
} from "./engines/ort-tdt.ts";
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
export type {
  OrtBinding,
  OrtGraph,
  OrtTensor,
} from "./engines/ort-tdt-graphs.ts";
export {
  createLocalOrtTdtEngine,
  loadOrtBinding,
  ORT_TDT_ROOT,
} from "./engines/ort-tdt-runtime.ts";
export {
  ensureFileAsset,
  isFileAssetComplete,
  type FileAssetSpec,
  type PinnedFile,
} from "./engines/file-assets.ts";
