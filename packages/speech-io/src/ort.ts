/**
 * @speakeasy/speech-io/ort: speech to text engines on onnxruntime-node, for
 * the bench. Kept outside `./stt` (which vendors only sherpa models) because
 * this closure loads the onnxruntime native package.
 */

export {
  OrtWhistleEngine,
  whistleDetail,
  type OrtWhistleEngineOptions,
  type WhistleFiles,
  type WhistleRecognizer,
  type WhistleRuntime,
} from "./engines/ort-whistle.ts";
export {
  ORT_WHISTLE_MODELS,
  resolveWhistleModel,
  type WhistleModel,
  type WhistleModelId,
} from "./engines/ort-whistle-models.ts";
export {
  createLocalWhistleEngine,
  loadWhistleRuntime,
  ORT_ROOT,
} from "./engines/ort-whistle-runtime.ts";
export {
  ensureFileAsset,
  isFileAssetComplete,
  type FileAssetSpec,
  type PinnedFile,
} from "./engines/file-assets.ts";
