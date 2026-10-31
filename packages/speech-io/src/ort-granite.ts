import { join } from "node:path";
import { ortCtcRuntime } from "./engines/ort-ctc-runtime.ts";
import { OrtGraniteEngine } from "./engines/ort-granite.ts";
import type { GraniteModelId } from "./engines/ort-granite-models.ts";

/**
 * @speakeasy/speech-io/ort-granite: Granite Speech CTC on onnxruntime-node,
 * the bench's entry. Kept out of the `./stt` closure because it loads the
 * onnxruntime native; stt.test.ts holds that line.
 */

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

/** On-disk root for downloaded onnxruntime models. */
export const ORT_ROOT = join(process.cwd(), "models", "ort");

export function createLocalGraniteEngine(model: GraniteModelId): OrtGraniteEngine {
  return new OrtGraniteEngine({ model, modelRoot: ORT_ROOT, runtime: ortCtcRuntime });
}
