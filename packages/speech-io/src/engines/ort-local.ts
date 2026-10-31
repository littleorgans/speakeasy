import { join } from "node:path";
import { ortCtcRuntime } from "./ort-ctc-runtime.ts";
import { OrtGraniteEngine } from "./ort-granite.ts";
import { isGraniteModelId } from "./ort-granite-models.ts";
import type { OrtModelId } from "./ort-models.ts";
import { OrtTdtEngine } from "./ort-tdt.ts";
import { isOrtTdtModelId, type OrtTdtModelId } from "./ort-tdt-models.ts";
import { loadOrtBinding } from "./ort-tdt-runtime.ts";
import { OrtWhistleEngine } from "./ort-whistle.ts";
import { loadWhistleRuntime } from "./ort-whistle-runtime.ts";

/**
 * speak-easy's own onnxruntime wiring for the CLI paths (bench, live tests):
 * each family's native runtime and models under the working directory.
 * Outside the `./stt` entry, like `sherpa-binding.ts`.
 */

/** On-disk root for downloaded onnxruntime-node models. */
export const ORT_ROOT = join(process.cwd(), "models", "ort");

export type AnyOrtEngine = OrtWhistleEngine | OrtGraniteEngine | OrtTdtEngine;

/** Build the engine for any `ORT_MODELS` id. */
export function createLocalOrtEngine(model: OrtTdtModelId): OrtTdtEngine;
export function createLocalOrtEngine(model: OrtModelId): AnyOrtEngine;
export function createLocalOrtEngine(model: OrtModelId): AnyOrtEngine {
  if (isGraniteModelId(model)) {
    return new OrtGraniteEngine({ model, modelRoot: ORT_ROOT, runtime: ortCtcRuntime });
  }
  if (isOrtTdtModelId(model)) {
    return new OrtTdtEngine({ model, modelRoot: ORT_ROOT, binding: loadOrtBinding() });
  }
  return new OrtWhistleEngine({ model, modelRoot: ORT_ROOT, runtime: loadWhistleRuntime() });
}
