import { join } from "node:path";
import { ortCtcRuntime } from "./ort-ctc-runtime.ts";
import { OrtGraniteEngine } from "./ort-granite.ts";
import { isGraniteModelId } from "./ort-granite-models.ts";
import { resolveOrtModel, type OrtModelId } from "./ort-models.ts";
import { OrtTdtEngine } from "./ort-tdt.ts";
import { isOrtTdtModelId, type OrtTdtModelId } from "./ort-tdt-models.ts";
import { loadOrtBinding } from "./ort-tdt-runtime.ts";
import { OrtWhistleEngine } from "./ort-whistle.ts";
import { loadWhistleRuntime } from "./ort-whistle-runtime.ts";
import type { SilenceGateMode } from "./offline-session.ts";
import { localSilenceGate } from "./sherpa-binding.ts";

/**
 * speak-easy's own onnxruntime wiring for the CLI paths (bench, live tests):
 * each family's native runtime and models under the working directory.
 * Outside the `./stt` entry, like `sherpa-binding.ts`.
 */

/** On-disk root for downloaded onnxruntime-node models. */
export const ORT_ROOT = join(process.cwd(), "models", "ort");

export type AnyOrtEngine = OrtWhistleEngine | OrtGraniteEngine | OrtTdtEngine;

/**
 * Build the engine for any `ORT_MODELS` id, with the silence gate its
 * descriptor asks for unless `gate` overrides.
 */
export function createLocalOrtEngine(
  model: OrtTdtModelId,
  gate?: SilenceGateMode,
): Promise<OrtTdtEngine>;
export function createLocalOrtEngine(
  model: OrtModelId,
  gate?: SilenceGateMode,
): Promise<AnyOrtEngine>;
export async function createLocalOrtEngine(
  model: OrtModelId,
  gate: SilenceGateMode = "model",
): Promise<AnyOrtEngine> {
  const local = {
    modelRoot: ORT_ROOT,
    silenceGate: await localSilenceGate(resolveOrtModel(model).silenceGate, gate),
  };
  if (isGraniteModelId(model)) {
    return new OrtGraniteEngine({ ...local, model, runtime: ortCtcRuntime });
  }
  if (isOrtTdtModelId(model)) {
    return new OrtTdtEngine({ ...local, model, binding: loadOrtBinding() });
  }
  return new OrtWhistleEngine({ ...local, model, runtime: loadWhistleRuntime() });
}
