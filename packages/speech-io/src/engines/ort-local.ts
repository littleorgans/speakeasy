import { join } from "node:path";
import { ortCtcRuntime } from "./ort-ctc-runtime.ts";
import { OrtGraniteEngine } from "./ort-granite.ts";
import type { GraniteModelId } from "./ort-granite-models.ts";
import { resolveOrtModel, type OrtModel, type OrtModelId } from "./ort-models.ts";
import { OrtTdtEngine } from "./ort-tdt.ts";
import type { OrtTdtModelId } from "./ort-tdt-models.ts";
import { loadOrtBinding } from "./ort-runtime.ts";
import { OrtWhistleEngine } from "./ort-whistle.ts";
import type { WhistleModelId } from "./ort-whistle-models.ts";
import { loadWhistleRuntime } from "./ort-whistle-runtime.ts";
import type { SilenceGate, SilenceGateMode } from "./offline-session.ts";
import { localSilenceGate } from "./sherpa-binding.ts";

/**
 * speak-easy's own onnxruntime wiring for the CLI paths (bench, live tests):
 * each family's native runtime and models under the working directory.
 * Outside the `./stt` entry, like `sherpa-binding.ts`.
 */

/** On-disk root for downloaded onnxruntime-node models. */
export const ORT_ROOT = join(process.cwd(), "models", "ort");

export type AnyOrtEngine = OrtWhistleEngine | OrtGraniteEngine | OrtTdtEngine;

type LocalOptions = { modelRoot: string; silenceGate: SilenceGate | undefined };

/**
 * Each descriptor family's engine on its native runtime; a new family fails
 * to compile until it has one. The id casts hold because `family` comes from
 * the id's own descriptor, and each constructor validates the id again.
 */
const FAMILY_ENGINES = {
  "granite-ctc": (model, local) =>
    new OrtGraniteEngine({ ...local, model: model as GraniteModelId, runtime: ortCtcRuntime }),
  "parakeet-tdt": (model, local) =>
    new OrtTdtEngine({ ...local, model: model as OrtTdtModelId, binding: loadOrtBinding() }),
  whistle: (model, local) =>
    new OrtWhistleEngine({ ...local, model: model as WhistleModelId, runtime: loadWhistleRuntime() }),
} satisfies Record<OrtModel["family"], (model: OrtModelId, local: LocalOptions) => AnyOrtEngine>;

/**
 * Build the engine for any `ORT_MODELS` id, with the silence gate its
 * descriptor asks for unless `gate` overrides.
 */
export async function createLocalOrtEngine(
  model: OrtModelId,
  gate: SilenceGateMode = "model",
): Promise<AnyOrtEngine> {
  const descriptor = resolveOrtModel(model);
  const local = {
    modelRoot: ORT_ROOT,
    silenceGate: await localSilenceGate(descriptor.silenceGate, gate),
  };
  return FAMILY_ENGINES[descriptor.family](model, local);
}
