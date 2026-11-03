import { hasModelId, lookupModel } from "./model.ts";
import { GRANITE_MODELS, type GraniteCtcModel } from "./ort-granite-models.ts";
import { ORT_TDT_MODELS, type OrtTdtModel } from "./ort-tdt-models.ts";
import { ORT_WHISTLE_MODELS, type WhistleModel } from "./ort-whistle-models.ts";
import { SHERPA_NUM_THREADS } from "./sherpa.ts";

/**
 * Every onnxruntime-node STT model in one view, keyed by id across the
 * family registries (Whistle, Granite CTC, Parakeet TDT); `family` names the
 * engine that runs it. Ids never collide with each other or with
 * `SHERPA_MODELS` (ort-models.test.ts).
 */

export const ORT_MODELS = {
  ...ORT_WHISTLE_MODELS,
  ...GRANITE_MODELS,
  ...ORT_TDT_MODELS,
} as const;

export type OrtModelId = keyof typeof ORT_MODELS;
export type OrtModel = WhistleModel | GraniteCtcModel | OrtTdtModel;

/**
 * Intra op threads for an engine's main graphs unless its descriptor sets
 * `threads`: sherpa's count, so bench timings compare like for like.
 */
export const ORT_THREADS = SHERPA_NUM_THREADS;

export function ortThreads(model: OrtModel): number {
  return model.threads ?? ORT_THREADS;
}

export function isOrtModelId(value: string): value is OrtModelId {
  return hasModelId(ORT_MODELS, value);
}

export function resolveOrtModel(id: string): OrtModel {
  return lookupModel(ORT_MODELS, id, "onnxruntime");
}
