import { isOrtModelId, ORT_MODELS, type OrtModelId } from "../engines/ort-models.ts";
import { isSherpaModelId, SHERPA_MODELS, type SherpaModelId } from "../engines/sherpa-models.ts";

/**
 * The bench's one view of every model it can build: sherpa-onnx registry ids
 * and onnxruntime registry ids. The id alone picks the runtime, so `--model`
 * and `--models` accept any of them.
 */

export type BenchModelId = SherpaModelId | OrtModelId;

export const BENCH_MODEL_IDS = [
  ...Object.keys(SHERPA_MODELS),
  ...Object.keys(ORT_MODELS),
] as readonly BenchModelId[];

export function isBenchModelId(value: string): value is BenchModelId {
  return isSherpaModelId(value) || isOrtModelId(value);
}

/** Parse a CLI model id, or throw with the valid set. */
export function parseBenchModelId(value: string): BenchModelId {
  if (isBenchModelId(value)) {
    return value;
  }
  throw new Error(`--model must be one of ${BENCH_MODEL_IDS.join(", ")}; received ${value}`);
}
