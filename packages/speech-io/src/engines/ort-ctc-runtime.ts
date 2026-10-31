import * as ort from "onnxruntime-node";
import type { CtcLogits } from "./ctc-decode.ts";

/**
 * onnxruntime-node wiring for single graph CTC models: features in, logits
 * out. The only file of the CTC engines that loads the native runtime; engines
 * take a `CtcRuntime` so tests inject a fake and hosts choose where the
 * native comes from. CPU provider only.
 */

export type CtcGraph = {
  /** `features` is row major `[frames, featureDim]`, batch of one. */
  run(features: Float32Array, frames: number, featureDim: number): Promise<CtcLogits>;
};

export type CtcRuntime = {
  version: string;
  loadGraph(path: string): Promise<CtcGraph>;
};

const INPUT = "input_features";
const OUTPUT = "logits";

export const ortCtcRuntime: CtcRuntime = {
  version: ort.env.versions.common,
  async loadGraph(path) {
    const session = await ort.InferenceSession.create(path, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3,
    });
    return {
      async run(features, frames, featureDim) {
        const input = new ort.Tensor("float32", features, [1, frames, featureDim]);
        const logits = (await session.run({ [INPUT]: input }))[OUTPUT];
        const [batch, outFrames, vocab] = logits?.dims ?? [];
        if (!logits || logits.type !== "float32" || batch !== 1 || !outFrames || !vocab) {
          throw new Error(`CTC graph returned no float32 [1, frames, vocab] ${OUTPUT}`);
        }
        return { data: logits.data as Float32Array, frames: outFrames, vocab };
      },
    };
  },
};
