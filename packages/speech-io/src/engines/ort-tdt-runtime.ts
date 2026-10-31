import * as ort from "onnxruntime-node";
import type { OrtBinding, OrtTensor } from "./ort-tdt-graphs.ts";

/**
 * onnxruntime-node wiring for OrtTdtEngine: the package from this package's
 * node_modules on the CPU provider.
 */

export function loadOrtBinding(): OrtBinding {
  return {
    version: ort.env.versions.node ?? "unknown",
    createSession: async (path, { threads }) => {
      const session = await ort.InferenceSession.create(path, {
        executionProviders: ["cpu"],
        intraOpNumThreads: threads,
        graphOptimizationLevel: "all",
        logSeverityLevel: 3,
      });
      return {
        run: (feeds) => session.run(feeds as ort.InferenceSession.FeedsType),
        release: () => session.release(),
      };
    },
    tensor: (type, data, dims): OrtTensor => new ort.Tensor(type, data, dims),
  };
}
