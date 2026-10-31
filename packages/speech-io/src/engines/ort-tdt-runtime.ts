import { join } from "node:path";
import * as ort from "onnxruntime-node";
import type { OrtBinding, OrtTensor } from "./ort-tdt-graphs.ts";
import { OrtTdtEngine } from "./ort-tdt.ts";
import type { OrtTdtModelId } from "./ort-tdt-models.ts";

/**
 * speak-easy's own onnxruntime-node wiring for OrtTdtEngine on the bench and
 * CLI paths: the package from this package's node_modules on the CPU
 * provider, and models under the working directory.
 */

/**
 * On-disk root for downloaded onnxruntime-node models. Same folder as the
 * Whistle engine's `ORT_ROOT`; integration keeps one constant.
 */
export const ORT_TDT_ROOT = join(process.cwd(), "models", "ort");

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

export function createLocalOrtTdtEngine(model: OrtTdtModelId): OrtTdtEngine {
  return new OrtTdtEngine({ model, modelRoot: ORT_TDT_ROOT, binding: loadOrtBinding() });
}
