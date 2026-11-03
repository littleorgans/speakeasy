import * as ort from "onnxruntime-node";
import type { OrtBinding, OrtTensor } from "./ort-tdt-graphs.ts";

/**
 * The one place the offline engines build onnxruntime-node sessions (Granite
 * CTC, Parakeet TDT, Whistle): the package from this package's node_modules on
 * the CPU provider, with the caller's intra op threads and the options of its
 * named profile. Engines take a seam (`CtcRuntime`, `OrtBinding`,
 * `WhistleRuntime`) and never import the native package themselves.
 */

/**
 * Options each runtime adds to provider and threads. The profiles must not
 * merge: `reference` loads as the vendored Whistle `js/example.mjs` does, at
 * onnxruntime's default optimization level (not known to equal "all") and
 * log level ("warning"), so moving Whistle to `tuned` could change its graph
 * rewrites, its numerics and its stderr.
 */
export const SESSION_PROFILES = {
  /** Granite CTC and Parakeet TDT: every graph rewrite, errors only on stderr. */
  tuned: { graphOptimizationLevel: "all", logSeverityLevel: 3 },
  /** Whistle: onnxruntime defaults. */
  reference: {},
} as const satisfies Record<string, ort.InferenceSession.SessionOptions>;

export type SessionProfile = keyof typeof SESSION_PROFILES;

/** onnxruntime-node version, for engine labels. */
export const ORT_VERSION = ort.env.versions.node ?? "unknown";

export function sessionOptions(
  threads: number,
  profile: SessionProfile,
): ort.InferenceSession.SessionOptions {
  return { executionProviders: ["cpu"], intraOpNumThreads: threads, ...SESSION_PROFILES[profile] };
}

export function createCpuSession(
  path: string,
  threads: number,
  profile: SessionProfile,
): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(path, sessionOptions(threads, profile));
}

/** The session seam of OrtTdtEngine. */
export function loadOrtBinding(): OrtBinding {
  return {
    version: ORT_VERSION,
    createSession: async (path, { threads }) => {
      const session = await createCpuSession(path, threads, "tuned");
      return {
        run: (feeds) => session.run(feeds as ort.InferenceSession.FeedsType),
        release: () => session.release(),
      };
    },
    tensor: (type, data, dims): OrtTensor => new ort.Tensor(type, data, dims),
  };
}
