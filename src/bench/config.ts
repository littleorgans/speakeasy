import type { EndpointConfig } from "../contract.ts";
import type { EngineName } from "./types.ts";

/** Shared bench configuration: defaults, thresholds, and the sweep grid. */

export const DEFAULT_RUNS = 5;
export const DEFAULT_ENGINE: EngineName = "stub";
export const DEFAULT_FRAME_MS = 20;
export const PASS_THRESHOLD_MS = 200;
export const SOFT_THRESHOLD_MS = 300;
export const SESSION_TIMEOUT_MS = 30_000;
/** How long the ptt harness waits for the final event after flush(). */
export const FLUSH_FINAL_TIMEOUT_MS = 10_000;
export const EXPECTED_JFK_TRANSCRIPT = "and so my fellow americans";
export const SHERPA_SWEEP_PATH = "results/sherpa-sweep.txt";
export const PARTIAL_ROOTCAUSE = "model-right-context";
export const PTT_MIN_RUNS = 6;
export const SHERPA_SWEEP: Required<EndpointConfig>[] = [
  { mode: "eager", minTrailingSilenceMs: 80, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 120, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 160, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 200, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 300, minUtteranceMs: 20_000 },
];
