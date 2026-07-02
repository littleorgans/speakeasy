import type { EndpointConfig, VoiceToText } from "../contract.ts";

/** Shared bench harness types. */

export type EngineName = "stub" | "moonshine" | "sherpa";

export type BenchMode = "sweep" | "ptt";

/**
 * Release-point variant for ptt mode. Strict releases at the last voiced
 * sample, so no real trailing audio is fed after release; it is the honest
 * push-to-talk headline. Loose releases at the RMS reference including the
 * hangover, matching the sweep mode's speech-end reference.
 */
export type PttVariant = "strict" | "loose";

export type BenchEngine = VoiceToText & {
  label?: string;
  prepare?: () => Promise<void>;
};

export type CliOptions = {
  engine: EngineName;
  wav?: string;
  runs: number;
  frameMs: number;
  mode: BenchMode;
};

export type FinalObservation = {
  endpointAt: number;
  finalAt: number;
  text: string;
};

export type RunResult = {
  run: number;
  firstPartialMs?: number;
  endpointToFinalMs: number;
  speechEndToFinalMs: number;
  endpointDelayMs: number;
  wallMs: number;
  finalText: string;
  finalizedAfterSpeechEnd: boolean;
  textCorrect: boolean;
};

export type Summary = {
  engineLabel: string;
  endpoint?: Required<EndpointConfig>;
  results: RunResult[];
  endpointToFinalMedian: number;
  speechEndToFinalMedian: number;
  firstPartialMedian?: number;
  firstPartialColdMs?: number;
  firstPartialWarmMedian?: number;
  textCorrect: boolean;
  passFail: "PASS" | "FAIL";
  ok: boolean;
};

export type PttRunResult = {
  run: number;
  /** NaN when the engine never emitted a final after flush(). */
  flushToFinalMs: number;
  firstPartialMs?: number;
  wordErrors: number;
  finalText: string;
  textCorrect: boolean;
};

export type PttSummary = {
  engineLabel: string;
  variant: PttVariant;
  /** Release point in ms from audio start for this variant. */
  releaseMs: number;
  results: PttRunResult[];
  coldFlushToFinalMs: number;
  warmFlushToFinalMedianMs: number;
  textCorrect: boolean;
  passFail: "PASS" | "FAIL";
};
