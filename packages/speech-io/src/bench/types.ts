import type { EndpointConfig, NumbersMode, VoiceToText } from "../contract.ts";
import type { BenchModelId } from "./models.ts";

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
  /** Intra op threads of the engine's main graphs; absent when the runtime picks. */
  threads?: number;
  prepare?: () => Promise<void>;
};

export type CliOptions = {
  engine: EngineName;
  /** Registry id, sherpa-onnx or onnxruntime; needs --engine sherpa. */
  model?: BenchModelId;
  /** Corpus scorer: one column per registry id, scored on the same takes. */
  models?: BenchModelId[];
  /** Corpus scorer: add a rewrite column (domain rules + numbers). Default false. */
  rewrite: boolean;
  /** Corpus scorer: print the report as JSON on stdout. Default false. */
  json: boolean;
  /** Number normalization mode for the rewrite pipeline. Default "off". */
  numbers: NumbersMode;
  wav?: string;
  /** Corpus directory of wav + json sidecar pairs; enables the WER scorer. */
  corpus?: string;
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
  /** NaN when the engine never emitted final or empty after flush(). */
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
