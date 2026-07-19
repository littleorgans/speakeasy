import { formatMs, median } from "@speakeasy/speech-io";

/**
 * Per-turn latency instrumentation. Every interval is anchored at the endpoint
 * (end of user speech), so the numbers answer one question directly: how long
 * after the user stops talking does each stage land. The headline is
 * endpoint -> first audio: end-of-user-speech to first spoken word.
 */

export type TurnTimestamps = {
  /** End of user speech (STT endpoint fired). */
  endpointAt: number;
  /** Committed STT transcript ready. */
  finalAt: number;
  /** First LLM token received. */
  firstTokenAt: number;
  /** First TTS audio segment ready. */
  firstAudioAt: number;
};

export type TurnMetrics = {
  turn: number;
  transcript: string;
  endpointToFinalMs: number;
  endpointToFirstTokenMs: number;
  /** Headline: end-of-user-speech to first spoken word. */
  endpointToFirstAudioMs: number;
  tokenCount: number;
  spokenMs: number;
};

export function buildTurnMetrics(
  turn: number,
  transcript: string,
  ts: TurnTimestamps,
  tokenCount: number,
  spokenMs: number,
): TurnMetrics {
  return {
    turn,
    transcript,
    endpointToFinalMs: ts.finalAt - ts.endpointAt,
    endpointToFirstTokenMs: ts.firstTokenAt - ts.endpointAt,
    endpointToFirstAudioMs: ts.firstAudioAt - ts.endpointAt,
    tokenCount,
    spokenMs,
  };
}

/** One line per turn, printed live as the turn completes. */
export function formatTurnLine(metrics: TurnMetrics): string {
  return [
    `turn ${metrics.turn}`,
    `stt-final=${formatMs(metrics.endpointToFinalMs)}`,
    `first-token=${formatMs(metrics.endpointToFirstTokenMs)}`,
    `first-audio=${formatMs(metrics.endpointToFirstAudioMs)}`,
    `tokens=${metrics.tokenCount}`,
    `spoken=${formatMs(metrics.spokenMs)}`,
  ].join(" | ");
}

/** End-of-session median table across all recorded turns. */
export function formatSessionSummary(turns: TurnMetrics[]): string {
  if (turns.length === 0) {
    return "session summary: no completed turns";
  }
  const medianOf = (pick: (m: TurnMetrics) => number): string =>
    formatMs(median(turns.map(pick)));
  return [
    `session summary: ${turns.length} turn(s), medians:`,
    `  endpoint->stt-final   ${medianOf((m) => m.endpointToFinalMs)}`,
    `  endpoint->first-token ${medianOf((m) => m.endpointToFirstTokenMs)}`,
    `  endpoint->first-audio ${medianOf((m) => m.endpointToFirstAudioMs)}  (headline)`,
  ].join("\n");
}
