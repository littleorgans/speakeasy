import type { FinalToken } from "../contract.ts";
import type { RecognizerResult } from "./sherpa-api.ts";

/**
 * Pure mapping from a sherpa recognizer result to committed text and tokens,
 * shared by the streaming and offline engines.
 *
 * Tokens keep their leading space, which marks a word start, so consumers can
 * group words. They are reported only when the model gives one timestamp per
 * token (Moonshine gives none); a log probability is attached only when the
 * model gives one per token, else null.
 *
 * Token times are `start_time + timestamps` on the stream's clock. A streaming
 * stream keeps its clock (and unprocessed flush padding) across reset, so the
 * caller passes the stream time its utterance began to get utterance offsets.
 */

const CONTROL_CHARACTERS = /\p{Cc}+/gu;

export type ResultDetail = {
  text: string;
  tokens: FinalToken[];
};

export function resultText(result: RecognizerResult): string {
  return stripControl(result.text ?? "").trim();
}

export function resultDetail(
  result: RecognizerResult,
  utteranceStartSeconds = 0,
): ResultDetail {
  const text = resultText(result);
  const tokens = result.tokens ?? [];
  const timestamps = result.timestamps ?? [];
  if (!text || tokens.length === 0 || timestamps.length !== tokens.length) {
    return { text, tokens: [] };
  }
  const logProbs = perToken(result.ys_log_probs ?? result.ys_probs, tokens.length);
  const offsetSeconds = (result.start_time ?? 0) - utteranceStartSeconds;
  return {
    text,
    tokens: tokens.map((token, index) => ({
      text: stripControl(token),
      startMs: Math.max(
        0,
        Math.round(((timestamps[index] ?? 0) + offsetSeconds) * 1_000),
      ),
      logProb: logProbs?.[index] ?? null,
    })),
  };
}

function perToken(
  values: number[] | undefined,
  count: number,
): number[] | undefined {
  return values?.length === count ? values : undefined;
}

function stripControl(value: string): string {
  return value.replace(CONTROL_CHARACTERS, "");
}
