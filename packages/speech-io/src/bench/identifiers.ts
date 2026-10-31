import { alignWords } from "./transcript.ts";

/**
 * Identifier error rate: how often dictated paths, flags, versions and
 * numbers come back exactly as typed. A token containing `/`, `_`, `.` or a
 * digit is an identifier ("packages/speech-io", "TM_CHECK_ONLY=1", ".kilo",
 * "561").
 *
 * Scored on raw tokens, never the normalized transcript: `normalizeTranscript`
 * lowercases, splits on punctuation and folds number words to digits, so
 * "dev-up/down" and "dev up down" would collide. Raw tokens split on
 * whitespace and shed only sentence punctuation at their edges ("now." ->
 * "now", "gpt-6.1-sol." -> "gpt-6.1-sol"); a leading "." stays (".kilo").
 * Comparison ignores casing by default; caseSensitive retains the exact rate.
 * A reference identifier counts as
 * an error unless the raw word alignment matches it to an identical
 * hypothesis token.
 */

export type IdentifierScore = {
  /** Identifier tokens in the reference; the rate denominator. */
  identifiers: number;
  /** Reference identifiers the hypothesis did not reproduce exactly. */
  errors: number;
  /** The missed reference identifiers, in reference order. */
  missed: string[];
};

const IDENTIFIER_CHARACTER = /[/_.\d]/;
const LEADING_PUNCTUATION = /^["'([{]+/;
const TRAILING_PUNCTUATION = /[.,;:!?"')\]}]+$/;

export function isIdentifier(token: string): boolean {
  return IDENTIFIER_CHARACTER.test(token);
}

/** Whitespace tokens with edge sentence punctuation removed; empty tokens dropped. */
export function rawTokens(text: string): string[] {
  return text
    .split(/\s+/)
    .map((token) =>
      token.replace(LEADING_PUNCTUATION, "").replace(TRAILING_PUNCTUATION, ""),
    )
    .filter(Boolean);
}

export function scoreIdentifiers(
  hypothesis: string,
  expected: string,
  { caseSensitive = false }: { caseSensitive?: boolean } = {},
): IdentifierScore {
  const reference = rawTokens(expected);
  const comparisonTokens = (tokens: string[]) =>
    caseSensitive ? tokens : tokens.map((token) => token.toLowerCase());
  const matched = new Set(
    alignWords(comparisonTokens(rawTokens(hypothesis)), comparisonTokens(reference)).matchedReference,
  );
  const identifierIndices = reference
    .map((token, index) => (isIdentifier(token) ? index : -1))
    .filter((index) => index >= 0);
  const missed = identifierIndices
    .filter((index) => !matched.has(index))
    .map((index) => reference[index]!);
  return {
    identifiers: identifierIndices.length,
    errors: missed.length,
    missed,
  };
}
