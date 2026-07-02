/**
 * Transcript normalization and correctness gates for the bench harness.
 *
 * Two gates exist:
 *   - exact: case/punct-insensitive string equality (endpoint sweep mode)
 *   - word-tolerant: word-level edit distance <= maxWordErrors (ptt mode,
 *     where quality is deprioritized). "and saw my fellow americans" passes
 *     (one substitution); a fragmented "and saw my fell ow a mericans" fails
 *     (one substitution plus insertions).
 */

export const DEFAULT_MAX_WORD_ERRORS = 1;

export function normalizeTranscript(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function isExactTranscript(text: string, expected: string): boolean {
  return normalizeTranscript(text) === normalizeTranscript(expected);
}

export function isWordTolerantTranscript(
  text: string,
  expected: string,
  maxWordErrors: number = DEFAULT_MAX_WORD_ERRORS,
): boolean {
  return wordErrorCount(text, expected) <= maxWordErrors;
}

/**
 * Word-level Levenshtein distance between the normalized transcripts:
 * the minimum number of word substitutions, insertions, and deletions
 * needed to turn `text` into `expected`.
 */
export function wordErrorCount(text: string, expected: string): number {
  const actual = toWords(text);
  const reference = toWords(expected);
  let previous = Array.from({ length: actual.length + 1 }, (_, col) => col);

  for (let row = 1; row <= reference.length; row += 1) {
    const current: number[] = [row];
    for (let col = 1; col <= actual.length; col += 1) {
      const substitution =
        previous[col - 1]! +
        (reference[row - 1] === actual[col - 1] ? 0 : 1);
      current.push(
        Math.min(substitution, previous[col]! + 1, current[col - 1]! + 1),
      );
    }
    previous = current;
  }

  return previous[actual.length]!;
}

function toWords(text: string): string[] {
  const normalized = normalizeTranscript(text);
  return normalized ? normalized.split(" ") : [];
}
