/**
 * Transcript correctness gates and word-level alignment for the bench harness.
 *
 * All comparison runs on the shared normalized form (see normalize.ts:
 * lowercase, strip punctuation, digit-canonical numbers), so cosmetic and
 * number-spelling differences never count as errors.
 *
 * Two gates exist:
 *   - exact: normalized string equality (endpoint sweep mode)
 *   - word-tolerant: word-level edit distance <= maxWordErrors (ptt mode,
 *     where quality is deprioritized). "and saw my fellow americans" passes
 *     (one substitution); a fragmented "and saw my fell ow a mericans" fails
 *     (one substitution plus insertions).
 */

import { normalizeTranscript } from "./normalize.ts";

export const DEFAULT_MAX_WORD_ERRORS = 1;

export { normalizeTranscript };

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

export type WordSubstitution = { expected: string; actual: string };

export type WordAlignment = {
  /** Word-level Levenshtein distance (substitutions + insertions + deletions). */
  errors: number;
  /** Word count of the normalized reference; the WER denominator. */
  referenceWords: number;
  substitutions: WordSubstitution[];
  /** Hypothesis words with no reference counterpart. */
  insertions: string[];
  /** Reference words missing from the hypothesis. */
  deletions: string[];
};

/**
 * Word-level Levenshtein distance between the normalized transcripts:
 * the minimum number of word substitutions, insertions, and deletions
 * needed to turn `text` into `expected`.
 */
export function wordErrorCount(text: string, expected: string): number {
  return wordErrorAlignment(text, expected).errors;
}

/**
 * Full word-level Levenshtein alignment: the distance plus the edits
 * themselves, backtracked from the DP matrix. Feeds per-utterance WER and
 * the corpus-wide worst-substitutions list.
 */
export function wordErrorAlignment(
  text: string,
  expected: string,
): WordAlignment {
  const actual = toWords(text);
  const reference = toWords(expected);
  const matrix = buildDistanceMatrix(actual, reference);
  return {
    errors: matrix[reference.length]![actual.length]!,
    referenceWords: reference.length,
    ...backtrackEdits(matrix, actual, reference),
  };
}

/** matrix[row][col] = distance between reference[0..row) and actual[0..col). */
function buildDistanceMatrix(
  actual: string[],
  reference: string[],
): number[][] {
  const matrix: number[][] = [
    Array.from({ length: actual.length + 1 }, (_, col) => col),
  ];
  for (let row = 1; row <= reference.length; row += 1) {
    const current: number[] = [row];
    for (let col = 1; col <= actual.length; col += 1) {
      const substitution =
        matrix[row - 1]![col - 1]! +
        (reference[row - 1] === actual[col - 1] ? 0 : 1);
      current.push(
        Math.min(substitution, matrix[row - 1]![col]! + 1, current[col - 1]! + 1),
      );
    }
    matrix.push(current);
  }
  return matrix;
}

function backtrackEdits(
  matrix: number[][],
  actual: string[],
  reference: string[],
): Pick<WordAlignment, "substitutions" | "insertions" | "deletions"> {
  const substitutions: WordSubstitution[] = [];
  const insertions: string[] = [];
  const deletions: string[] = [];
  let row = reference.length;
  let col = actual.length;

  while (row > 0 || col > 0) {
    const cost = matrix[row]![col]!;
    if (
      row > 0 &&
      col > 0 &&
      reference[row - 1] === actual[col - 1] &&
      cost === matrix[row - 1]![col - 1]!
    ) {
      row -= 1;
      col -= 1;
    } else if (row > 0 && col > 0 && cost === matrix[row - 1]![col - 1]! + 1) {
      substitutions.push({
        expected: reference[row - 1]!,
        actual: actual[col - 1]!,
      });
      row -= 1;
      col -= 1;
    } else if (col > 0 && cost === matrix[row]![col - 1]! + 1) {
      insertions.push(actual[col - 1]!);
      col -= 1;
    } else {
      deletions.push(reference[row - 1]!);
      row -= 1;
    }
  }

  substitutions.reverse();
  insertions.reverse();
  deletions.reverse();
  return { substitutions, insertions, deletions };
}

function toWords(text: string): string[] {
  const normalized = normalizeTranscript(text);
  return normalized ? normalized.split(" ") : [];
}
