/**
 * Text normalization for WER evaluation.
 *
 * Applied identically to hypothesis and reference before word alignment, so
 * cosmetic differences never count as recognition errors. Standard ASR eval
 * methodology (cf. Whisper's EnglishTextNormalizer): lowercase, strip
 * punctuation, and canonicalize spelled numbers to digits so "10" and "ten"
 * collapse to one token. Digits are the canonical direction; number words map
 * token-by-token, so genuine mishears ("tone", "tan") stay as errors.
 *
 * Scope: single number tokens (zero..twenty, the tens, hundred/thousand).
 * Compound composition ("twenty three" -> "23") is intentionally not handled;
 * the same transform hits both sides, so "twenty three" collapses to "20 3" on
 * each and still matches. The command corpus has no compound numerals.
 */

const NUMBER_WORDS: Record<string, string> = {
  zero: "0",
  one: "1",
  two: "2",
  three: "3",
  four: "4",
  five: "5",
  six: "6",
  seven: "7",
  eight: "8",
  nine: "9",
  ten: "10",
  eleven: "11",
  twelve: "12",
  thirteen: "13",
  fourteen: "14",
  fifteen: "15",
  sixteen: "16",
  seventeen: "17",
  eighteen: "18",
  nineteen: "19",
  twenty: "20",
  thirty: "30",
  forty: "40",
  fifty: "50",
  sixty: "60",
  seventy: "70",
  eighty: "80",
  ninety: "90",
  hundred: "100",
  thousand: "1000",
};

export function normalizeTranscript(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => NUMBER_WORDS[word] ?? word)
    .join(" ");
}
