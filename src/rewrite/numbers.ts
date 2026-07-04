import type { NumbersMode } from "../contract.ts";

/**
 * Number normalization (rewrite pipeline stage 2), config-gated.
 *
 * "digits": spelled numbers -> digits ("ten" -> "10", "twenty three" -> "23").
 * "words":  digit numbers -> spelled ("10" -> "ten", "23" -> "twenty three").
 * "off":    passthrough (default; behavior unchanged unless enabled).
 *
 * Correct for integers 0..999; larger values and leading-zero tokens pass
 * through unchanged. Idempotent, and non-number text is untouched (only exact
 * number words / digit runs are matched, so "tone" is never mistaken for "ten").
 * A whitespace-separated run compounds ("twenty three" -> 23) but a sequence of
 * bare units stays separate ("one two three" -> "1 2 3"), matching how the two
 * are spoken.
 */

const ONES: Record<string, number> = {
  zero: 0, one: 1, two: 2, three: 3, four: 4,
  five: 5, six: 6, seven: 7, eight: 8, nine: 9,
};
const TEENS: Record<string, number> = {
  ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14,
  fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19,
};
const TENS: Record<string, number> = {
  twenty: 20, thirty: 30, forty: 40, fifty: 50,
  sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};

const ONES_WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const TEENS_WORD = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS_WORD = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

type Kind = "ones" | "teen" | "tens" | "hundred";

const NUMBER_WORDS = [
  ...Object.keys(TEENS), // teens before "ten"-prefixed shorter tokens is moot; sorted below
  ...Object.keys(TENS),
  ...Object.keys(ONES),
  "hundred",
].sort((a, b) => b.length - a.length);

const RUN_RE = new RegExp(
  `\\b(?:${NUMBER_WORDS.join("|")})(?:\\s+(?:${NUMBER_WORDS.join("|")}))*\\b`,
  "gi",
);

/** Validate a CLI numbers-mode value (shared by demo and bench). */
export function parseNumbersMode(value: string): NumbersMode {
  if (value === "digits" || value === "words" || value === "off") {
    return value;
  }
  throw new Error(`--numbers must be digits, words, or off; received ${value}`);
}

export function applyNumbers(text: string, mode: NumbersMode): string {
  if (mode === "digits") {
    return text.replace(RUN_RE, (run) =>
      segment(run.split(/\s+/)).join(" "),
    );
  }
  if (mode === "words") {
    return text.replace(/\b\d+\b/g, (token) => {
      const value = Number(token);
      return value <= 999 && String(value) === token
        ? numberToWords(value)
        : token;
    });
  }
  return text;
}

function classify(word: string): { kind: Kind; value: number } {
  if (word === "hundred") return { kind: "hundred", value: 100 };
  if (word in TEENS) return { kind: "teen", value: TEENS[word]! };
  if (word in TENS) return { kind: "tens", value: TENS[word]! };
  return { kind: "ones", value: ONES[word]! };
}

/** True when `next` continues the number being built rather than starting one. */
function canExtend(last: Kind, next: Kind): boolean {
  if (next === "hundred") return last === "ones";
  if (next === "tens") return last === "hundred";
  if (next === "ones") return last === "tens" || last === "hundred";
  return last === "hundred"; // teen
}

/** Segment a whitespace run of number words into the integers it spells. */
function segment(tokens: string[]): number[] {
  const out: number[] = [];
  let current = 0;
  let last: Kind | null = null;
  for (const token of tokens) {
    const { kind, value } = classify(token.toLowerCase());
    if (last === null) {
      current = value;
    } else if (canExtend(last, kind)) {
      current = kind === "hundred" ? current * 100 : current + value;
    } else {
      out.push(current);
      current = value;
    }
    last = kind;
  }
  out.push(current);
  return out;
}

function numberToWords(n: number): string {
  if (n === 0) return "zero";
  const parts: string[] = [];
  let rest = n;
  if (rest >= 100) {
    parts.push(ONES_WORD[Math.floor(rest / 100)]!, "hundred");
    rest %= 100;
  }
  if (rest >= 20) {
    parts.push(TENS_WORD[Math.floor(rest / 10)]!);
    rest %= 10;
    if (rest > 0) parts.push(ONES_WORD[rest]!);
  } else if (rest >= 10) {
    parts.push(TEENS_WORD[rest - 10]!);
  } else if (rest > 0) {
    parts.push(ONES_WORD[rest]!);
  }
  return parts.join(" ");
}
