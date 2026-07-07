import { readFileSync } from "node:fs";
import type { RewriteRule } from "../contract.ts";

/**
 * Domain rule application (rewrite pipeline stage 1).
 *
 * Whole-word (\b), case-insensitive literal replacement, longest `from` first
 * so multi-word phrases win over any single-word rule they contain. Fixes the
 * recognizer's systematic vocabulary mis-hears (proper nouns). Rules live in
 * the user-editable rules.json; numbers are NOT here (see numbers.ts).
 *
 * Idempotent and passthrough: text with no match is returned unchanged.
 *
 * Brittleness: rules flagged `overTrigger` rewrite words that are legitimate
 * elsewhere ("the crown jewels" -> "the chrome jewels"). Acceptable for a
 * narrow command vocabulary; a broad deployment should scope rules carefully.
 */

const rulesUrl = new URL("./rules.json", import.meta.url);

/** The committed domain ruleset, loaded once from rules.json. */
export const DEFAULT_RULES: RewriteRule[] = JSON.parse(
  readFileSync(rulesUrl, "utf8"),
) as RewriteRule[];

function ruleMatcher(from: string): RegExp {
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped}\\b`, "gi");
}

export function applyRules(
  text: string,
  rules: RewriteRule[] = DEFAULT_RULES,
): string {
  const ordered = [...rules].sort((a, b) => b.from.length - a.from.length);
  let out = text;
  for (const rule of ordered) {
    out = out.replace(ruleMatcher(rule.from), rule.to);
  }
  return out;
}
