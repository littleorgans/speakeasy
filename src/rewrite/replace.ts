import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

/**
 * Post-decode text replacement (experimental, Path B "in-house" arm).
 *
 * Fixes the kroko default's SYSTEMATIC residuals by rewriting the committed
 * final text a consumer receives: proper-noun mis-splits and a spoken-number
 * rule. This is the counterpart to the sherpa ruleFsts arm; both encode the
 * SAME ruleset (rules.json) so the two are directly comparable.
 *
 * NOT part of normalize.ts on purpose: normalize.ts canonicalizes BOTH sides
 * for a fair WER measurement, whereas this transforms the ENGINE OUTPUT only.
 * In the corpus scorer it is applied to the hypothesis alone.
 *
 * Brittleness: whole-word, case-insensitive literal replacement. Several rules
 * (crown->chrome, pain->pane, ten->10) rewrite words that are legitimate in
 * other contexts; on the director-command corpus they appear only as mishears,
 * but a general deployment would over-trigger ("the crown jewels"). Rules
 * flagged overTrigger in rules.json carry that risk.
 */

export type RewriteRule = {
  from: string;
  to: string;
  /** True when `from` is a common word that would over-trigger elsewhere. */
  overTrigger?: boolean;
};

const rulesUrl = new URL("./rules.json", import.meta.url);
export const REWRITE_RULES: RewriteRule[] = JSON.parse(
  readFileSync(rulesUrl, "utf8"),
) as RewriteRule[];

/** Committed OpenFst rewrite built from rules.json (see scripts/build-fst.py). */
export const REWRITE_FST_PATH = fileURLToPath(
  new URL("./replace.fst", import.meta.url),
);

/** Compile a rule to a whole-word, case-insensitive matcher. */
function ruleMatcher(from: string): RegExp {
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escaped}\\b`, "gi");
}

/**
 * Apply the ruleset to `text`, longest `from` first so multi-word phrases win
 * over any single-word rule they contain. Case-insensitive match; the
 * replacement is emitted verbatim from the rule.
 */
export function applyRewrites(
  text: string,
  rules: RewriteRule[] = REWRITE_RULES,
): string {
  const ordered = [...rules].sort((a, b) => b.from.length - a.from.length);
  let out = text;
  for (const rule of ordered) {
    out = out.replace(ruleMatcher(rule.from), rule.to);
  }
  return out;
}
