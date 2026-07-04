import type { RewriteConfig } from "../contract.ts";
import { applyNumbers } from "./numbers.ts";
import { applyRules } from "./rules.ts";

/**
 * The rewrite pipeline: domain rules first, then number normalization. Both
 * stages are idempotent and pass non-matching text through untouched, so the
 * order only matters where a rule output feeds the number stage (it does not
 * here: rules fix vocabulary, numbers format digits/words).
 */
export function rewriteText(text: string, config: RewriteConfig): string {
  return applyNumbers(applyRules(text, config.rules), config.numbers);
}
