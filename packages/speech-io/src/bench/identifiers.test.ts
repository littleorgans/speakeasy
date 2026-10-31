import assert from "node:assert/strict";
import { test } from "node:test";
import { isIdentifier, rawTokens, scoreIdentifiers } from "./identifiers.ts";

test("identifiers contain a slash, underscore, dot or digit", () => {
  for (const token of ["dev-up/down", "run_id", ".kilo", "README.md", "561", "level1"]) {
    assert.equal(isIdentifier(token), true, token);
  }
  for (const token of ["merge", "cmdoptg", "AFTER", "transport-matters", "let's"]) {
    assert.equal(isIdentifier(token), false, token);
  }
});

test("raw tokens shed edge sentence punctuation only", () => {
  assert.deepEqual(
    rawTokens('Recycle the lead on gpt-6.1-sol. What is .kilo? love it . ship it ..'),
    ["Recycle", "the", "lead", "on", "gpt-6.1-sol", "What", "is", ".kilo", "love", "it", "ship", "it"],
  );
  assert.deepEqual(rawTokens('("open" packages/speech-io/src)'), ["open", "packages/speech-io/src"]);
  assert.deepEqual(rawTokens("   "), []);
});

test("exact identifiers score clean; sentence words never count", () => {
  const score = scoreIdentifiers(
    "Is wire_exchange indexed by run_id",
    "Is wire_exchange indexed by run_id?",
  );
  assert.deepEqual(score, { identifiers: 2, errors: 0, missed: [] });
  assert.deepEqual(scoreIdentifiers("totally different words", "merge it now"), {
    identifiers: 0,
    errors: 0,
    missed: [],
  });
});

test("split or spelled identifiers are errors the normalized WER misses", () => {
  // normalizeTranscript makes these pairs identical; raw scoring does not.
  const score = scoreIdentifiers(
    "Set TM CHECK ONLY equals one and rerun. Open lilo docs index md first.",
    "Set TM_CHECK_ONLY=1 and rerun. Open .lilo/docs/index.md first.",
  );
  assert.deepEqual(score, {
    identifiers: 2,
    errors: 2,
    missed: ["TM_CHECK_ONLY=1", ".lilo/docs/index.md"],
  });
  assert.deepEqual(scoreIdentifiers("yes do 561 and 563", "yes do 561 and 563 now"), {
    identifiers: 2,
    errors: 0,
    missed: [],
  });
  assert.deepEqual(scoreIdentifiers("YES DO FIVE SIXTY ONE", "yes do 561"), {
    identifiers: 1,
    errors: 1,
    missed: ["561"],
  });
  assert.equal(scoreIdentifiers("open readme.md", "open README.md").errors, 0);
  assert.equal(scoreIdentifiers("open readme.md", "open README.md", { caseSensitive: true }).errors, 1);
});

test("an identifier matched out of alignment order still counts as missed", () => {
  // The hypothesis carries 561 but in a position the alignment cannot match.
  const score = scoreIdentifiers("561 yes do 563", "yes do 561 and 563");
  assert.deepEqual(score.missed, ["561"]);
});

test("primary alignment ignores casing and preserves raw identifier spelling", () => {
  const expected = "Open README.md beside run_id and .lilo/docs/index.md.";
  const hypothesis = "OPEN README.MD BESIDE RUN_ID AND .LILO/DOCS/INDEX.MD.";
  assert.deepEqual(scoreIdentifiers(hypothesis, expected), {
    identifiers: 3, errors: 0, missed: [],
  });
  assert.deepEqual(scoreIdentifiers(hypothesis, expected, { caseSensitive: true }), {
    identifiers: 3, errors: 3, missed: ["README.md", "run_id", ".lilo/docs/index.md"],
  });
});
