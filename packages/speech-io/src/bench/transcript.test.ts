import assert from "node:assert/strict";
import { test } from "node:test";
import { normalizeTranscript } from "./normalize.ts";
import {
  isExactTranscript,
  isWordTolerantTranscript,
  wordErrorAlignment,
  wordErrorCount,
} from "./transcript.ts";

const EXPECTED = "and so my fellow americans";

test("exact transcript matches through case and punctuation", () => {
  assert.equal(isExactTranscript("And so, my fellow Americans!", EXPECTED), true);
  assert.equal(isExactTranscript("AND SAW MY FELLOW AMERICANS", EXPECTED), false);
});

test("single word substitution passes word-tolerant gate", () => {
  assert.equal(wordErrorCount("AND SAW MY FELLOW AMERICANS", EXPECTED), 1);
  assert.equal(isWordTolerantTranscript("AND SAW MY FELLOW AMERICANS", EXPECTED), true);
});

test("fragmented decode fails word-tolerant gate", () => {
  assert.equal(isWordTolerantTranscript("AND SAW MY FELL OW A MERICANS", EXPECTED), false);
  assert.equal(isWordTolerantTranscript("AND SAW MY FELL OW A MERICANS", EXPECTED, 3), false);
});

test("perfect transcript has zero word errors", () => {
  assert.equal(wordErrorCount("and so my fellow americans", EXPECTED), 0);
});

test("empty and unrelated text fail", () => {
  assert.equal(isWordTolerantTranscript("", EXPECTED), false);
  assert.equal(isWordTolerantTranscript("ask not what your country", EXPECTED), false);
});

test("one missing word passes, two fail", () => {
  assert.equal(isWordTolerantTranscript("and so my fellow", EXPECTED), true);
  assert.equal(isWordTolerantTranscript("and so my", EXPECTED), false);
});

test("alignment reports the substitution pair", () => {
  const alignment = wordErrorAlignment("AND SAW MY FELLOW AMERICANS", EXPECTED);
  assert.equal(alignment.errors, 1);
  assert.equal(alignment.referenceWords, 5);
  assert.deepEqual(alignment.substitutions, [{ expected: "so", actual: "saw" }]);
  assert.deepEqual(alignment.insertions, []);
  assert.deepEqual(alignment.deletions, []);
});

test("normalizes case, punctuation, and number words to digits", () => {
  assert.equal(normalizeTranscript("Ten, 10  TEN!"), "10 10 10");
  assert.equal(normalizeTranscript("One, two, three"), "1 2 3");
  assert.equal(normalizeTranscript(""), "");
  assert.equal(normalizeTranscript("!?.,"), "");
});

test("digit/word spellings of a number collapse in WER", () => {
  assert.equal(isExactTranscript("spawn ten agents", "spawn 10 agents"), true);
  assert.equal(wordErrorCount("spawn ten agents", "spawn 10 agents"), 0);
  assert.equal(
    isExactTranscript("one two three four five", "1 2 3 4 5"),
    true,
  );
});

test("genuine number mishears still count as errors", () => {
  assert.equal(wordErrorCount("spawn tone agents", "spawn 10 agents"), 1);
  const alignment = wordErrorAlignment("Close pain ten", "Close Pane 10");
  assert.equal(alignment.errors, 1);
  assert.deepEqual(alignment.substitutions, [
    { expected: "pane", actual: "pain" },
  ]);
});

test("alignment reports insertions and deletions", () => {
  const missing = wordErrorAlignment("and so my fellow", EXPECTED);
  assert.deepEqual(missing.deletions, ["americans"]);
  assert.equal(missing.errors, 1);

  const extra = wordErrorAlignment("and so so my fellow americans", EXPECTED);
  assert.deepEqual(extra.insertions, ["so"]);
  assert.equal(extra.errors, 1);

  const empty = wordErrorAlignment("", EXPECTED);
  assert.equal(empty.errors, 5);
  assert.deepEqual(empty.deletions, EXPECTED.split(" "));
});
