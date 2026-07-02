import assert from "node:assert/strict";
import { test } from "node:test";
import {
  isExactTranscript,
  isWordTolerantTranscript,
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
