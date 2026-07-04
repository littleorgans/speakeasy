import assert from "node:assert/strict";
import { test } from "node:test";
import { applyNumbers, parseNumbersMode } from "./numbers.ts";

test("off mode passes everything through", () => {
  assert.equal(applyNumbers("close pane ten", "off"), "close pane ten");
  assert.equal(applyNumbers("close pane 10", "off"), "close pane 10");
});

test("digits mode: single words, teens, tens, hundreds", () => {
  assert.equal(applyNumbers("close pane ten", "digits"), "close pane 10");
  assert.equal(applyNumbers("nineteen", "digits"), "19");
  assert.equal(applyNumbers("twenty three agents", "digits"), "23 agents");
  assert.equal(applyNumbers("one hundred twenty three", "digits"), "123");
  assert.equal(applyNumbers("nine hundred ninety nine", "digits"), "999");
  assert.equal(applyNumbers("zero", "digits"), "0");
});

test("digits mode: a spoken counting list stays separate, not compounded", () => {
  assert.equal(
    applyNumbers("one two three four five six seven eight nine ten", "digits"),
    "1 2 3 4 5 6 7 8 9 10",
  );
});

test("digits mode is case-insensitive and leaves non-numbers alone", () => {
  assert.equal(applyNumbers("Spawn Ten agents", "digits"), "Spawn 10 agents");
  // "tone" is not a number word and must never be mistaken for "ten"
  assert.equal(applyNumbers("spawn tone agents", "digits"), "spawn tone agents");
});

test("words mode: digits to spelled numbers across ranges", () => {
  assert.equal(applyNumbers("close pane 10", "words"), "close pane ten");
  assert.equal(applyNumbers("23", "words"), "twenty three");
  assert.equal(applyNumbers("100", "words"), "one hundred");
  assert.equal(applyNumbers("123", "words"), "one hundred twenty three");
  assert.equal(applyNumbers("0", "words"), "zero");
});

test("words mode leaves out-of-range and leading-zero tokens untouched", () => {
  assert.equal(applyNumbers("port 1000", "words"), "port 1000");
  assert.equal(applyNumbers("agent 007", "words"), "agent 007");
});

test("both directions are idempotent", () => {
  const d = applyNumbers("twenty three", "digits");
  assert.equal(applyNumbers(d, "digits"), d);
  const w = applyNumbers("23", "words");
  assert.equal(applyNumbers(w, "words"), w);
});

test("parseNumbersMode validates", () => {
  assert.equal(parseNumbersMode("digits"), "digits");
  assert.equal(parseNumbersMode("words"), "words");
  assert.equal(parseNumbersMode("off"), "off");
  assert.throws(() => parseNumbersMode("nope"));
});
