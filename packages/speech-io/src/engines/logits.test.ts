import assert from "node:assert/strict";
import { test } from "node:test";
import { argMax, logSoftmaxAt, logSumExp } from "./logits.ts";

const row = Float32Array.of(0.5, 2, -1, 2, 1);

test("argMax keeps the first of tied maxima", () => {
  assert.equal(argMax(row), 1);
  assert.equal(argMax(Float32Array.of(3, 3, 3)), 0);
});

test("argMax over a range is relative to its start", () => {
  assert.equal(argMax(row, 2, 5), 1);
  assert.equal(argMax(row, 3, 3), 0);
  assert.equal(argMax(new Float32Array(0)), 0);
});

test("argMax skips NaN unless it sits at the range start", () => {
  assert.equal(argMax(Float32Array.of(1, NaN, 2)), 2);
  assert.equal(argMax(Float32Array.of(NaN, 5, 9)), 0);
  assert.equal(argMax(Float32Array.of(0, NaN, 9), 1), 0);
});

test("logSumExp is stable for large logits", () => {
  assert.ok(Math.abs(logSumExp(Float32Array.of(1000, 1000)) - (1000 + Math.LN2)) < 1e-9);
  assert.ok(Math.abs(logSumExp(row, 0, 2) - Math.log(Math.exp(0.5) + Math.exp(2))) < 1e-9);
  assert.ok(Number.isNaN(logSumExp(Float32Array.of(1, NaN))));
});

test("logSoftmaxAt normalizes within the range", () => {
  const expected = 2 - Math.log(Math.exp(0.5) + Math.exp(2) + Math.exp(-1));
  assert.ok(Math.abs(logSoftmaxAt(row, 1, 0, 3) - expected) < 1e-9);
  assert.ok(Math.abs(logSoftmaxAt(row, 0, 3, 5) - (2 - Math.log(Math.exp(2) + Math.exp(1)))) < 1e-9);
  assert.equal(logSoftmaxAt(Float32Array.of(-4), 0), 0);
  assert.ok(Number.isNaN(logSoftmaxAt(Float32Array.of(1, NaN), 0)));
});
