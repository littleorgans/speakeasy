import assert from "node:assert/strict";
import { test } from "node:test";
import { fftPlan } from "./fft.ts";

/** Direct O(n^2) DFT for comparison. */
function dft(re: Float64Array, im: Float64Array): [Float64Array, Float64Array] {
  const size = re.length;
  const outRe = new Float64Array(size);
  const outIm = new Float64Array(size);
  for (let k = 0; k < size; k += 1) {
    for (let n = 0; n < size; n += 1) {
      const angle = (-2 * Math.PI * k * n) / size;
      outRe[k]! += re[n]! * Math.cos(angle) - im[n]! * Math.sin(angle);
      outIm[k]! += re[n]! * Math.sin(angle) + im[n]! * Math.cos(angle);
    }
  }
  return [outRe, outIm];
}

test("fftPlan matches a direct DFT for power of two sizes", () => {
  for (const size of [2, 8, 64, 512]) {
    const re = Float64Array.from({ length: size }, (_, n) => Math.sin(n * 0.7) + 0.3 * Math.cos(n * 2.1));
    const im = Float64Array.from({ length: size }, (_, n) => 0.1 * Math.sin(n * 1.3));
    const [expectedRe, expectedIm] = dft(re, im);
    fftPlan(size)(re, im);
    for (let k = 0; k < size; k += 1) {
      assert.ok(Math.abs(re[k]! - expectedRe[k]!) < 1e-9, `size ${size} re[${k}]`);
      assert.ok(Math.abs(im[k]! - expectedIm[k]!) < 1e-9, `size ${size} im[${k}]`);
    }
  }
});

test("fftPlan of an impulse is flat", () => {
  const re = new Float64Array(16);
  const im = new Float64Array(16);
  re[0] = 1;
  fftPlan(16)(re, im);
  assert.deepEqual([...re], Array(16).fill(1));
  assert.deepEqual([...im], Array(16).fill(0));
});

test("fftPlan refuses sizes that are not a power of two", () => {
  for (const size of [0, 1, 3, 400]) {
    assert.throws(() => fftPlan(size), /power of two/);
  }
});
