import assert from "node:assert/strict";
import { test } from "node:test";
import {
  frontendChunks,
  logMel,
  parseMelFilters,
  subsampledFrames,
  type MelFilters,
} from "./ort-tdt-mel.ts";

const FLUID_MEL = { nFft: 512, hop: 160, win: 400, preemphasis: 0.97 };
const FLUID_SUBSAMPLING = { kernel: 3, stride: 2, layers: 3 };

/**
 * Reference values: the pinned Fluid transcribe.py `log_mel` run on this
 * signal and filter bank (numpy 2.5.4), rounded to 5 decimals.
 */
const REFERENCE_FEATURES = [
  -2.84593, 2.7632, -2.84593, 0.31274, 0.13255, 0.31231, 0.31724, -0.43268, 0.31731,
  0.31724, -0.43435, 0.3173, 0.31725, -0.43598, 0.3173, 0.31725, -0.43614, 0.3173,
  0.31724, -0.43483, 0.3173, 0.31724, -0.43341, 0.3173, 0.31724, -0.43289, 0.31731,
  0.31249, 0.14457, 0.31249, 0, 0, 0,
];

function twoTones(): Float32Array {
  return Float32Array.from(
    { length: 1_600 },
    (_, n) =>
      0.5 * Math.sin((2 * Math.PI * 440 * n) / 16_000) +
      0.25 * Math.sin((2 * Math.PI * 1_250 * n) / 16_000 + 0.3),
  );
}

/** Three triangular bands over bins [4 + 10b, 24 + 10b], peak at 12 + 10b. */
function triangleFilters(): MelFilters {
  const bins = 257;
  const weights = new Float32Array(3 * bins);
  for (let band = 0; band < 3; band += 1) {
    const [low, peak, high] = [4 + 10 * band, 12 + 10 * band, 24 + 10 * band];
    for (let k = 0; k < bins; k += 1) {
      weights[band * bins + k] =
        k >= low && k <= peak
          ? (k - low) / (peak - low)
          : k > peak && k <= high
            ? (high - k) / (high - peak)
            : 0;
    }
  }
  return { bands: 3, bins, weights };
}

test("logMel matches the reference runner's features", () => {
  const mel = logMel(twoTones(), FLUID_MEL, triangleFilters());
  assert.equal(mel.frames, 11);
  assert.equal(mel.valid, 10);
  assert.equal(mel.features.length, REFERENCE_FEATURES.length);
  mel.features.forEach((value, index) => {
    assert.ok(
      Math.abs(value - REFERENCE_FEATURES[index]!) < 1e-4,
      `feature ${index}: ${value} vs ${REFERENCE_FEATURES[index]}`,
    );
  });
});

test("logMel refuses a filter bank sized for another FFT", () => {
  assert.throws(
    () => logMel(twoTones(), FLUID_MEL, { bands: 1, bins: 201, weights: new Float32Array(201) }),
    /257/,
  );
});

test("subsampledFrames matches the reference for edge lengths", () => {
  // Reference `Model.subsampled` outputs for kernel 3, stride 2, 3 layers.
  const reference: [number, number][] = [
    [1, 1], [2, 1], [8, 1], [9, 2], [17, 3], [300, 38], [301, 38], [401, 51], [3_001, 376],
  ];
  for (const [frames, expected] of reference) {
    assert.equal(subsampledFrames(frames, FLUID_SUBSAMPLING), expected, `${frames} frames`);
  }
});

test("frontendChunks tile the output with one frame of context each side", () => {
  const frames = 3_001;
  const chunks = frontendChunks(frames, 50, FLUID_SUBSAMPLING);
  assert.equal(chunks.reduce((total, chunk) => total + chunk.count, 0), 376);
  assert.deepEqual(chunks[0], { start: 0, end: 408, skip: 0, count: 50 });
  assert.deepEqual(chunks[1], { start: 392, end: 808, skip: 1, count: 50 });
  assert.deepEqual(chunks.at(-1), { start: 2_792, end: 3_001, skip: 1, count: 26 });
});

function npy(descr: string, shape: string, values: number[], width: 4 | 8): Uint8Array {
  const header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shape}, }`.padEnd(117) + "\n";
  const bytes = new Uint8Array(10 + header.length + values.length * width);
  const view = new DataView(bytes.buffer);
  bytes.set([0x93, ...Buffer.from("NUMPY"), 1, 0]);
  view.setUint16(8, header.length, true);
  bytes.set(Buffer.from(header, "latin1"), 10);
  values.forEach((value, index) => {
    const at = 10 + header.length + index * width;
    if (width === 4) view.setFloat32(at, value, true);
    else view.setFloat64(at, value, true);
  });
  return bytes;
}

test("parseMelFilters reads float32 and float64 numpy files", () => {
  for (const [descr, width] of [["<f4", 4], ["<f8", 8]] as const) {
    const filters = parseMelFilters(npy(descr, "(2, 3)", [0, 0.5, 1, 0.25, 0, 0.75], width));
    assert.equal(filters.bands, 2);
    assert.equal(filters.bins, 3);
    assert.deepEqual([...filters.weights], [0, 0.5, 1, 0.25, 0, 0.75]);
  }
});

test("parseMelFilters rejects other dtypes and non numpy bytes", () => {
  assert.throws(() => parseMelFilters(npy("<i4", "(1, 1)", [1], 4)), /dtype/);
  assert.throws(() => parseMelFilters(new Uint8Array(16)), /npy/);
});
