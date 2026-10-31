import assert from "node:assert/strict";
import { test } from "node:test";
import {
  GRANITE_FEATURE_DIM,
  graniteFeatureFrames,
  graniteFeatures,
} from "./granite-frontend.ts";

/**
 * Reference values come from transformers 5.16.0
 * `GraniteSpeech5FeatureExtractor` (torch and torchaudio, CPU), built from the
 * export's processor_config.json and run on the float32 bytes of
 * `referenceSignal()`. Recreate: write the signal to a raw f32 file, call the
 * extractor with `sampling_rate=16000`, read `input_features[0]`.
 */

const TOLERANCE = 1e-4;

/** A chirp plus a tone with 0.1 s of silence inside; 12,345 samples, an odd mel count. */
function referenceSignal(): Float32Array {
  const signal = new Float32Array(12_345);
  for (let index = 0; index < signal.length; index += 1) {
    if (index >= 5_000 && index < 6_600) continue;
    const time = index / 16_000;
    signal[index] =
      0.4 * Math.sin(2 * Math.PI * (150 * time + 900 * time * time)) +
      0.05 * Math.sin(2 * Math.PI * 3_100 * time + 1.3);
  }
  return signal;
}

test("features match the transformers reference on a short clip", () => {
  const { data, frames } = graniteFeatures(referenceSignal());
  assert.equal(frames, 39);
  assert.equal(data.length, 39 * GRANITE_FEATURE_DIM);
  const reference: [frame: number, dim: number, value: number][] = [
    [0, 0, 1.526052],
    [0, 100, -0.250757],
    [0, 170, 0.903625],
    [15, 5, -0.165244],
    [16, 40, 0.776302],
    [20, 250, 0.494641],
    [25, 187, 1.263545],
    [31, 27, 0.316884],
    [38, 90, 0.58915],
    [38, 319, 0.42813],
  ];
  for (const [frame, dim, value] of reference) {
    const actual = data[frame * GRANITE_FEATURE_DIM + dim]!;
    assert.ok(Math.abs(actual - value) < TOLERANCE, `[${frame}, ${dim}] ${actual} vs ${value}`);
  }
  const sum = data.reduce((total, value) => total + value, 0);
  assert.ok(Math.abs(sum - 831.878533154726) < 0.02, `sum ${sum}`);
});

test("frames round the mel count up to whole pairs", () => {
  assert.equal(graniteFeatureFrames(159), 0);
  assert.equal(graniteFeatureFrames(160), 1);
  assert.equal(graniteFeatureFrames(1_280), 4);
  assert.equal(graniteFeatureFrames(16_000), 50);
});

test("input below the graph minimum is refused", () => {
  assert.throws(() => graniteFeatures(new Float32Array(1_000)), /at least 4 feature frames/);
});
