import assert from "node:assert/strict";
import { test } from "node:test";
import { assertSampleRate, concatFrames, samplesToMs, SPEECH_SAMPLE_RATE } from "./pcm.js";

test("concatFrames joins frames in order into a new buffer", () => {
  const first = Float32Array.of(0.1, 0.2);
  const audio = concatFrames([first, Float32Array.of(), Float32Array.of(0.3)]);
  assert.deepEqual([...audio], [...Float32Array.of(0.1, 0.2, 0.3)]);
  assert.notEqual(concatFrames([first]), first);
  assert.equal(concatFrames([]).length, 0);
});

test("concatFrames presizes to a supplied sample count", () => {
  const audio = concatFrames([Float32Array.of(1, 2)], 4);
  assert.deepEqual([...audio], [1, 2, 0, 0]);
  assert.throws(() => concatFrames([Float32Array.of(1, 2)], 1), RangeError);
});

test("samplesToMs converts at the speech rate", () => {
  assert.equal(SPEECH_SAMPLE_RATE, 16_000);
  assert.equal(samplesToMs(16_000), 1_000);
  assert.equal(samplesToMs(160), 10);
});

test("assertSampleRate accepts an unset or matching rate only", () => {
  assert.doesNotThrow(() => assertSampleRate("Engine"));
  assert.doesNotThrow(() => assertSampleRate("Engine", {}));
  assert.doesNotThrow(() => assertSampleRate("Engine", { sampleRate: 16_000 }));
  assert.throws(
    () => assertSampleRate("Engine", { sampleRate: 48_000 }),
    /^Error: Engine expects 16000 Hz frames, received 48000 Hz$/,
  );
});
