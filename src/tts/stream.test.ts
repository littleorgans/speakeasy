import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildTimingReport,
  planSegments,
  splitSentences,
  trimSilence,
} from "./stream.ts";

test("splitSentences splits on terminators and keeps them", () => {
  assert.deepEqual(
    splitSentences(
      "The build finished. Two checks are running! Say the word?",
    ),
    ["The build finished.", "Two checks are running!", "Say the word?"],
  );
});

test("splitSentences treats unterminated text as one sentence", () => {
  assert.deepEqual(splitSentences("promote it to staging"), [
    "promote it to staging",
  ]);
});

test("splitSentences keeps trailing quotes with the sentence", () => {
  assert.deepEqual(splitSentences('She said "go." Then left.'), [
    'She said "go."',
    "Then left.",
  ]);
});

test("splitSentences drops whitespace-only pieces", () => {
  assert.deepEqual(splitSentences("  One.   "), ["One."]);
  assert.deepEqual(splitSentences(""), []);
});

test("buildTimingReport proves gapless playback when synth stays ahead", () => {
  const report = buildTimingReport([
    { index: 0, sentence: "a.", readyAtMs: 400, synthMs: 400, audioDurationMs: 2000 },
    { index: 1, sentence: "b.", readyAtMs: 900, synthMs: 500, audioDurationMs: 2000 },
    { index: 2, sentence: "c.", readyAtMs: 1500, synthMs: 600, audioDurationMs: 2000 },
  ]);
  assert.equal(report.timeToFirstAudioMs, 400);
  assert.equal(report.aheadOfPlayback, true);
  // segment 1 must be ready by 400 + 2000 = 2400; it was ready at 900.
  assert.equal(report.segments[1]!.playStartMs, 2400);
  assert.equal(report.segments[1]!.marginMs, 1500);
});

test("buildTimingReport flags an underrun", () => {
  const report = buildTimingReport([
    { index: 0, sentence: "a.", readyAtMs: 400, synthMs: 400, audioDurationMs: 1000 },
    { index: 1, sentence: "b.", readyAtMs: 3000, synthMs: 2600, audioDurationMs: 1000 },
  ]);
  assert.equal(report.aheadOfPlayback, false);
  assert.equal(report.segments[1]!.marginMs, 1400 - 3000);
});

test("buildTimingReport rejects empty input", () => {
  assert.throws(() => buildTimingReport([]));
});

test("trimSilence removes edge silence, keeps a guard, appends the gap", () => {
  const sr = 1000; // 1 sample per ms keeps the arithmetic obvious
  const samples = new Float32Array(120); // 50ms lead, 20ms speech, 50ms trail
  for (let i = 50; i < 70; i += 1) samples[i] = 0.5;
  const out = trimSilence(samples, sr, 10); // 10ms controlled gap
  // 20ms speech + 8ms guard each side = 36 samples, then a 10ms gap = 46.
  assert.equal(out.length, 46);
  assert.equal(out[8], 0.5); // first speech sample, after the 8ms leading guard
  assert.equal(out[36], 0); // gap begins right after the kept region
  assert.equal(out[out.length - 1], 0); // gap is trailing silence
});

test("trimSilence collapses all-silence input to just the gap", () => {
  const out = trimSilence(new Float32Array(500), 1000, 10);
  assert.equal(out.length, 10);
  assert.ok(out.every((sample) => sample === 0));
});

test("planSegments carves a short first chunk at the word cap", () => {
  assert.deepEqual(
    planSegments(
      "The build finished on the staging channel. Two checks are still running.",
    ),
    [
      "The build finished on", // 4-word cap
      "the staging channel.",
      "Two checks are still running.",
    ],
  );
});

test("planSegments cuts the first chunk at an earlier clause boundary", () => {
  assert.deepEqual(planSegments("However, we should proceed. Then stop."), [
    "However,",
    "we should proceed.",
    "Then stop.",
  ]);
});

test("planSegments leaves a short first sentence whole", () => {
  assert.deepEqual(planSegments("Open browser. Then wait."), [
    "Open browser.",
    "Then wait.",
  ]);
  assert.deepEqual(planSegments("Hello there."), ["Hello there."]);
});
