import assert from "node:assert/strict";
import { test } from "node:test";
import { EnergyVad } from "./vad.ts";

const FRAME = 320; // 20ms @ 16kHz

function frame(peak: number): Float32Array {
  return new Float32Array(FRAME).fill(peak);
}

test("EnergyVad trips only after sustained speech", () => {
  const vad = new EnergyVad({ threshold: 0.1, minSpeechMs: 100, frameMs: 20 }); // 5 frames
  let tripped = false;
  for (let i = 0; i < 4; i += 1) {
    tripped = vad.accept(frame(0.5)) || tripped;
  }
  assert.equal(tripped, false); // 4 loud frames, not yet 5
  assert.equal(vad.accept(frame(0.5)), true); // the 5th trips it
});

test("EnergyVad ignores quiet frames and resets on a gap", () => {
  const vad = new EnergyVad({ threshold: 0.1, minSpeechMs: 100, frameMs: 20 });
  vad.accept(frame(0.5));
  vad.accept(frame(0.5));
  vad.accept(frame(0.0)); // silence resets the run
  for (let i = 0; i < 4; i += 1) {
    assert.equal(vad.accept(frame(0.5)), false);
  }
  assert.equal(vad.accept(frame(0.5)), true); // needs 5 fresh consecutive
});

test("EnergyVad never trips on sub-threshold audio", () => {
  const vad = new EnergyVad({ threshold: 0.2, minSpeechMs: 40, frameMs: 20 });
  for (let i = 0; i < 50; i += 1) {
    assert.equal(vad.accept(frame(0.05)), false);
  }
});
