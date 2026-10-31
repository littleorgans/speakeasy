import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

type FrameMessage = { kind: string; samples: Float32Array; audioTime: number };

type CaptureInstance = {
  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
};

test("each frame carries the context time at its last source sample", () => {
  const { context, messages, Processor } = loadCapture(48_000);
  const worklet = new Processor();
  for (let block = 0; block < 10; block += 1) {
    context.currentFrame = block * 128;
    worklet.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
  }
  // 48 kHz -> 16 kHz, 20 ms frames: each frame consumes 960 source samples.
  assert.deepEqual(
    messages.map((message) => message.audioTime),
    [960 / 48_000],
  );
  assert.equal(messages[0]?.samples.length, 320);
});

test("audioTime follows currentFrame across blocks at 44.1 kHz", () => {
  const { context, messages, Processor } = loadCapture(44_100);
  const worklet = new Processor();
  const startFrame = 44_100;
  for (let block = 0; block < 20; block += 1) {
    context.currentFrame = startFrame + block * 128;
    worklet.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
  }
  const consumed = Math.floor(320 * (44_100 / 16_000));
  assert.deepEqual(
    messages.map((message) => message.audioTime),
    [1, 2].map((count) => (startFrame + count * consumed) / 44_100),
  );
});

function loadCapture(sampleRate: number): {
  context: { currentFrame: number };
  messages: FrameMessage[];
  Processor: new () => CaptureInstance;
} {
  const messages: FrameMessage[] = [];
  let registered: (new () => CaptureInstance) | undefined;
  class AudioWorkletProcessorFake {
    readonly port = {
      postMessage: (message: FrameMessage) => messages.push(message),
    };
  }
  const context = vm.createContext({
    AudioWorkletProcessor: AudioWorkletProcessorFake,
    Float32Array,
    Math,
    sampleRate,
    currentFrame: 0,
    registerProcessor: (_name: string, Processor: new () => CaptureInstance) => {
      registered = Processor;
    },
  }) as { currentFrame: number };
  const source = readFileSync(
    new URL("../public/capture-worklet.js", import.meta.url),
    "utf8",
  );
  vm.runInContext(source, context);
  if (!registered) throw new Error("capture processor did not register");
  return { context, messages, Processor: registered };
}
