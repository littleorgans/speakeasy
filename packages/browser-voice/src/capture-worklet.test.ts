import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

type FrameMessage = {
  kind: string;
  samples: Float32Array;
  level: number;
  audioTime: number;
  flush?: boolean;
};

type CaptureInstance = {
  port: { onmessage?: (event: { data: unknown }) => void };
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

test("flush posts the buffered partial as a marked short frame", () => {
  const { context, messages, Processor } = loadCapture(48_000);
  const worklet = new Processor();
  feed(context, worklet, 0, 2, 0.5);
  assert.equal(messages.length, 0);
  flush(worklet);
  assert.equal(messages.length, 1);
  const [partial] = messages;
  assert.equal(partial?.kind, "frame");
  assert.equal(partial?.flush, true);
  // 256 buffered source samples at ratio 3 -> 85 output samples.
  assert.deepEqual(Array.from(partial?.samples ?? []), Array(85).fill(0.5));
  assert.equal(partial?.level, 0.5);
  assert.equal(partial?.audioTime, 256 / 48_000);
});

test("a frame after flush carries no earlier samples", () => {
  for (const rate of [48_000, 44_100]) {
    const { context, messages, Processor } = loadCapture(rate);
    const worklet = new Processor();
    feed(context, worklet, 0, 3, 1);
    flush(worklet);
    feed(context, worklet, 3, 20, 0);
    const after = messages.slice(1);
    assert.ok(after.length > 0, `no frame after flush at ${rate} Hz`);
    for (const message of after) {
      assert.equal(message.flush, undefined);
      assert.deepEqual(Array.from(message.samples), Array(320).fill(0));
    }
  }
});

test("flush with an empty buffer posts one empty marked frame", () => {
  const { messages, Processor } = loadCapture(48_000);
  const worklet = new Processor();
  flush(worklet);
  flush(worklet);
  assert.equal(messages.length, 2);
  for (const message of messages) {
    assert.equal(message.flush, true);
    assert.equal(message.samples.length, 0);
    assert.equal(message.level, 0);
  }
});

test("normal frames around a flush keep their shape", () => {
  const { context, messages, Processor } = loadCapture(48_000);
  const worklet = new Processor();
  feed(context, worklet, 0, 10, 0.25);
  flush(worklet);
  feed(context, worklet, 10, 18, 0.25);
  const normal = messages.filter((message) => message.flush !== true);
  assert.equal(normal.length, 2);
  for (const message of normal) {
    assert.deepEqual(Object.keys(message).sort(), [
      "audioTime",
      "kind",
      "level",
      "samples",
    ]);
    assert.deepEqual(Array.from(message.samples), Array(320).fill(0.25));
  }
});

function feed(
  context: { currentFrame: number },
  worklet: CaptureInstance,
  fromBlock: number,
  toBlock: number,
  value: number,
): void {
  for (let block = fromBlock; block < toBlock; block += 1) {
    context.currentFrame = block * 128;
    worklet.process([[new Float32Array(128).fill(value)]], [[new Float32Array(128)]]);
  }
}

function flush(worklet: CaptureInstance): void {
  const handler = worklet.port.onmessage;
  if (!handler) throw new Error("capture processor accepts no port messages");
  handler({ data: { kind: "flush" } });
}

function loadCapture(sampleRate: number): {
  context: { currentFrame: number };
  messages: FrameMessage[];
  Processor: new () => CaptureInstance;
} {
  const messages: FrameMessage[] = [];
  let registered: (new () => CaptureInstance) | undefined;
  class AudioWorkletProcessorFake {
    readonly port: CaptureInstance["port"] & {
      postMessage(message: FrameMessage): void;
    } = {
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
