import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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

test("normal frames match the pre flush worklet byte for byte", () => {
  // SHA-256 over each frame's sample bytes then Float64 level and audioTime,
  // recorded from capture-worklet.js at 9c0b357 with this same input.
  const parent: Record<number, [number, string]> = {
    8_000: [51, "eaa3323de2090ff92199642988b7c5b7c15622ff6365d0012ae506602a874372"],
    12_000: [34, "cbaafe824c44aeadb5133b01af561e6a534f21dfd053ac204367cacfc0d6948c"],
    16_000: [25, "0ec9d82e8c373d570541504418a927f9e2f772548b0c2ed3535e3d8d28be53b3"],
    44_100: [9, "c26832d0b5f14fe0a08a9d6433f8a177b9c6f1138fcc606a3b0298075c0f6d89"],
    48_000: [8, "98ff2f6d57f9829658cc548144328c3218d52683e033e4e7afd8e64d4f1a861e"],
  };
  for (const [rate, expected] of Object.entries(parent)) {
    const { context, messages, Processor } = loadCapture(Number(rate));
    const worklet = new Processor();
    for (let block = 0; block < 64; block += 1) {
      context.currentFrame = block * 128;
      const input = Float32Array.from({ length: 128 }, (_, index) => {
        const n = block * 128 + index;
        return Math.sin(n * 0.137) * 0.7 + Math.cos(n * 0.811) * 0.2;
      });
      worklet.process([[input]], [[new Float32Array(128)]]);
    }
    const digest = createHash("sha256");
    for (const message of messages) {
      digest.update(new Uint8Array(message.samples.buffer));
      digest.update(
        new Uint8Array(Float64Array.of(message.level, message.audioTime).buffer),
      );
    }
    assert.deepEqual([messages.length, digest.digest("hex")], expected, `${rate} Hz`);
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
