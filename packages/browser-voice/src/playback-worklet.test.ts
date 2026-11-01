import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

type WorkletMessage = {
  kind: string;
  playbackId: number;
  audioEndMs?: number;
  audioTime?: number;
  sampleOffset?: number;
};

type WorkletInstance = {
  port: {
    onmessage?: (event: { data: unknown }) => void;
    messages: WorkletMessage[];
  };
  process(inputs: unknown[], outputs: Float32Array[][]): boolean;
};

test("playback clear reports source audio rendered before interruption", () => {
  const Processor = loadProcessor();
  const worklet = new Processor();
  worklet.port.onmessage?.({
    data: { kind: "start", playbackId: 7, sampleRate: 24_000 },
  });
  worklet.port.onmessage?.({
    data: {
      kind: "audio",
      playbackId: 7,
      samples: new Float32Array(2_400).fill(0.25),
    },
  });

  for (const blockSize of [64, 192, 73, 451, 500]) {
    worklet.process(
      [],
      [[new Float32Array(blockSize), new Float32Array(blockSize)]],
    );
  }
  worklet.port.onmessage?.({ data: { kind: "clear", playbackId: 7 } });

  const messages = worklet.port.messages.map((message) => ({
    kind: message.kind,
    playbackId: message.playbackId,
    audioEndMs: message.audioEndMs,
  }));
  assert.deepEqual(messages, [
    { kind: "drained", playbackId: 7, audioEndMs: 26 },
  ]);
});

function loadProcessor(): new () => WorkletInstance {
  let registered: (new () => WorkletInstance) | undefined;
  class AudioWorkletProcessorFake {
    readonly port = {
      messages: [] as WorkletMessage[],
      onmessage: undefined as ((event: { data: unknown }) => void) | undefined,
      postMessage: (message: WorkletMessage) => this.port.messages.push(message),
    };
  }
  const source = readFileSync(
    new URL("../public/playback-worklet.js", import.meta.url),
    "utf8",
  );
  vm.runInNewContext(source, {
    AudioWorkletProcessor: AudioWorkletProcessorFake,
    Float32Array,
    Math,
    sampleRate: 48_000,
    currentTime: 2,
    registerProcessor: (_name: string, Processor: new () => WorkletInstance) => {
      registered = Processor;
    },
  });
  if (!registered) throw new Error("playback processor did not register");
  return registered;
}

test("opt in first render reports the first nonzero output once with its audio clock offset", () => {
  const worklet = new (loadProcessor())();
  const send = (data: unknown) => worklet.port.onmessage?.({ data });
  send({ kind: "start", playbackId: 8, sampleRate: 48_000, reportFirstRender: true });
  const samples = new Float32Array(128);
  samples.fill(0.25, 70);
  send({ kind: "audio", playbackId: 8, samples });
  worklet.process([], [[new Float32Array(128)]]);
  worklet.process([], [[new Float32Array(128)]]);
  const rendered = worklet.port.messages.filter((message) => message.kind === "first-render");
  assert.equal(rendered.length, 1);
  assert.equal(rendered[0]?.playbackId, 8);
  assert.equal(rendered[0]?.sampleOffset, 70);
  assert.equal(rendered[0]?.audioTime, 2 + 70 / 48_000);
});

test("clear closes a generation to late PCM and first render notifications", () => {
  const worklet = new (loadProcessor())();
  const send = (data: unknown) => worklet.port.onmessage?.({ data });
  send({ kind: "start", playbackId: 9, sampleRate: 48_000, reportFirstRender: true });
  send({ kind: "clear", playbackId: 9 });
  send({ kind: "audio", playbackId: 9, samples: new Float32Array(128).fill(1) });
  const output = new Float32Array(128);
  worklet.process([], [[output]]);
  assert.ok(output.every((sample) => sample === 0));
  assert.equal(worklet.port.messages.some((message) => message.kind === "first-render"), false);
});
