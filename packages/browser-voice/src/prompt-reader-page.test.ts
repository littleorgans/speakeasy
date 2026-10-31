import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import { feed, loadCapture } from "./capture-worklet-harness.ts";

type PagePort = { onmessage?: (event: { data: unknown }) => void; postMessage(data: unknown): void };

/** Stands in for the worklet: answers each flush with an empty marked frame. */
function echoEmptyFlush(port: PagePort, data: unknown) {
  if ((data as { kind?: string })?.kind !== "flush") return;
  port.onmessage?.({ data: { kind: "frame", samples: new Float32Array(0), level: 0, flush: true } });
}

/** Execute the shipped page with controlled mic acquisition and worklet messages. */
function pageHarness({ worklet = echoEmptyFlush }: { worklet?: (port: PagePort, data: unknown) => void } = {}) {
  const pending: Array<(stream: unknown) => void> = [];
  const constraints: unknown[] = [];
  const denied: Array<(error: Error) => void> = [];
  const connected: unknown[] = [];
  const played: Array<{ samples: Float32Array }> = [];
  const fetched: string[] = [];
  const posted: unknown[] = [];
  const elements = new Map<string, Record<string, unknown>>();
  const document = {
    body: { dataset: {} },
    getElementById(id: string) {
      if (!elements.has(id)) elements.set(id, {});
      return elements.get(id);
    },
  };
  class AudioContext {
    destination = {};
    audioWorklet = { async addModule() {} };
    async resume() {}
    createGain() { return { gain: { value: 1 }, connect() {} }; }
    createBuffer(_channels: number, length: number) {
      const buffer = { samples: new Float32Array(length), copyToChannel(data: Float32Array) { buffer.samples.set(data); } };
      return buffer;
    }
    async decodeAudioData(bytes: ArrayBuffer) { return { samples: new Float32Array(bytes) }; }
    createBufferSource() {
      return {
        buffer: null as { samples: Float32Array } | null,
        onended: null,
        connect() {},
        start() { played.push(this.buffer!); },
        stop() {},
      };
    }
    createMediaStreamSource(stream: unknown) {
      return { connect: () => connected.push(stream), disconnect() {} };
    }
  }
  class AudioWorkletNode {
    port: PagePort = {
      postMessage: (data) => {
        posted.push(data);
        worklet(this.port, data);
      },
    };
    connect(node: unknown) { return node; }
  }
  const source = readFileSync(new URL("../public/prompt-reader.js", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "")
    .replace("void new PromptReaderPage().start();", "globalThis.Reader = PromptReaderPage;");
  const context = vm.createContext({
    document, AudioContext, AudioWorkletNode, HoldReleaseGate,
    Float32Array, URLSearchParams,
    fetch: async (url: string) => {
      fetched.push(url);
      return { ok: true, arrayBuffer: async () => new Float32Array([0.5, -0.5]).buffer };
    },
    navigator: { mediaDevices: { getUserMedia: (options: unknown) => {
      constraints.push(options);
      return new Promise((resolve, reject) => { pending.push(resolve); denied.push(reject); });
    } } },
    window: { setTimeout, clearTimeout },
  });
  vm.runInContext(readFileSync(new URL("../public/capture-constraints.js", import.meta.url), "utf8").replace("export ", ""), context);
  vm.runInContext(source, context);
  const reader = new context.Reader();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this" }];
  return { reader, pending, denied, connected, constraints, played, fetched, posted, expectedConstraints: vm.runInContext("CAPTURE_AUDIO_CONSTRAINTS", context) };
}

function mic() {
  const track = { label: "test mic", stopped: false, stop() { this.stopped = true; } };
  return { getTracks: () => [track], getAudioTracks: () => [track], track };
}

async function waitForRequests(pending: unknown[], count: number) {
  for (let tries = 0; pending.length < count && tries < 20; tries++) await Promise.resolve();
  assert.equal(pending.length, count);
}

test("a late mic acquisition cannot attach to a later take", async () => {
  const { reader, pending, connected } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  reader.cancelTake("cancelled");
  const retaking = reader.press();
  await waitForRequests(pending, 2);
  const stale = mic();
  const current = mic();
  pending[0]!(stale);
  await opening;
  pending[1]!(current);
  await retaking;
  assert.equal(stale.track.stopped, true);
  assert.deepEqual(connected, [current]);
  reader.cancelTake("done");
  assert.equal(current.track.stopped, true);
});

test("review keeps worklet frame values and shared mic constraints", async () => {
  const { reader, pending, constraints, expectedConstraints } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  pending[0]!(mic());
  await opening;
  assert.equal(JSON.stringify(constraints), JSON.stringify([{ audio: expectedConstraints }]));
  const frames = [new Float32Array(3_200).fill(0.123456), new Float32Array(3_200).fill(-0.654321)];
  for (const samples of frames) reader.onFrame({ kind: "frame", samples, level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(reader.take.samples.length, 6_400);
  assert.equal(reader.take.samples[0], frames[0]![0]);
  assert.equal(reader.take.samples[3_200], frames[1]![0]);
});

test("each take drains its worklet tail, so the next take holds none of it", async () => {
  // Page and real worklet, connected through asynchronous ports at 48 kHz.
  let page: PagePort | undefined;
  const { context, Processor } = loadCapture(48_000, (message) =>
    queueMicrotask(() => page?.onmessage?.({ data: message })),
  );
  const processor = new Processor();
  const { reader, pending, posted } = pageHarness({
    worklet: (_port, data) => queueMicrotask(() => processor.port.onmessage?.({ data })),
  });
  const timers: Array<() => void> = [];
  reader.gate.schedule = (callback: () => void) => timers.push(callback);
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const takes: Float32Array[] = [];
  let block = 0;
  for (const value of [1, 0]) {
    const opening = reader.press();
    await waitForRequests(pending, takes.length + 1);
    pending[takes.length]!(mic());
    await opening;
    page = reader.capture.port;
    feed(context, processor, block, (block += 55), value);
    await settle();
    reader.gate.release();
    feed(context, processor, block, (block += 75), value);
    await settle();
    timers.shift()!();
    await settle();
    takes.push(reader.take.samples);
  }
  // 130 blocks of 128 source samples: 17 frames of 320 plus a 107 sample tail.
  assert.ok(takes[1]!.every((sample) => sample === 0), "second take holds no earlier samples");
  assert.equal(takes[1]!.length, 17 * 320 + 107);
  assert.equal(takes[0]!.length, 17 * 320 + 107, "first take keeps its drained tail");
  assert.ok(takes[0]!.every((sample) => sample === 1));
  assert.equal(posted.filter((data) => (data as { kind?: string }).kind === "flush").length, 2);
});

test("a stale mic denial cannot cancel the current take", async () => {
  const { reader, pending, denied } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  reader.cancelTake("cancelled");
  const retaking = reader.press();
  await waitForRequests(pending, 2);
  const current = mic();
  pending[1]!(current);
  await retaking;
  denied[0]!(new Error("late denial"));
  await opening;
  assert.equal(reader.gate.capturing, true);
  assert.equal(current.track.stopped, false);
  reader.cancelTake("done");
});

test("play replays the unsaved take without opening the mic or changing it", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  pending[0]!(mic());
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(0.3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  const take = reader.take;
  const before = Array.from(take.samples);
  await reader.play();
  assert.equal(pending.length, 1, "playback must not request the mic");
  assert.equal(reader.gate.capturing, false);
  assert.equal(reader.take, take);
  assert.deepEqual(Array.from(take.samples), before);
  assert.deepEqual(Array.from(played[0]!.samples), before);
  assert.deepEqual(fetched, []);
  assert.equal(reader.elements.play.disabled, false);
});

test("play on a revisited prompt fetches the latest saved take; T steps to earlier ones", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: true, takes: ["utt-p001-a.wav", "utt-p001-b.wav"] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, false);
  assert.equal(reader.elements.playLabel.textContent, "Play take 2 of 2");
  assert.equal(reader.elements.earlier.hidden, false);
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav"]);
  assert.equal(played.length, 1);
  reader.stepTake();
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav", "/prompt-reader/takes/utt-p001-a.wav"]);
  assert.equal(pending.length, 0, "playback must not request the mic");
  assert.equal(reader.take, null);
});

test("play is disabled on a prompt with no take", () => {
  const { reader } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, true);
  assert.equal(reader.elements.earlier.hidden, true);
});
