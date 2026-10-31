import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";

/** Execute the shipped page with controlled mic acquisition and worklet messages. */
function pageHarness() {
  const pending: Array<(stream: unknown) => void> = [];
  const constraints: unknown[] = [];
  const denied: Array<(error: Error) => void> = [];
  const connected: unknown[] = [];
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
    createMediaStreamSource(stream: unknown) {
      return { connect: () => connected.push(stream), disconnect() {} };
    }
  }
  class AudioWorkletNode {
    port = {};
    connect(node: unknown) { return node; }
  }
  const source = readFileSync(new URL("../public/prompt-reader.js", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "")
    .replace("void new PromptReaderPage().start();", "globalThis.Reader = PromptReaderPage;");
  const context = vm.createContext({
    document, AudioContext, AudioWorkletNode, HoldReleaseGate,
    Float32Array, URLSearchParams,
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
  return { reader, pending, denied, connected, constraints, expectedConstraints: vm.runInContext("CAPTURE_AUDIO_CONSTRAINTS", context) };
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
