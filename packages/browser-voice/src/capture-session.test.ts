import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import * as preRoll from "../public/pre-roll.js";

const source = (name: string) => readFileSync(new URL(`../public/${name}`, import.meta.url), "utf8")
  .replace(/^import .*;$/gm, "").replace(/^export /gm, "");

/** Runs the shipped session in a VM; `globals` supply the mic and worklet node. */
function loadSession(globals: Record<string, unknown>) {
  const context = vm.createContext({ ...preRoll, Float32Array, Promise, ...globals });
  vm.runInContext(source("capture-constraints.js"), context);
  vm.runInContext(source("capture-session.js"), context);
  return vm.runInContext("CaptureSession", context);
}

/** An audio context whose worklet loads at once. */
function audioContext() {
  return { destination: {}, audioWorklet: { async addModule() {} },
    createGain: () => ({ gain: { value: 1 }, connect: (node: unknown) => node }), createMediaStreamSource: () => ({ connect() {}, disconnect() {} }) };
}

test("a session leads a take with pre roll and ends it at the worklet's flush reply", async () => {
  const posted: unknown[] = [];
  class Node { port = { onmessage: undefined as any, postMessage: (data: unknown) => posted.push(data) }; connect(node: unknown) { return node; } }
  const Session = loadSession({ navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } }, AudioWorkletNode: Node });
  const frames: unknown[] = [];
  const mic = new Session({ context: audioContext(), onFrame: (frame: unknown) => frames.push(frame) });
  await mic.start(false);
  const frame = (value: number, flush = false) => mic.node.port.onmessage({ data: { kind: "frame", samples: new Float32Array(320).fill(value), level: value, ...(flush ? { flush } : {}) } });
  for (let index = 0; index < 20; index++) frame(0.01);
  assert.equal(mic.beginTake().length, 4_800, "300 ms of idle audio leads the take and is returned for live words");
  frame(0.3);
  const ended = mic.endTake();
  assert.equal(JSON.stringify(posted), '[{"kind":"flush"}]');
  frame(0.2);
  frame(0, true);
  frame(0.9);
  const take = await ended;
  assert.equal(take.samples.length, 4_800 + 640 + 320);
  assert.equal(take.preRollSamples, 4_800);
  assert.equal(mic.recording, false);
  assert.equal(mic.ring.samples, 320, "frames after the flush reply feed the next pre roll");
});

test("a voice copy session records at the context rate with its own constraints, and its pre roll keeps 300 ms at that rate", async () => {
  const nodeOptions: unknown[] = [];
  const asked: unknown[] = [];
  class Node { port = { onmessage: undefined as any, postMessage() {} }; constructor(_context: unknown, _name: string, options: unknown) {
    nodeOptions.push(options); } connect(node: unknown) { return node; } }
  const Session = loadSession({ navigator: { mediaDevices: { getUserMedia: async (constraints: unknown) => {
    asked.push(constraints); return { getTracks: () => [] }; } } }, AudioWorkletNode: Node });
  const raw = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  const mic = new Session({ context: audioContext(), rate: 48_000, constraints: raw });
  await mic.start(false);
  assert.equal(JSON.stringify(nodeOptions), JSON.stringify([{ processorOptions: { targetRate: 48_000 } }]));
  assert.equal(JSON.stringify(asked), JSON.stringify([{ audio: raw }]));
  for (let index = 0; index < 20; index++) mic.node.port.onmessage({ data: { kind: "frame", samples: new Float32Array(960), level: 0 } });
  assert.equal(mic.beginTake().length, 14_400);
  assert.equal(new preRoll.PreRollRing().limit, 4_800, "16 kHz unless asked");
});
