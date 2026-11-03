import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import * as preRoll from "../public/pre-roll.js";
import { feed, loadCapture, type CaptureInstance } from "./capture-worklet-harness.ts";

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

const settle = () => new Promise((resolve) => setImmediate(resolve));

function mic() {
  const track = { label: "test mic", stopped: false, stop() { this.stopped = true; } };
  return { getTracks: () => [track], getAudioTracks: () => [track], track };
}

/**
 * A session over scripted fakes: each getUserMedia waits in `pending` (or
 * `denied`), the worklet module loads when `loaded` resolves, and `frame`
 * stands in for the worklet's port.
 */
function rig({ loaded = Promise.resolve(), ...options }: { loaded?: Promise<void>; onFrame?: (data: any) => unknown; deviceId?: string } = {}) {
  const pending: Array<(stream: unknown) => void> = [];
  const denied: Array<(error: Error) => void> = [];
  const asked: unknown[] = [];
  const connected: unknown[] = [];
  const posted: unknown[] = [];
  let modules = 0;
  let nodes = 0;
  class Node { port = { onmessage: undefined as any, postMessage: (data: unknown) => posted.push(data) };
    constructor() { nodes++; } connect(node: unknown) { return node; } disconnect() {} }
  const Session = loadSession({ AudioWorkletNode: Node, navigator: { mediaDevices: { getUserMedia: (constraints: unknown) => {
    asked.push(constraints);
    return new Promise((resolve, reject) => { pending.push(resolve); denied.push(reject); });
  } } } });
  const context = { ...audioContext(), audioWorklet: { addModule: async () => { modules++; await loaded; } },
    createMediaStreamSource: (stream: unknown) => ({ connect: () => connected.push(stream), disconnect() {} }) };
  const session = new Session({ context, ...options });
  const frame = (value: number, flush = false) => session.node.port.onmessage({ data: { kind: "frame", samples: new Float32Array(320).fill(value), level: value, ...(flush ? { flush } : {}) } });
  const requested = async (count: number) => {
    for (let tries = 0; pending.length < count && tries < 20; tries++) await settle();
    assert.equal(pending.length, count);
  };
  return { session, pending, denied, asked, connected, posted, frame, requested, get modules() { return modules; }, get nodes() { return nodes; } };
}

const flushes = (posted: unknown[]) => posted.filter((data) => (data as { kind?: string }).kind === "flush").length;

test("each take drains its worklet tail at 48 kHz, so a silent second take holds none of the first", async () => {
  // Session and real worklet, connected through asynchronous ports.
  let processor: CaptureInstance | undefined;
  let session: any;
  const { context, Processor } = loadCapture(48_000, (message) => queueMicrotask(() => session.node.port.onmessage({ data: message })));
  class Node { port = { onmessage: undefined as any, postMessage: (data: unknown) => queueMicrotask(() => processor!.port.onmessage?.({ data })) };
    constructor(_context: unknown, _name: string, options: { processorOptions?: { targetRate?: number } }) { processor = new Processor(options); }
    connect(node: unknown) { return node; } }
  const Session = loadSession({ AudioWorkletNode: Node, navigator: { mediaDevices: { getUserMedia: async () => mic() } } });
  session = new Session({ context: audioContext() });
  await session.start();
  const takes: Float32Array[] = [];
  let block = 0;
  for (const value of [1, 0]) {
    session.beginTake();
    feed(context, processor!, block, (block += 130), value);
    await settle();
    takes.push((await session.endTake()).samples);
  }
  // 130 blocks of 128 source samples: 17 frames of 320 plus a 107 sample tail.
  assert.ok(takes[1]!.every((sample) => sample === 0), "second take holds no earlier samples");
  assert.equal(takes[1]!.length, 17 * 320 + 107);
  assert.equal(takes[0]!.length, 17 * 320 + 107, "first take keeps its drained tail");
  assert.ok(takes[0]!.every((sample) => sample === 1));
});

test("stop during a pending start discards the late stream, and a late denial never reaches the next caller", async () => {
  const { session, pending, denied, connected, requested } = rig();
  const first = session.start();
  await requested(1);
  session.stop();
  const second = session.start();
  await requested(2);
  const stale = mic();
  const current = mic();
  pending[0]!(stale);
  await first;
  assert.equal(stale.track.stopped, true);
  assert.deepEqual(connected, []);
  pending[1]!(current);
  await second;
  assert.deepEqual(connected, [current]);
  assert.equal(session.device, "test mic");
  session.stop();
  assert.equal(current.track.stopped, true);
  assert.equal(session.device, "");
  const denial = session.start();
  await requested(3);
  session.stop();
  const retry = session.start();
  await requested(4);
  denied[2]!(new Error("late denial"));
  await denial;
  pending[3]!(mic());
  await retry;
  assert.equal(session.open, true);
});

test("concurrent starts share one acquisition and one worklet load", async () => {
  let load!: () => void;
  const counts = rig({ loaded: new Promise((resolve) => { load = resolve; }), deviceId: "usb" });
  const { session, pending, asked, requested } = counts;
  const starts = [session.start(), session.start()];
  load();
  await requested(1);
  starts.push(session.start());
  pending[0]!(mic());
  await Promise.all(starts);
  await session.start();
  assert.equal(counts.modules, 1);
  assert.equal(counts.nodes, 1);
  assert.equal(JSON.stringify(asked), JSON.stringify([{ audio: { deviceId: { exact: "usb" }, ...session.constraints } }]), "the device leads the shared constraints");
});

test("stop while a take drains resolves it null and drops the frames until each reply", async () => {
  const { session, pending, posted, frame, requested } = rig();
  const opening = session.start();
  await requested(1);
  pending[0]!(mic());
  await opening;
  session.beginTake();
  frame(0.3);
  const ended = session.endTake();
  assert.equal(session.draining, true);
  session.stop();
  assert.equal(session.draining, false);
  assert.equal(await ended, null);
  assert.equal(flushes(posted), 2, "stop flushes the worklet's partial too");
  const reopening = session.start();
  await requested(2);
  pending[1]!(mic());
  await reopening;
  frame(0.5);
  frame(0, true);
  frame(0.5);
  frame(0, true);
  assert.equal(session.ring.samples, 0, "frames before both replies reach no take and no pre roll");
  frame(0.7);
  assert.equal(session.ring.samples, 320);
});

test("a take ended before the worklet loads resolves at once with what it holds", async () => {
  const { session, posted } = rig({ loaded: new Promise(() => {}) });
  void session.start();
  session.beginTake();
  const take = await session.endTake();
  assert.equal(take.samples.length, 0);
  assert.equal(take.preRollSamples, 0);
  assert.equal(flushes(posted), 0);
});

test("onFrame returning false keeps a frame out of the pre roll and the take", async () => {
  const seen: number[] = [];
  const { session, pending, frame, requested } = rig({ onFrame: (data) => { seen.push(data.level); return data.level !== 0.9; } });
  const opening = session.start();
  await requested(1);
  pending[0]!(mic());
  await opening;
  frame(0.9);
  frame(0.2);
  assert.equal(session.ring.samples, 320);
  assert.equal(session.takePreRoll()[0], Math.fround(0.2));
  assert.equal(session.ring.samples, 0, "the pre roll is taken once");
  session.beginTake();
  frame(0.9);
  frame(0.3);
  const ended = session.endTake();
  frame(0, true);
  const take = await ended;
  assert.deepEqual(Array.from(new Set(take.samples)), [Math.fround(0.3), 0], "the take frame, then the flush reply's zeros");
  assert.deepEqual(seen, [0.9, 0.2, 0.9, 0.3], "draining frames skip onFrame");
});

test("a cold start keeps a take begun before it", async () => {
  const { session, pending, frame, requested } = rig();
  session.beginTake();
  const opening = session.start();
  await requested(1);
  pending[0]!(mic());
  await opening;
  assert.equal(session.recording, true);
  frame(0.3);
  const ended = session.endTake();
  frame(0, true);
  const take = await ended;
  assert.equal(take.samples.length, 640, "the frame and the flush reply");
  assert.equal(take.samples[0], Math.fround(0.3));
});
