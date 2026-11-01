import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as preRoll from "../public/pre-roll.js";

type PagePort = { onmessage?: (event: { data: unknown }) => void; postMessage(data: unknown): void };

function element(): Record<string, unknown> {
  const children: unknown[] = [];
  return {
    children,
    options: [],
    dataset: {},
    style: { setProperty() {} },
    value: "",
    textContent: "",
    listeners: new Map<string, (event?: unknown) => void>(),
    addEventListener(type: string, callback: (event?: unknown) => void) { (this.listeners as Map<string, (event?: unknown) => void>).set(type, callback); },
    append: (child: unknown) => children.push(child),
    replaceChildren(...nodes: unknown[]) { children.splice(0, children.length, ...nodes); },
    reportValidity: () => true,
    setAttribute() {},
  };
}

/** Execute the shipped room page with a fake mic, worklet and voice socket. */
async function roomHarness(mode: "hold" | "natural") {
  const sent: unknown[] = [];
  const listeners = new Map<string, (event: any) => void>();
  const elements = new Map<string, Record<string, unknown>>();
  const document = {
    body: { dataset: {} },
    getElementById(id: string) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    querySelectorAll: () => [],
    createElement: element,
  };
  const workletPorts: PagePort[] = [];
  class AudioContext {
    state = "running";
    destination = {};
    audioWorklet = { async addModule() {} };
    async resume() {}
    async close() {}
    createGain() { return { gain: { value: 1 }, connect: (node: unknown) => node }; }
    createMediaStreamSource() { return { connect: (node: unknown) => node }; }
  }
  class AudioWorkletNode {
    port: PagePort = { postMessage() {} };
    constructor() { workletPorts.push(this.port); }
    connect(node: unknown) { return node; }
    disconnect() {}
  }
  const track = { stop() {} };
  class HTMLElement { matches() { return false; } closest() { return this; } }
  const source = readFileSync(new URL("../public/app.js", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "")
    .replace("new SpeakEasyRoom();", "globalThis.Room = SpeakEasyRoom;");
  const context = vm.createContext({
    ...preRoll,
    document, AudioContext, AudioWorkletNode, HoldReleaseGate, Float32Array, HTMLElement,
    fetch: async () => ({ ok: false, json: async () => ({ error: "catalogue unavailable in this room test" }) }),
    localStorage: { getItem: () => JSON.stringify({ mode }), setItem() {} },
    requestAnimationFrame() {},
    performance,
    clearTimeout,
    WebSocket: { OPEN: 1 },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [track] }) } },
    window: { setTimeout, clearTimeout, addEventListener(type: string, callback: (event: any) => void) { listeners.set(type, callback); } },
  });
  vm.runInContext(readFileSync(new URL("../public/capture-constraints.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/speech-output-controls.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(source, context);
  const room = new context.Room();
  await room.setupAudio();
  room.activeSettings = { ...room.settings };
  room.socket = { readyState: 1, send: (data: unknown) => sent.push(data) };
  room.sessionPhase = "ready";
  const capture = workletPorts[0]!;
  const frame = (value: number, length = 320) =>
    capture.onmessage!({ data: { kind: "frame", samples: new Float32Array(length).fill(value), level: 0.2 } });
  const audio = () => sent.filter((data) => data instanceof ArrayBuffer).map((data) => new Float32Array(data as ArrayBuffer));
  return { room, frame, sent, audio, elements, listeners, nativeTarget: new HTMLElement() };
}

const samplesOf = (buffers: Float32Array[]) => preRoll.concat(buffers);

test("a hold take opens with the 300 ms of idle frames before its press", async () => {
  const { room, frame, audio } = await roomHarness("hold");
  // Idle frames older than 300 ms fall out of the ring and are never sent.
  for (let index = 0; index < 10; index++) frame(9);
  for (let index = 0; index < 15; index++) frame(2);
  assert.equal(audio().length, 0, "idle hold frames are not streamed");
  room.beginHold();
  for (let index = 0; index < 5; index++) frame(3);
  const take = samplesOf(audio());
  assert.equal(take.length, 4_800 + 5 * 320);
  assert.ok(take.subarray(0, 4_800).every((sample) => sample === 2), "pre roll is the idle ring");
  assert.ok(take.subarray(4_800).every((sample) => sample === 3));
});

test("each hold take carries only idle frames after the last take", async () => {
  const { room, frame, audio } = await roomHarness("hold");
  for (let index = 0; index < 15; index++) frame(2);
  room.beginHold();
  frame(3);
  room.hold.cancel();
  const first = audio().length;
  frame(4);
  room.beginHold();
  const second = samplesOf(audio().slice(first));
  assert.equal(second.length, 320);
  assert.ok(second.every((sample) => sample === 4));
});

test("natural mode streams every frame with no pre roll", async () => {
  const { frame, audio } = await roomHarness("natural");
  frame(1);
  frame(2);
  assert.deepEqual(audio().map((samples) => samples[0]), [1, 2]);
});

test("room Space preserves focused native controls and releases only its keyboard hold", async () => {
  const { room, listeners, nativeTarget } = await roomHarness("hold");
  room.hold.schedule = () => undefined;
  let prevented = 0;
  const event = { code: "Space", target: nativeTarget, preventDefault() { prevented++; } };
  listeners.get("keydown")!(event);
  assert.equal(room.hold.held, false, "Space on a focused button keeps its native activation");
  assert.equal(prevented, 0);
  room.beginHold(); // Pointer ownership must survive Space on another control.
  listeners.get("keyup")!(event);
  assert.equal(room.hold.held, true);
  room.hold.cancel();
  listeners.get("keydown")!({ ...event, target: room.elements.main });
  assert.equal(room.hold.held, true, "the main hold control still owns keyboard Space");
  listeners.get("keyup")!(event);
  assert.equal(room.hold.held, false);
  assert.equal(room.hold.releasing, true);
  room.hold.cancel();
});

test("output drafts leave the active room unchanged; saved output is sent only on next Mercury entry", async () => {
  const { room, sent, elements } = await roomHarness("natural");
  const oldOutput = { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 };
  const newOutput = { ...oldOutput, model: "kokoro-v1.0", voice: "af_heart" };
  room.settings.speechOutput = oldOutput;
  room.activeSettings = { ...room.settings, speechOutput: oldOutput };
  room.output.selection = () => newOutput;
  room.elements.engine.value = "mercury-instant";
  assert.equal(room.settings.speechOutput, oldOutput, "Try/draft does not save");
  assert.equal(room.saveSettings(), true);
  assert.equal(room.settings.speechOutput, newOutput);
  assert.equal(room.activeSettings.speechOutput, oldOutput);
  assert.equal(sent.length, 0, "saving does not send a start or change the running room");
  room.setupAudio = room.connect = async () => {};
  await room.start();
  const start = JSON.parse(sent.at(-1) as string);
  assert.equal(start.engine, "mercury-instant");
  assert.equal(start.speechOutput.model, "kokoro-v1.0");
  assert.match(elements.get("room-transfer")!.textContent as string, /Reply speech is generated on this Mac/);
  room.settings.engine = "realtime";
  await room.start();
  assert.equal(Object.hasOwn(JSON.parse(sent.at(-1) as string), "speechOutput"), false);
});

test("closing Room setup stops its independent tryout", async () => {
  const { room, elements } = await roomHarness("natural");
  let stopped = 0;
  room.output.stop = () => { stopped++; };
  (elements.get("settings-dialog")!.listeners as Map<string, () => void>).get("close")!();
  assert.equal(stopped, 1);
});
