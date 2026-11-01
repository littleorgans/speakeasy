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
  const contexts: AudioContext[] = [];
  const audioWaits: { mic?: Promise<unknown>; module?: Promise<void>; resume?: Promise<void> } = {};
  class AudioContext {
    state = "running";
    destination = {};
    constructor() { contexts.push(this); }
    audioWorklet = { async addModule() { await audioWaits.module; } };
    async resume() { await audioWaits.resume; }
    async close() { this.state = "closed"; }
    createGain() { return { gain: { value: 1 }, connect: (node: unknown) => node }; }
    createMediaStreamSource() { return { connect: (node: unknown) => node }; }
  }
  class AudioWorkletNode {
    port: PagePort = { postMessage() {} };
    constructor() { workletPorts.push(this.port); }
    connect(node: unknown) { return node; }
    disconnect() {}
  }
  const track = { stopped: false, stop() { this.stopped = true; } };
  const sockets: Socket[] = [];
  class Socket {
    static OPEN = 1;
    readyState = 0;
    failStart = false;
    listeners = new Map<string, (event: any) => void>();
    constructor() { sockets.push(this); }
    addEventListener(type: string, callback: (event: any) => void) { this.listeners.set(type, callback); }
    removeEventListener(type: string, callback: (event: any) => void) { if (this.listeners.get(type) === callback) this.listeners.delete(type); }
    send(data: string) { if (this.failStart && JSON.parse(data).type === "start") throw new Error("client send failed"); sent.push(data); }
    close() { this.readyState = 3; this.listeners.get("close")?.({}); }
    open() { this.readyState = 1; this.listeners.get("open")?.({}); }
    event(data: any) { this.listeners.get("message")?.({ data: JSON.stringify(data) }); }
  }
  class HTMLElement { matches() { return false; } closest() { return this; } }
  const source = readFileSync(new URL("../public/app.js", import.meta.url), "utf8")
    .replace(/^import .*;$/gm, "")
    .replace("new SpeakEasyRoom();", "globalThis.Room = SpeakEasyRoom;");
  const context = vm.createContext({
    ...preRoll,
    document, AudioContext, AudioWorkletNode, HoldReleaseGate, Float32Array, ArrayBuffer, HTMLElement, Error, DOMException,
    Option: function(text: string, value: string) { return { text, value }; },
    location: { protocol: "http:", host: "fake-host" },
    fetch: async () => ({ ok: false, json: async () => ({ error: "catalogue unavailable in this room test" }) }),
    localStorage: { getItem: () => JSON.stringify({ mode }), setItem() {} },
    requestAnimationFrame() {},
    performance,
    clearTimeout,
    WebSocket: Socket,
    navigator: { mediaDevices: { getUserMedia: async () => audioWaits.mic ?? ({ getTracks: () => [track] }) } },
    window: { setTimeout, clearTimeout, addEventListener(type: string, callback: (event: any) => void) { listeners.set(type, callback); } },
  });
  vm.runInContext(readFileSync(new URL("../public/capture-constraints.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/speech-observations.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/speech-output-controls.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "").replace(/^export /gm, ""), context);
  vm.runInContext(source, context);
  const room = new context.Room();
  await room.setupAudio();
  Object.assign(room.settings, { engine: "mercury-instant", pauseMs: 700, voice: "cedar" });
  const selection = { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 };
  room.output.catalog = { revision: "combined-revision", default: selection };
  room.output.models = [{ id: selection.model, provider: selection.provider, label: "Kristin", voices: [{ id: "kristin", label: "Kristin" }], speed: { min: 1, max: 1 }, readiness: { state: "ready" } }];
  room.output.draft = selection;
  room.activeSettings = { ...room.settings };
  const socket = new Socket();
  socket.readyState = 1;
  room.socket = socket;
  room.wireSocket(socket);
  room.sessionPhase = "ready";
  const capture = workletPorts[0]!;
  const frame = (value: number, length = 320) =>
    capture.onmessage!({ data: { kind: "frame", samples: new Float32Array(length).fill(value), level: 0.2 } });
  const audio = () => sent.filter((data) => data instanceof ArrayBuffer).map((data) => new Float32Array(data as ArrayBuffer));
  const commands = () => sent.filter((data) => typeof data === "string").map((data) => JSON.parse(data as string));
  const acknowledge = (changes = {}, target = socket) => target.event({ ...commands().findLast((command: any) => command.type === "select-output"), type: "output-selected", selectionRevision: 11, ...changes });
  return { room, frame, sent, audio, elements, listeners, socket, sockets, commands, acknowledge, track, audioWaits, contexts, nativeTarget: new HTMLElement() };
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
  const { room, sent, elements, acknowledge } = await roomHarness("natural");
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
  const opening = room.start();
  await settle();
  acknowledge();
  await opening;
  const start = JSON.parse(sent.at(-1) as string);
  assert.equal(start.engine, "mercury-instant");
  assert.equal(start.speechOutput.model, "kokoro-v1.0");
  assert.match(elements.get("room-transfer")!.textContent as string, /Reply speech is generated on this Mac/);
  room.settings.engine = "realtime";
  await room.start();
  assert.equal(Object.hasOwn(JSON.parse(sent.at(-1) as string), "speechOutput"), false);
});

const settle = () => new Promise((resolve) => setImmediate(resolve));

test("Enter selects the displayed default and waits for its exact acknowledgement before start", async () => {
  const h = await roomHarness("natural");
  h.room.settings.speechOutput = { ...h.room.output.draft, model: "old-preference" };
  const opening = h.room.start();
  await settle();
  assert.deepEqual(h.commands().map((command) => command.type), ["select-output"]);
  assert.equal(h.commands()[0].selection.model, "piper-kristin-medium");
  h.acknowledge({ requestId: "old-try-request", selectionRevision: 99 });
  assert.equal(h.commands().length, 1);
  h.acknowledge();
  await opening;
  const start = h.commands()[1];
  assert.equal(start.type, "start");
  assert.equal(start.selectionRevision, 11);
  assert.equal(start.catalogueRevision, "combined-revision");
  assert.equal(start.speechOutput.model, "piper-kristin-medium");
  assert.match(h.elements.get("room-transfer")!.textContent as string, /Reply speech is generated on this Mac/);
  await h.room.end();
});

for (const phase of ["mic", "module", "resume", "connect", "selection"] as const) test(`Stop while awaiting ${phase} prevents room continuation and releases audio`, async () => {
  const h = await roomHarness("natural");
  let release!: (value?: any) => void;
  const waiting = new Promise<any>((resolve) => { release = resolve; });
  if (phase === "connect") h.room.connect = () => waiting;
  else if (phase !== "selection") h.audioWaits[phase] = waiting;
  const opening = h.room.start();
  await settle();
  assert.equal(h.commands().some((command) => command.type === "start"), false);
  await h.room.end();
  release({ getTracks: () => [h.track] });
  h.acknowledge();
  await opening;
  assert.equal(h.commands().some((command) => command.type === "start"), false);
  assert.equal(h.commands().filter((command) => command.type === "select-output").length, phase === "selection" ? 1 : 0);
  assert.equal(h.track.stopped, true);
  assert.equal(h.room.micStream, null);
  assert.ok(h.contexts.every((context) => context.state === "closed"));
});

test("ending during an actual socket connection prevents select and start when it later opens", async () => {
  const h = await roomHarness("natural");
  h.room.socket = null;
  const opening = h.room.start();
  await settle();
  assert.equal(h.sockets.length, 2);
  await h.room.end();
  h.sockets[1]!.open();
  await opening;
  assert.equal(h.commands().length, 0);
  assert.equal(h.room.sessionPhase, "stopped");
});

test("Stop immediately after acknowledgement prevents the queued start continuation", async () => {
  const h = await roomHarness("natural");
  const opening = h.room.start();
  await settle();
  h.acknowledge();
  await h.room.end();
  await opening;
  assert.deepEqual(h.commands().map((command) => command.type), ["select-output", "stop"]);
});

test("a newer Enter invalidates the earlier selection and old socket traffic is ignored", async () => {
  const h = await roomHarness("natural");
  const first = h.room.start();
  await settle();
  const firstSelection = h.commands()[0];
  const second = h.room.start();
  await settle();
  h.socket.event({ ...firstSelection, type: "output-selected", selectionRevision: 99 });
  assert.equal(h.commands().some((command) => command.type === "start"), false);
  h.acknowledge();
  await Promise.all([first, second]);
  assert.equal(h.commands().filter((command) => command.type === "start").length, 1);
  const current = h.room.roomAttempt;
  h.room.socket = null;
  h.socket.event({ type: "session", phase: "stopped" });
  assert.equal(h.room.roomAttempt, current, "the replaced socket cannot mutate the current attempt");
  await h.room.end();
});

test("a changed display, mismatched acknowledgement or null selection error never starts and sends Stop", async () => {
  for (const kind of ["config", "revision", "display", "error", "client-send"]) {
    const h = await roomHarness("natural");
    const opening = h.room.start();
    await settle();
    if (kind === "display") h.room.output.draft = { ...h.room.output.draft, voice: "removed" };
    if (kind === "client-send") h.socket.failStart = true;
    if (kind === "error") h.socket.event({ type: "output-selection-error", requestId: null, message: "Selection unavailable" });
    else h.acknowledge(kind === "config" ? { selection: { ...h.room.output.draft, speed: 9 } } : kind === "revision" ? { catalogueRevision: "stale" } : {});
    await opening;
    assert.equal(h.commands().some((command) => command.type === "start"), false, kind);
    assert.equal(h.commands().at(-1).type, "stop", kind);
    assert.equal(h.room.sessionPhase, "stopped");
  }
});

test("stopped and close invalidate pending room attempts; late old socket events cannot start the next attempt", async () => {
  const h = await roomHarness("natural");
  const first = h.room.start();
  await settle();
  const oldSelection = h.commands()[0];
  h.socket.event({ type: "session", phase: "stopped" });
  await first;
  const second = h.room.start();
  await settle();
  h.socket.event({ ...oldSelection, type: "output-selected", selectionRevision: 22 });
  assert.equal(h.commands().some((command) => command.type === "start"), false);
  h.socket.close();
  await second;
  assert.equal(h.room.sessionPhase, "stopped");
});

test("unavailable restored output remains unavailable and Realtime starts with its separate voice", async () => {
  const h = await roomHarness("natural");
  h.room.output.draft = { ...h.room.output.draft, voice: "unavailable" };
  await h.room.start();
  assert.equal(h.commands().some((command) => command.type === "select-output" || command.type === "start"), false);
  assert.equal(h.room.output.draft.voice, "unavailable");
  h.sent.length = 0;
  h.room.settings.engine = "realtime";
  h.room.settings.voice = "independent-realtime-voice";
  await h.room.start();
  assert.equal(h.commands().length, 1);
  const command = h.commands()[0];
  assert.equal(command.voice, "independent-realtime-voice");
  for (const field of ["speechOutput", "catalogueRevision", "selectionRevision"]) assert.equal(Object.hasOwn(command, field), false);
  await h.room.end();
});

test("coupled room producer accepts both actual hosted catalogue selections and retires their trusted contexts on Stop", async () => {
  const { EventEmitter } = await import("node:events");
  const { createProviderRuntime } = await import("./provider-runtime.ts");
  const { createProviderCredentials } = await import("./provider-availability.ts");
  const { BrowserConversationSession } = await import("./session.ts");
  const { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_MODELS } = await import("../../speech-io/src/tts/catalogue.ts");
  type HostSocket = import("ws").WebSocket;
  type RuntimeContext = import("./session.ts").BrowserRuntimeContext;
  for (const kind of ["cartesia", "openai"]) {
    const h = await roomHarness("natural");
    let transports = 0;
    const assembly = createProviderRuntime({
      credentials: createProviderCredentials(() => ({ CARTESIA_API_KEY: "fake-cartesia", OPENAI_API_KEY: "fake-openai" })),
      readiness: () => [],
      localOutput: { create() { throw new Error("Unexpected local synthesis"); }, async quiesce() {}, async close() {} },
      localCatalogue: { default: DEFAULT_LOCAL_OUTPUT, models: LOCAL_OUTPUT_MODELS.map((model) => ({ ...model, readiness: { state: "ready", reason: null } })) },
      createSocket() { transports++; throw new Error("Unexpected provider socket"); },
      fetch: async () => { transports++; throw new Error("Unexpected provider request"); },
    });
    const outputCatalogue = () => ({ ...assembly.speechOutput.catalogue(), revision: assembly.providers.catalogue().revision });
    const catalog = outputCatalogue();
    const selection = catalog.output.find((entry) => entry.config.provider.kind === kind)!.config;
    h.room.output.catalog = catalog;
    h.room.output.models = catalog.models;
    h.room.output.draft = selection;
    const captured: { config?: unknown; context?: RuntimeContext } = {};
    class ServerSocket extends EventEmitter {
      readyState = 1;
      send(data: string) { h.socket.event(JSON.parse(data)); }
    }
    const serverSocket = new ServerSocket();
    const session = new BrowserConversationSession({
      socket: serverSocket as unknown as HostSocket, sessionId: "protected-room", runtimeConfig: {},
      outputRegistry: () => assembly.providers.registries().output,
      outputCatalogue, policy: assembly.providers.policy,
      createRuntime: async (config, context) => {
        captured.config = config.speechOutput;
        captured.context = context;
        context!.speechOutputContext!.beforeRequest!();
        return new Promise<never>((_resolve, reject) => {
          context!.speechOutputContext!.signal!.addEventListener("abort", () => reject(new Error("Fake runtime opening cancelled")), { once: true });
        });
      },
    });
    const send = h.socket.send.bind(h.socket);
    h.socket.send = (raw: string) => { send(raw); serverSocket.emit("message", Buffer.from(raw), false); };
    try {
      await h.room.start();
      await settle();
      assert.deepEqual(h.commands().map((command) => command.type), ["select-output", "start"]);
      assert.deepEqual(captured.config, selection);
      assert.equal(h.commands()[1].catalogueRevision, catalog.revision);
      assert.equal(captured.context!.speechOutputContext!.signal!.aborted, false);
      assert.equal(transports, 0, "metadata, selection and a fake opening runtime issue no speech request");
      await h.room.end();
      assert.equal(captured.context!.speechOutputContext!.signal!.aborted, true);
      assert.throws(() => captured.context!.speechOutputContext!.beforeRequest!());
    } finally {
      await session.close();
      await assembly.output.close();
    }
  }
});

test("closing Room setup stops its independent tryout", async () => {
  const { room, elements } = await roomHarness("natural");
  let stopped = 0;
  room.output.stop = () => { stopped++; };
  (elements.get("settings-dialog")!.listeners as Map<string, () => void>).get("close")!();
  assert.equal(stopped, 1);
});
