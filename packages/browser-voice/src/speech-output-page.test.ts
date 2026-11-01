import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import type { SpeechOutputConfig } from "@speakeasy/speech-io/providers";
import { parseSpeechOutputCommand } from "./speech-output-protocol.ts";

const local: SpeechOutputConfig = { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 };
const catalogue = {
  schema: 1, revision: "test-catalogue", default: local, maxTextLength: 2000,
  models: [
    { id: local.model, label: "Piper Kristin medium", provider: local.provider, defaultVoice: "kristin", voices: [{ id: "kristin", label: "Kristin" }], defaultSpeed: 1, speed: { min: 1, max: 1 }, configured: true, readiness: { state: "ready" } },
    { id: "kokoro-v1.0", label: "Kokoro", provider: local.provider, defaultVoice: "af_heart", voices: [{ id: "af_heart", label: "Heart" }, { id: "af_bella", label: "Bella" }], defaultSpeed: 1, speed: { min: 1, max: 1 }, configured: true, readiness: { state: "ready" } },
    { id: "test-api-model", label: "Configured API", provider: { kind: "cartesia" }, defaultVoice: "voice-a", voices: [{ id: "voice-a", label: "Voice A" }], defaultSpeed: 1, speed: { min: 1, max: 1 }, configured: true, readiness: { state: "ready" } },
  ],
};

type Port = { messages: any[]; onmessage?: (event: { data: any }) => void; postMessage(data: any): void };

async function outputHarness({ stored, active = false, catalog = catalogue, audioTime = 4, setupMs = 0 }: { stored?: unknown; active?: boolean; catalog?: any; audioTime?: number; setupMs?: number } = {}) {
  const elements = new Map<string, any>();
  const element = () => ({ value: "", disabled: false, hidden: false, textContent: "", children: [] as any[], listeners: new Map<string, (event?: any) => void>(), addEventListener(type: string, callback: (event?: any) => void) { this.listeners.set(type, callback); }, append(...nodes: any[]) { this.children.push(...nodes); }, replaceChildren(...nodes: any[]) { this.children = nodes; } });
  const document = { createElement: element, getElementById(id: string) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); } };
  document.getElementById("output-text").value = "  The room is ready.\nTake your time.  ";
  const ports: Port[] = [];
  const contexts: any[] = [];
  const sockets: Socket[] = [];
  const fetches: string[] = [];
  class AudioContext {
    state = "running";
    currentTime = audioTime;
    outputTimestamp = { contextTime: audioTime, performanceTime: now };
    destination = {};
    audioWorklet = { async addModule() { now += setupMs; } };
    listeners = new Map<string, () => void>();
    addEventListener(type: string, callback: () => void) { this.listeners.set(type, callback); }
    constructor() { contexts.push(this); }
    getOutputTimestamp() { return this.outputTimestamp; }
    async resume() {}
    async close() { this.state = "closed"; }
  }
  class AudioWorkletNode {
    port: Port = { messages: [], postMessage(data) { this.messages.push(data); } };
    constructor() { ports.push(this.port); }
    connect() {}
    disconnect() {}
  }
  class Socket {
    static OPEN = 1;
    readyState = 0;
    listeners = new Map<string, (event: any) => void>();
    sent: any[] = [];
    constructor() { sockets.push(this); }
    addEventListener(type: string, callback: (event: any) => void) { this.listeners.set(type, callback); }
    send(raw: string) { this.sent.push(JSON.parse(raw)); }
    close() { this.readyState = 3; this.listeners.get("close")?.({}); }
    open() { this.readyState = 1; this.listeners.get("open")?.({}); }
    event(data: any) { this.listeners.get("message")?.({ data: data instanceof ArrayBuffer ? data : JSON.stringify(data) }); }
  }
  let now = 1000;
  const context = vm.createContext({
    document, AudioContext, AudioWorkletNode, WebSocket: Socket, Float32Array, ArrayBuffer, DataView,
    Option: function(text: string, value: string) { return { text, value }; },
    location: { protocol: "http:", host: "fake-host" }, performance: { now: () => now },
    fetch: async (url: string) => { fetches.push(url); return { ok: true, json: async () => catalog }; },
  });
  vm.runInContext(readFileSync(new URL("../public/speech-observations.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/speech-output-controls.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "").replace(/^export /gm, "") + "\nglobalThis.Controls = SpeechOutputControls;", context);
  const controls = new context.Controls({ activeRoom: () => active });
  await controls.load(stored);
  const click = (id: string, type = "click") => elements.get(id).listeners.get(type)?.();
  const start = async () => {
    await controls.tryVoice();
    const socket = sockets.at(-1)!;
    socket.open();
    const requestId = controls.job.requestId;
    socket.event({ type: "output-selected", requestId, catalogueRevision: catalog.revision, selectionRevision: 5, selection: controls.selection() });
    return { socket, requestId, port: ports.at(-1)! };
  };
  const accepted = (socket: Socket, requestId: string, playbackId = 7) => {
    socket.event({ type: "sample-accepted", requestId, playbackId, generation: 2, selection: controls.selection(), text: elements.get("output-text").value, textId: "typed-text-id" });
    socket.event({ type: "playback", requestId, action: "start", playbackId, sampleRate: 24_000 });
  };
  return { controls, elements, ports, contexts, sockets, fetches, click, start, accepted, advance: (value: number) => { now = value; } };
}

function packet(playbackId: number, sampleRate = 24_000) {
  const bytes = new ArrayBuffer(17);
  const view = new DataView(bytes);
  view.setUint8(0, 1); view.setUint32(1, playbackId, true); view.setUint32(5, sampleRate, true);
  view.setFloat32(9, 0.25, true); view.setFloat32(13, -0.25, true);
  return bytes;
}

test("room selection uses its own socket and exact current snapshot, never the Try acknowledgement", async () => {
  const h = await outputHarness();
  const trial = await h.start();
  const commands: any[] = [];
  const room = { readyState: 1, send: (raw: string) => commands.push(JSON.parse(raw)) };
  const waiting = h.controls.selectForRoom(room);
  const sent = commands[0];
  assert.equal(sent.type, "select-output");
  assert.notEqual(sent.requestId, trial.requestId);
  const acknowledgement = { ...sent, type: "output-selected", selectionRevision: 19 };
  h.controls.acceptRoom(trial.socket, acknowledgement);
  h.controls.acceptRoom(room, { ...acknowledgement, requestId: trial.requestId });
  assert.ok(h.controls.roomSelection, "wrong socket/request cannot resolve the pending exchange");
  h.controls.acceptRoom(room, acknowledgement);
  const selected = await waiting;
  assert.deepEqual(JSON.parse(JSON.stringify(selected)), { speechOutput: local, catalogueRevision: catalogue.revision, selectionRevision: 19 });
  assert.equal(h.controls.job.requestId, trial.requestId, "room selection does not replace the Try job");
  h.controls.stop();
});

for (const [name, change] of Object.entries({
  catalogue: { catalogueRevision: "stale" },
  selection: { selection: { ...local, speed: 1.1 } },
  "typed voice": { selection: { ...local, voice: 1 } },
  revision: { selectionRevision: 0 },
})) test(`room rejects mismatched ${name} without starting or borrowing another permit`, async () => {
  const h = await outputHarness();
  const sent: any[] = [];
  const socket = { readyState: 1, send: (raw: string) => sent.push(JSON.parse(raw)) };
  const waiting = h.controls.selectForRoom(socket);
  const rejection = assert.rejects(waiting, /match|selection/i);
  h.controls.acceptRoom(socket, { ...sent[0], type: "output-selected", selectionRevision: 4, ...change });
  await rejection;
  assert.equal(h.controls.roomSelection, null);
  assert.equal(sent.length, 1);
});

test("room cancellation, replacement, null errors and stopped events retire pending exchanges", async () => {
  const h = await outputHarness();
  const sent: any[] = [];
  const socket = { readyState: 1, send: (raw: string) => sent.push(JSON.parse(raw)) };
  for (const event of [null, { type: "output-selection-error", requestId: null, message: "Selection unavailable" }, { type: "session", phase: "stopped" }]) {
    const pending = h.controls.selectForRoom(socket);
    const rejection = assert.rejects(pending);
    if (event) h.controls.acceptRoom(socket, event);
    else h.controls.cancelRoomSelection();
    await rejection;
  }
  const first = h.controls.selectForRoom(socket);
  const rejected = assert.rejects(first);
  const latest = h.controls.selectForRoom(socket);
  await rejected;
  h.controls.acceptRoom(socket, { ...sent.at(-2), type: "output-selected", selectionRevision: 2 });
  assert.ok(h.controls.roomSelection);
  h.controls.acceptRoom(socket, { ...sent.at(-1), type: "output-selected", selectionRevision: 3 });
  assert.equal((await latest).selectionRevision, 3);
});

test("produced and drained snapshots retain exact requested text and each real request's observations", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  const measured = (ms: number) => ({ status: "measured", ms, clock: "server", boundary: "provider request" });
  const unavailable = { status: "unavailable", reason: "not exposed" };
  const requests = [
    { model: "applied-a", voice: "resolved-a", text: "First <sentence>.", returnedTranscript: "First <sentence>.", timing: { providerRequestWallMs: measured(0), providerProcessingMs: unavailable, providerNetworkMs: unavailable } },
    { model: "applied-b", voice: "resolved-b", text: "Second sentence.", returnedTranscript: "", timing: { providerRequestWallMs: measured(25), providerProcessingMs: unavailable, providerNetworkMs: unavailable }, sessionSetupWallMs: measured(7), providerFirstAudioMs: measured(9) },
    { model: "applied-c", voice: 0, text: "Third.", returnedTranscript: null, timing: { providerRequestWallMs: measured(11), providerProcessingMs: unavailable, providerNetworkMs: unavailable } },
  ];
  const completion = { type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", timing: { serverFirstAudioReadyMs: measured(0) },
    segments: [{ index: 0, sentence: "First <sentence>.", sampleRate: 24000, audioDurationMs: 500, timing: {} }], requests };
  socket.event({ ...completion, outcome: "produced" });
  assert.equal(h.controls.result.text, "  The room is ready.\nTake your time.  ");
  assert.equal(h.controls.result.generation, 2);
  assert.deepEqual(JSON.parse(JSON.stringify(h.controls.result.requests)), requests);
  const detailText = () => h.elements.get("output-observations").children.map((node: any) => node.textContent).join("\n");
  assert.match(detailText(), /First <sentence>\./);
  assert.match(detailText(), /0 ms.*measured.*server/);
  assert.match(detailText(), /Transcript: same/);
  assert.match(detailText(), /Returned transcript: "".*Transcript: different/);
  assert.match(detailText(), /Returned transcript: Unavailable.*Transcript: unavailable/);
  assert.match(detailText(), /Setup wall: 7 ms.*First provider PCM: 9 ms/);
  assert.match(detailText(), /Provider network: Unavailable: not exposed/);
  socket.event({ ...completion, textId: "stale-text", requests: [], outcome: "drained" });
  assert.equal(h.controls.result.textId, "typed-text-id");
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 500 } });
  socket.event({ ...completion, outcome: "drained" });
  assert.equal(h.controls.job, null);
  assert.equal(h.controls.result.outcome, "drained");
  h.elements.get("output-model").value = "kokoro-v1.0";
  h.click("output-model", "change");
  assert.equal(h.controls.result.selection.model, local.model);
  const next = await h.start();
  assert.equal(h.controls.result, null, "new trial clears the prior snapshot");
  socket.event({ ...completion, outcome: "produced" });
  assert.equal(h.controls.result, null, "late old completion cannot label the new job");
  h.accepted(next.socket, next.requestId);
  h.controls.stop();
});

test("catalogue and selector changes only change the draft; Try preserves exact typed text", async () => {
  const h = await outputHarness();
  assert.deepEqual(h.fetches, ["/speech-output/models"]);
  h.elements.get("output-provider").value = "cartesia";
  h.click("output-provider", "change");
  assert.equal(h.sockets.length, 0);
  assert.equal(h.contexts.length, 0);
  assert.match(h.elements.get("output-transfer").textContent, /Text is sent to cartesia/);
  const { socket } = await h.start();
  assert.equal(socket.sent[0].type, "select-output");
  assert.equal(socket.sent[1].type, "speak-sample");
  assert.equal(socket.sent[1].selectionRevision, 5);
  assert.equal(socket.sent[1].text, h.elements.get("output-text").value);
  assert.equal(socket.sent[1].selection.provider.kind, "cartesia");
  assert.equal(Object.hasOwn(socket.sent[1], "source"), false);
  const registry = catalogue.models.map((model) => ({ kind: model.provider.kind as "local" | "cartesia", model: model.id, voices: model.voices.map((voice) => voice.id), speed: model.speed }));
  for (const command of socket.sent) assert.doesNotThrow(() => parseSpeechOutputCommand(JSON.stringify(command), registry));
  h.controls.stop();
});

test("Try needs no microphone or LLM and accepts only the current playback's PCM", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  socket.event(packet(7));
  assert.equal(port.messages.length, 0, "no PCM before sample acceptance and start");
  h.accepted(socket, requestId);
  socket.event(packet(6));
  socket.event(packet(7, 16_000));
  socket.event(packet(7));
  assert.deepEqual(port.messages.map((message) => message.kind), ["start", "audio"]);
  assert.equal(port.messages[0].reportFirstRender, true);
  assert.deepEqual(Array.from(port.messages[1].samples), [0.25, -0.25]);
  h.controls.stop();
});

test("Stop clears locally and isolates stale events, PCM and render metrics from the next try", async () => {
  const h = await outputHarness();
  const first = await h.start();
  h.accepted(first.socket, first.requestId);
  h.controls.stop();
  assert.equal(first.port.messages.at(-1).kind, "clear");
  assert.equal(first.socket.sent.at(-1).type, "cancel-sample");
  assert.equal(h.contexts[0].state, "closed");
  const second = await h.start();
  h.accepted(second.socket, second.requestId);
  first.socket.event(packet(7));
  first.socket.event({ type: "sample-error", requestId: first.requestId, message: "stale failure" });
  first.port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.5 } });
  assert.equal(h.controls.browserMs, null);
  assert.equal(h.controls.error, "");
  assert.deepEqual(second.port.messages.map((message) => message.kind), ["start"]);
  h.controls.stop();
});

test("host clear silences playback while preserving the following trial failure", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  socket.event({ type: "playback", requestId, playbackId: 7, action: "clear" });
  socket.event(packet(7));
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 12 } });
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.25 } });
  socket.event({ type: "sample-error", requestId, playbackId: 7, code: "TIMEOUT", message: "Speech trial timed out before playback completed." });
  assert.match(h.controls.error, /timed out/);
  assert.deepEqual(port.messages.map((message) => message.kind), ["start", "clear"]);
  assert.equal(socket.sent.some((command) => command.type === "playback-drained"), false);
  assert.equal(h.controls.status, "Voice unavailable");
  assert.equal(h.controls.browserMs, null);
  assert.equal(h.controls.job, null);
  assert.equal(h.contexts[0].state, "closed");
});

test("browser TTFA uses the audio clock and Finished waits for server and browser drain", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  h.advance(9000); // Delivery time is deliberately much later than rendering.
  h.contexts[0].currentTime = 12;
  h.contexts[0].outputTimestamp = { contextTime: 11.9, performanceTime: 8900 };
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.125, sampleOffset: 70 } });
  assert.equal(h.controls.browserMs, 125);
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.5 } });
  assert.equal(h.controls.browserMs, 125, "first observation only");
  socket.event({ type: "playback", requestId, action: "end", playbackId: 7 });
  const timing = { serverFirstAudioReadyMs: { status: "measured", ms: 42, clock: "server", boundary: "acceptance to PCM" } };
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "produced", timing, segments: [{}] });
  assert.notEqual(h.controls.status, "Finished");
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 40 } });
  assert.equal(socket.sent.at(-1).type, "playback-drained");
  assert.notEqual(h.controls.status, "Finished");
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "drained", timing, segments: [{}] });
  assert.equal(h.controls.status, "Finished");
  assert.equal(h.controls.hostMs, 42);
  assert.equal(h.controls.job, null);
});

for (const { name, outputTimestamp, expected } of [
  { name: "native audio clock start delay uses the browser supplied clock origin",
    outputTimestamp: { contextTime: 0.01, performanceTime: 83789.3 }, expected: 131.632653 },
  { name: "native startup without an output clock position finishes with TTFA unavailable",
    outputTimestamp: { contextTime: 0, performanceTime: 83801.80000001192 }, expected: null },
]) test(name, async () => {
  // Saved native start delay; only the positive output timestamp pair is synthetic.
  const h = await outputHarness({ audioTime: 0.005804988662131519, setupMs: 24.1 });
  h.advance(83659.30000001192);
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  h.contexts[0].currentTime = 0.017414965986394557;
  h.contexts[0].outputTimestamp = outputTimestamp;
  h.advance(83802.20000004768);
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 0.01163265306122449, sampleOffset: 1 } });
  if (expected == null) assert.equal(h.controls.browserMs, null, "zero output position cannot establish a clock origin");
  else assert.ok(Math.abs(h.controls.browserMs - expected) < 0.000001);
  assert.notEqual(h.controls.browserMs, 83802.20000004768 - 83659.30000001192, "callback delivery is a separate boundary");
  const timing = { serverFirstAudioReadyMs: { status: "measured", ms: 80.36745899997186 } };
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "produced", timing, segments: [{}] });
  socket.event({ type: "playback", requestId, playbackId: 7, action: "end" });
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 40 } });
  assert.equal(socket.sent.at(-1).type, "playback-drained");
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "drained", timing, segments: [{}] });
  assert.equal(h.controls.job, null);
  assert.equal(h.controls.hostMs, timing.serverFirstAudioReadyMs.ms);
  if (expected == null) {
    assert.equal(h.controls.browserMs, null);
    assert.match(h.controls.status, /Finished.*clock.*unavailable/i);
    assert.doesNotMatch(h.controls.status, /No nonzero audio/);
    assert.match(h.elements.get("output-timing").textContent, /Browser TTFA: Unavailable/);
  } else {
    assert.equal(h.controls.status, "Finished");
    assert.match(h.elements.get("output-timing").textContent, /Browser TTFA: 132 ms/);
  }
});

for (const { name, getOutputTimestamp } of [
  { name: "missing output timestamp support", getOutputTimestamp: undefined },
  { name: "missing output timestamp result", getOutputTimestamp: () => undefined },
  { name: "zero output timestamp performance position", getOutputTimestamp: () => ({ contextTime: 4, performanceTime: 0 }) },
  { name: "nonfinite output timestamp position", getOutputTimestamp: () => ({ contextTime: NaN, performanceTime: 1000 }) },
  { name: "throwing output timestamp query", getOutputTimestamp: () => { throw new Error("Output timestamp unavailable."); } },
]) test(`${name} keeps the first render unavailable without replacing it`, async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  h.contexts[0].getOutputTimestamp = getOutputTimestamp;
  assert.doesNotThrow(() => port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.125 } }));
  assert.equal(h.controls.job.rendered, true, "nonzero render is observed even without timing");
  assert.equal(h.controls.browserMs, null);
  h.contexts[0].getOutputTimestamp = () => ({ contextTime: 4.5, performanceTime: 1500 });
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.5 } });
  assert.equal(h.controls.browserMs, null, "a later notification cannot replace the first render");
  const timing = { serverFirstAudioReadyMs: { status: "measured", ms: 42 } };
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "produced", timing, segments: [{}] });
  socket.event({ type: "playback", requestId, action: "end", playbackId: 7 });
  assert.notEqual(h.controls.job, null, "production alone cannot finish playback");
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 40 } });
  assert.equal(socket.sent.at(-1).type, "playback-drained");
  assert.notEqual(h.controls.job, null, "browser drain still waits for service completion");
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "drained", timing, segments: [{}] });
  assert.equal(h.controls.job, null);
  assert.equal(h.controls.browserMs, null);
  assert.equal(h.controls.hostMs, 42);
  assert.match(h.controls.status, /Finished.*clock.*unavailable/i);
  assert.doesNotMatch(h.controls.status, /No nonzero audio/);
  assert.match(h.elements.get("output-timing").textContent, /Browser TTFA: Unavailable/);
});

test("editing sample text or model stops a preview and never saves or restarts it", async () => {
  const h = await outputHarness();
  const { socket, requestId } = await h.start();
  h.accepted(socket, requestId);
  h.elements.get("output-text").value = "new typed text";
  h.click("output-text", "input");
  assert.equal(h.controls.job, null);
  h.elements.get("output-model").value = "kokoro-v1.0";
  h.click("output-model", "change");
  assert.equal(h.controls.selection().voice, "af_heart");
  assert.equal(h.elements.get("output-text").value, "new typed text");
  assert.equal(h.sockets.length, 1);
});

test("missing, stale and invalid selections stay unavailable without substitution", async () => {
  const h = await outputHarness({ stored: { ...local, voice: "removed-voice" } });
  assert.equal(h.controls.selection(), null);
  assert.equal(h.controls.draft.voice, "removed-voice");
  assert.equal(h.elements.get("output-try").disabled, true);
  await h.controls.tryVoice();
  assert.equal(h.sockets.length, 0);
  assert.match(h.elements.get("output-error").textContent, /Saved speech output is unavailable/);
  await h.controls.load();
  assert.equal(h.controls.draft.voice, "removed-voice", "metadata refresh retains the unavailable preference");
  assert.equal(h.controls.selection(), null);
});

test("active room, blank text, too long text and missing assets prevent preview", async () => {
  const busy = await outputHarness({ active: true });
  await busy.controls.tryVoice();
  assert.equal(busy.sockets.length, 0);
  assert.match(busy.elements.get("output-status").textContent, /End the room/);
  const h = await outputHarness();
  for (const text of [" \n ", "x".repeat(2001)]) {
    h.elements.get("output-text").value = text;
    h.click("output-text", "input");
    await h.controls.tryVoice();
    assert.equal(h.sockets.length, 0);
    assert.equal(h.elements.get("output-try").disabled, true);
  }
  const missing = await outputHarness({ catalog: { ...catalogue, models: [{ ...catalogue.models[0], readiness: { state: "assets-missing" } }] } });
  await missing.controls.tryVoice();
  assert.equal(missing.sockets.length, 0);
  assert.equal(missing.elements.get("output-status").textContent, "Voice needs setup");
});

test("Try and Stop are native buttons that do not submit Room setup", () => {
  const html = readFileSync(new URL("../public/index.html", import.meta.url), "utf8");
  for (const id of ["output-try", "output-stop"]) assert.match(html, new RegExp(`<button[^>]*id="${id}"[^>]*type="button"`));
});

test("a local drain before production is held until the service can accept it", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 5 } });
  assert.equal(socket.sent.some((command) => command.type === "playback-drained"), false);
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "produced", timing: {}, segments: [{}] });
  assert.equal(socket.sent.at(-1).type, "playback-drained");
  h.controls.stop();
});

test("only configured API models are offered; stale speed is not restored as supported", async () => {
  const hidden = { ...catalogue.models[2], id: "unconfigured", provider: { kind: "openai" }, configured: false };
  const h = await outputHarness({ catalog: { ...catalogue, models: [...catalogue.models, hidden] } });
  assert.equal(h.elements.get("output-provider").children.some((option: any) => option.value === "openai"), false);
  const stale = await outputHarness({ stored: { ...local, speed: 2 } });
  assert.equal(stale.controls.selection(), null);
});

for (const resumed of [false, true]) test(`suspension ${resumed ? "followed by resume" : "still active"} leaves browser TTFA unavailable`, async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  h.contexts[0].state = "suspended";
  h.contexts[0].listeners.get("statechange")();
  if (resumed) h.contexts[0].state = "running";
  port.onmessage?.({ data: { kind: "first-render", playbackId: 7, audioTime: 4.25 } });
  assert.equal(h.controls.browserMs, null);
  h.controls.stop();
});

test("silent audio drains successfully while first nonzero render timing stays unavailable", async () => {
  const h = await outputHarness();
  const { socket, requestId, port } = await h.start();
  h.accepted(socket, requestId);
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "produced", timing: {}, segments: [{}] });
  socket.event({ type: "playback", requestId, action: "end", playbackId: 7 });
  port.onmessage?.({ data: { kind: "drained", playbackId: 7, audioEndMs: 50 } });
  socket.event({ type: "sample-complete", requestId, playbackId: 7, selection: local, textId: "typed-text-id", outcome: "drained", timing: {}, segments: [{}] });
  assert.equal(h.controls.job, null);
  assert.equal(h.controls.browserMs, null);
  assert.match(h.controls.status, /No nonzero audio rendered/);
});
