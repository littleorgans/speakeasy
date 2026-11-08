import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as text from "../public/road-test-text.js";
import * as trail from "../public/live-trail.js";
import { finalSetup } from "./final-settings.ts";
import { liveStream } from "./live-settings.ts";
import { QWEN_COPY_SETTINGS, QWEN_COPY_VOICES, QWEN_CUSTOM_17B_SETTINGS, QWEN_CUSTOM_SETTINGS, QWEN_SPEAKERS }
  from "../../speech-io/src/tts/qwen-voices.ts";
import { END_SILENCE_MS, TalkOverDetector } from "../public/talk-over.js";
import { VoiceReadback } from "../public/voice-readback.js";

const source = (name: string) => readFileSync(new URL(`../public/${name}`, import.meta.url), "utf8")
  .replace(/^import .*;$/gm, "").replace(/^export /gm, "");

test("talk over needs a window of loud frames and reports the first frame's start", () => {
  const detector = new TalkOverDetector();
  let time = 1;
  const push = (level: number) => detector.push(level, time += 0.02);
  for (let index = 0; index < 20; index++) assert.equal(push(0.002), null);
  assert.equal(push(0.2), null);
  assert.equal(push(0.002), null, "a single click is not a voice");
  const first = time + 0.02;
  push(0.2);
  push(0.2);
  const onset = push(0.2);
  assert.equal(onset?.type, "onset");
  assert.ok(Math.abs(onset!.onsetTime - (first - 0.02)) < 1e-9);
  let end = null;
  const quietStart = time;
  while (!end) end = push(0.002);
  assert.equal(end.type, "end");
  assert.ok((time - quietStart) * 1_000 >= END_SILENCE_MS);
});

test("talk over tracks the noise floor, so steady echo below three times it never counts", () => {
  const detector = new TalkOverDetector();
  let time = 0;
  for (let index = 0; index < 200; index++) detector.push(0.03, time += 0.02);
  for (let index = 0; index < 10; index++) assert.equal(detector.push(0.06, time += 0.02), null);
  for (let index = 0; index < 2; index++) detector.push(0.2, time += 0.02);
  assert.equal(detector.push(0.2, time += 0.02)?.type, "onset");
});

test("model cards are plain text: where it runs, size, runtime and terms", () => {
  assert.equal(text.inputCard({ readiness: { ready: true, sizeBytes: 2_509_042_622 },
    card: { runtime: "parakeet-mlx 0.5.3 on MLX, Metal GPU", transcript: "final" } }),
    "Local · 2.5 GB · parakeet-mlx 0.5.3 on MLX, Metal GPU", "the Live words choice says whether words show while you speak");
  assert.equal(text.inputCard({ readiness: { ready: false, reason: "runtime or model files need setup" },
    card: { runtime: "speech.cpp", transcript: "partials" } }),
    "Local · size unknown · speech.cpp · needs setup: runtime or model files need setup");
  assert.equal(text.outputCard({ provider: { kind: "local" }, sizeBytes: 63_537_415, runtime: "sherpa-onnx VITS, CPU",
    streaming: "sentence", readiness: { state: "ready" } }), "Local · 64 MB · sherpa-onnx VITS, CPU · streams by sentence");
  assert.equal(text.inputCard({ readiness: { ready: true, sizeBytes: 731_357_568 },
    card: { runtime: "transcribe.cpp", transcript: "final", terms: "NVIDIA Open Model License" } }),
  "Local · 731 MB · transcribe.cpp · Terms:\u00a0NVIDIA\u00a0Open\u00a0Model\u00a0License",
  "a terms line where not permissive, never split across lines");
  assert.equal(text.outputCard({ provider: { kind: "local" }, sizeBytes: 1, runtime: "libpiper", streaming: "sentence",
    terms: "GPL 3.0 runtime", readiness: { state: "ready" } }), "Local · 1 MB · libpiper · streams by sentence · Terms:\u00a0GPL\u00a03.0\u00a0runtime");
  assert.equal(text.outputCard({ provider: { kind: "local" }, sizeBytes: 1, runtime: "pocket", terms: "CC BY 4.0, credit required",
    readiness: { state: "ready" } }), "Local · 1 MB · pocket · Terms:\u00a0CC\u00a0BY\u00a04.0, credit\u00a0required",
  "long terms break only after their punctuation");
  assert.equal(text.formatDuration(312.4), "312 ms");
  assert.equal(text.formatDuration(1_234), "1.23 s");
  assert.equal(text.formatDuration(null), null);
  assert.deepEqual(text.outputLatency({ outputLatency: 0.012, baseLatency: 0.003 }), { outputMs: 12, baseMs: 3 });
  assert.deepEqual(text.outputLatency({}), { outputMs: null, baseMs: null });
});

test("comparison events parse across chunk boundaries", async () => {
  const lines = '{"type":"accepted"}\n{"type":"row","row":{"status":"complete"}}\n{"type":"done"}';
  const chunks = [lines.slice(0, 10), lines.slice(10, 31), lines.slice(31)].map((chunk) => new TextEncoder().encode(chunk));
  const body = new ReadableStream({ start(controller) { for (const chunk of chunks) controller.enqueue(chunk); controller.close(); } });
  const seen: string[] = [];
  await text.readEvents({ body }, (event: { type: string }) => seen.push(event.type));
  assert.deepEqual(seen, ["accepted", "row", "done"]);
});

/** One fake page context: a socket, a playback worklet port and an output clock. */
function readbackHarness() {
  const sockets: FakeSocket[] = [];
  const ports: Array<{ messages: any[]; onmessage?: (event: { data: any }) => void; postMessage(data: any): void }> = [];
  class FakeSocket {
    static OPEN = 1;
    readyState = 0;
    sent: any[] = [];
    listeners = new Map<string, Array<(event: any) => void>>();
    constructor() { sockets.push(this); queueMicrotask(() => { this.readyState = 1; this.fire("open", {}); }); }
    addEventListener(type: string, listener: (event: any) => void) { this.listeners.set(type, [...this.listeners.get(type) ?? [], listener]); }
    fire(type: string, event: any) { for (const listener of this.listeners.get(type) ?? []) listener(event); }
    send(raw: string) { this.sent.push(JSON.parse(raw)); }
    event(data: unknown) { this.fire("message", { data: data instanceof ArrayBuffer ? data : JSON.stringify(data) }); }
  }
  const globals = globalThis as any;
  globals.WebSocket = FakeSocket;
  globals.location = { protocol: "http:", host: "127.0.0.1:4870" };
  globals.AudioWorkletNode = class {
    port = { messages: [] as any[], onmessage: undefined, postMessage(data: any) { this.messages.push(data); } };
    constructor() { ports.push(this.port); }
    connect() {}
  };
  const context = { destination: {}, audioWorklet: { async addModule() {} },
    getOutputTimestamp: () => ({ contextTime: 10, performanceTime: performance.now() }) };
  const readback = new VoiceReadback({ context });
  return { readback, sockets, ports };
}

function packet(playbackId: number, samples: number[]) {
  const bytes = new ArrayBuffer(9 + samples.length * 4);
  const view = new DataView(bytes);
  view.setUint8(0, 1);
  view.setUint32(1, playbackId, true);
  view.setUint32(5, 22_050, true);
  samples.forEach((sample, index) => view.setFloat32(9 + index * 4, sample, true));
  return bytes;
}

const selection = { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 };

test("readback selects, speaks, retries a busy host, plays, acknowledges drain and keeps the PCM", async () => {
  const { readback, sockets, ports } = readbackHarness();
  const done = readback.speak({ selection, catalogueRevision: "rev", text: "Clear the screen." });
  while (!sockets[0]?.sent.length) await new Promise((resolve) => setTimeout(resolve, 1));
  const socket = sockets[0]!;
  const select = socket.sent[0];
  assert.equal(select.type, "select-output");
  socket.event({ type: "output-selected", requestId: select.requestId, catalogueRevision: "rev", selectionRevision: 1, selection });
  assert.equal(socket.sent[1].type, "speak-sample");
  assert.equal(socket.sent[1].text, "Clear the screen.");
  socket.event({ type: "sample-error", requestId: select.requestId, code: "BUSY", message: "busy" });
  await new Promise((resolve) => setTimeout(resolve, 200));
  const retry = socket.sent[2];
  assert.equal(retry.type, "select-output");
  assert.notEqual(retry.requestId, select.requestId, "each try uses a new request id");
  socket.event({ type: "output-selected", requestId: retry.requestId, catalogueRevision: "rev", selectionRevision: 2, selection });
  socket.event({ type: "sample-accepted", requestId: retry.requestId, playbackId: 4 });
  socket.event({ type: "playback", requestId: retry.requestId, action: "start", playbackId: 4, sampleRate: 22_050 });
  socket.event(packet(4, [0.5, -0.5]));
  socket.event({ type: "playback", requestId: retry.requestId, action: "end", playbackId: 4 });
  socket.event({ type: "sample-complete", requestId: retry.requestId, playbackId: 4, outcome: "produced",
    timing: { serverFirstAudioReadyMs: { status: "measured", ms: 80 } } });
  const port = ports[0]!;
  assert.deepEqual(port.messages.map((message) => message.kind), ["start", "audio", "end"]);
  port.onmessage!({ data: { kind: "first-render", playbackId: 4, audioTime: 10.05 } });
  assert.equal(readback.playing, true);
  port.onmessage!({ data: { kind: "drained", playbackId: 4, audioEndMs: 1, audioTime: 10.5 } });
  const result = await done;
  assert.equal(result.outcome, "finished");
  assert.deepEqual([...result.samples], [0.5, -0.5]);
  assert.equal(result.sampleRate, 22_050);
  assert.equal(result.hostFirstAudioMs, 80);
  assert.ok(result.totalMs! > result.firstAudioMs!, "total ends after first audio");
  assert.deepEqual(socket.sent.at(-1), { type: "playback-drained", playbackId: 4, audioEndMs: 1 });
});

test("stop clears playback, cancels the request and times press to silence on the output clock", async () => {
  const { readback, sockets, ports } = readbackHarness();
  const done = readback.speak({ selection, catalogueRevision: "rev", text: "Long reading." });
  while (!sockets[0]?.sent.length) await new Promise((resolve) => setTimeout(resolve, 1));
  const socket = sockets[0]!;
  const id = socket.sent[0].requestId;
  socket.event({ type: "output-selected", requestId: id, catalogueRevision: "rev", selectionRevision: 1, selection });
  socket.event({ type: "sample-accepted", requestId: id, playbackId: 9 });
  socket.event({ type: "playback", requestId: id, action: "start", playbackId: 9, sampleRate: 22_050 });
  const pressedAt = performance.now();
  readback.stop(pressedAt);
  assert.equal(ports[0]!.messages.at(-1).kind, "clear");
  assert.deepEqual(socket.sent.at(-1), { type: "cancel-sample", requestId: id });
  // Silence starts 30 ms of context time after the stamped context time 10.
  ports[0]!.onmessage!({ data: { kind: "drained", playbackId: 9, audioEndMs: 0, audioTime: 10.03 } });
  const result = await done;
  assert.equal(result.outcome, "stopped");
  assert.ok(result.stopMs! >= 30 && result.stopMs! < 80, String(result.stopMs));
});

/**
 * The page with real gate, detector and text helpers; the mic and voice are
 * scripted fakes. With holdVoice the voice plays until stopped, then reports
 * a 41 ms stop; clearing the readback's hold lets later readings finish. Each mic take is one sample longer than the last, so replays
 * show which take they play. With transcript "partials" the setup streams: live records what reaches the live words channel.
 */
function pageHarness({ holdVoice = false, transcript = "final", models = [] as unknown[] } = {}, rows: unknown[] = [
  { type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "processing", text: null, timings: {} } },
  { type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "complete", text: "Clear the screen.", timings: { firstUse: false, processingMs: 90 } } },
]) {
  // The real gate schedules its release tail on the window timers.
  (globalThis as any).window ??= { setTimeout, clearTimeout };
  const elements = new Map<string, any>();
  const element = () => ({ textContent: "", value: "", disabled: false, hidden: false, checked: true, className: "", style: {} as Record<string, string>,
    dataset: {} as Record<string, string>,
    children: [] as any[], listeners: new Map<string, (event?: any) => void>(), classList: { toggle() {} },
    addEventListener(type: string, listener: (event?: any) => void) { this.listeners.set(type, listener); },
    setAttribute() {}, append(...nodes: any[]) { this.children.push(...nodes); }, replaceChildren(...nodes: any[]) { this.children = nodes; } });
  const select = (): any => Object.assign(element(), { options: [] as any[],
    replaceChildren(this: any, ...options: any[]) { this.options = options; this.value = options[0]?.value ?? ""; } });
  const document = { listeners: new Map<string, (event: any) => void>(), getElementById(id: string) {
    if (!elements.has(id)) elements.set(id, id.endsWith("-model") || id.endsWith("-voice") ? select() : element());
    return elements.get(id);
  }, createElement: (tag: string) => Object.assign(tag === "select" ? select() : element(), { tag }),
  addEventListener(type: string, listener: (event: any) => void) { this.listeners.set(type, listener); } };
  const spoken: any[] = [];
  const posted: string[] = [];
  const played: number[] = [];
  let takes = 0;
  const live: { begun: [string, number][]; settings: unknown[]; pushed: number; ended: number } = { begun: [], settings: [], pushed: 0, ended: 0 };
  class LiveWords { takeId: string | null = null; serial = 0;
    begin(setupId: string, preRoll: Float32Array, settings: unknown) {
      live.begun.push([setupId, preRoll.length]);
      live.settings.push(settings);
      return this.takeId = `live-${++this.serial}`;
    }
    push() { if (this.takeId) live.pushed++; }
    end() { if (this.takeId) live.ended++; this.takeId = null; } }
  const microphones: any[] = [];
  class Mic { open = true; recording = false; rate: number;
    constructor({ rate = 16_000 } = {}) { this.rate = rate; microphones.push(this); }
    async start() { this.open = true; } stop() { this.open = this.recording = false; }
    beginTake() { this.recording = true; return new Float32Array(4_800); }
    async endTake() { this.recording = false; return { samples: new Float32Array(8_000 + takes++).fill(0.1), preRollSamples: 0 }; } }
  let voiceLoads = [4_900];
  const voice = () => ({ firstAudioMs: 120, samples: new Float32Array(4), sampleRate: 22_050, loadMs: voiceLoads.shift() ?? null });
  class Readback { active = false; playing = false; hold = holdVoice; stopped?: (result: unknown) => void; async prepare() {}
    speak(request: any) {
      spoken.push(request);
      // The host echoes the settings it applied; a reading without settings gets none back here.
      if (!this.hold) return Promise.resolve({ ...voice(), outcome: "finished", totalMs: 900, settings: request.settings ?? null });
      this.active = this.playing = true;
      return new Promise((resolve) => { this.stopped = resolve; });
    }
    stop() {
      if (!this.active) return;
      this.active = this.playing = false;
      this.stopped!({ ...voice(), outcome: "stopped", totalMs: null, stopMs: 41 });
    } }
  // The setups route's live and final fields, from the real registries; Hex streams too, for a pairing.
  const streaming = transcript === "partials";
  const setups = { setups: [
    { id: "mlx-parakeet/v3-greedy", label: "Parakeet v3 MLX", card: { runtime: "mlx", transcript }, readiness: { ready: true, sizeBytes: 1 },
      live: streaming ? liveStream("mlx-parakeet/v3-greedy") : null, liveDefault: streaming ? "mlx-parakeet/v3-greedy" : null },
    { id: "hex/unified", label: "Hex Parakeet Unified", card: { runtime: "transcribe.cpp", transcript: "partials" },
      readiness: { ready: true, sizeBytes: 1 }, live: liveStream("hex/unified"), liveDefault: "hex/unified" },
    { id: "cohere/transcribe", label: "Cohere Transcribe", card: { runtime: "transcribe.cpp", transcript: "final" },
      readiness: { ready: true, sizeBytes: 1 }, live: null, liveDefault: null, final: finalSetup("cohere/transcribe") },
  ] };
  const voices = { revision: "rev", default: { model: "piper-kristin-medium" }, models: [{ id: "piper-kristin-medium", label: "Piper Kristin medium",
    provider: { kind: "local" }, defaultVoice: "kristin", defaultSpeed: 1, voices: [{ id: "kristin", label: "Kristin" }], readiness: { state: "ready" } },
    ...models] };
  // Like the browser's fetch, a call with any receiver but the window throws.
  let realm: unknown;
  async function fetch(this: unknown, url: string, init?: { method?: string }) {
    if (this !== undefined && this !== realm) throw new TypeError("Illegal invocation");
    posted.push(url);
    if (url.startsWith("/speech-output/picks/")) return { ok: true, arrayBuffer: async () => new ArrayBuffer(8) };
    if (url.startsWith("/speech-output/references")) return { ok: true, json: async () => (init?.method === "POST" ? { id: "d".repeat(64) } : {}) };
    if (url === "/prompt-reader/setups") return { ok: true, json: async () => setups };
    if (url === "/speech-output/models") return { ok: true, json: async () => voices };
    const body = new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(rows.map((row) => JSON.stringify(row)).join("\n")));
      controller.close();
    } });
    return { ok: true, status: 200, body };
  }
  const context = vm.createContext({ document, fetch, performance, Math, Number, String, Object, setInterval, clearInterval, setTimeout, clearTimeout, URLSearchParams, TextEncoder, TextDecoder,
    JSON, Float32Array, Promise, Option: function(label: string, value: string) { return { label, value }; },
    HoldReleaseGate, CaptureSession: Mic, VoiceReadback: Readback, LiveWords, TalkOverDetector, ...text, ...trail,
    AudioContext: class { currentTime = 0; sampleRate = 48_000; destination = {}; async resume() {}
      createBuffer(_channels: number, length: number) { return { length, copyToChannel() {} }; }
      async decodeAudioData() { return { sampleRate: 24_000, getChannelData: () => new Float32Array(2_400) }; }
      createBufferSource() { const source = { buffer: { length: 0 }, connect() {}, stop() {}, start() { played.push(source.buffer.length); } }; return source; } },
    window: { setTimeout, clearTimeout } });
  realm = vm.runInContext("globalThis", context);
  vm.runInContext(["setting-controls.js", "voice-settings.js", "voice-copy.js", "take-history.js", "road-test.js"].map(source).join("\n")
    + "\nglobalThis.Page = RoadTestPage;", context);
  return { page: new context.Page(), elements, spoken, posted, played, live, document, microphones };
}

const until = async (done: () => unknown) => {
  for (let tries = 0; !done(); tries++) {
    if (tries > 200) throw new Error("timed out");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
};
const listed = (list: any) => list.children.map((row: any) => row.children.map((cell: any) => cell.textContent).join(": "));

/** Holds and releases one take, then waits for the voice to start or finish. */
async function take(page: any, pressAt: number) {
  page.press(pressAt);
  await new Promise((resolve) => setTimeout(resolve, 0));
  page.release(pressAt + 400);
  await until(() => page.phase === "speaking" || page.run?.output);
}

test("the page loop: hold, release, final text, then the chosen voice reads it back", async () => {
  const { page, elements, spoken, posted } = pageHarness();
  await page.start();
  assert.equal(elements.get("input-card").textContent, "Local · 1 MB · mlx");
  page.press(100);
  await new Promise((resolve) => setTimeout(resolve, 0));
  page.release(500);
  for (let tries = 0; !page.run?.output && tries < 100; tries++) await new Promise((resolve) => setTimeout(resolve, 10));
  assert.ok(posted.some((url) => url.startsWith("/prompt-reader/comparisons?")));
  assert.equal(elements.get("transcript").textContent, "Clear the screen.");
  assert.equal(spoken.length, 1);
  assert.equal(spoken[0].text, "Clear the screen.");
  assert.equal(JSON.stringify(spoken[0].selection), JSON.stringify({ schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 }));
  assert.equal(elements.get("take-models").textContent, "Parakeet v3 MLX · Piper Kristin medium");
  elements.get("output-model").value = "another-voice";
  page.render();
  assert.equal(elements.get("take-models").textContent, "Parakeet v3 MLX · Piper Kristin medium",
    "the latest take keeps naming the models that produced it after the selects change");
  const timings = elements.get("output-timings").children.map((row: any) => row.children.map((cell: any) => cell.textContent).join(": "));
  assert.ok(timings.includes("Text to first audio: 120 ms"), timings.join(" | "));
  assert.ok(timings.includes("Text to end: 900 ms"));
  assert.equal(page.phase, "idle");
});

test("model start sums every load phase, so a cleanup row counts its own model after Ultra", async () => {
  const row = (status: string, extra = {}) => ({ type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status, text: null, timings: {}, ...extra } });
  const { page } = pageHarness({}, [row("loading"), row("processing"), row("loading"), row("processing"),
    row("complete", { text: "Clear the screen.", timings: { firstUse: true, processingMs: 90 } })]);
  await page.start();
  let now = 0;
  const clock = performance.now;
  const times = [0, 100, 300, 1_300, 1_400, 1_500];
  performance.now = () => times[Math.min(now++, times.length - 1)]!;
  try {
    const result = await page.transcribe({ setup: page.setups[0] }, new Float32Array(16), 0, () => {});
    assert.equal(result.loadMs, 100 + 1_000);
  } finally { performance.now = clock; }
});

test("Space over the voice stops it like Esc, and the new take keeps the stop time through its own readback", async () => {
  const { page, elements } = pageHarness({ holdVoice: true });
  await page.start();
  await take(page, 100);
  const first = page.run;
  page.readback.hold = false;
  page.press(3_000);
  assert.notEqual(page.run, first, "the press starts a new take");
  assert.equal(first.interrupt.kind, "stop");
  await until(() => page.history.includes(first));
  assert.ok(listed(elements.get("interrupt-timings")).includes("Stop to silence: 41 ms"));
  const second = page.run;
  await new Promise((resolve) => setTimeout(resolve, 0));
  page.release(3_400);
  await until(() => second.output?.outcome === "finished" && page.phase === "idle");
  assert.ok(listed(elements.get("interrupt-timings")).includes("Stop to silence: 41 ms"),
    "a readback that finished leaves the carried stop time alone");
});

test("after a talk over both recordings replay, each under its own button", async () => {
  const { page, elements, played } = pageHarness({ holdVoice: true });
  await page.start();
  await take(page, 100);
  let time = 1;
  const frame = (level: number) => page.frame({ level, audioTime: time += 0.02 });
  for (let index = 0; index < 10; index++) frame(0.002);
  for (let index = 0; index < 3; index++) frame(0.2);
  assert.equal(page.phase, "cut-in");
  while (page.phase === "cut-in") frame(0.002);
  await until(() => page.phase === "idle");
  assert.equal(page.run.interrupt.text, "Clear the screen.");
  assert.equal(elements.get("replay-take").disabled, false);
  assert.equal(elements.get("replay-talk-over").hidden, false);
  page.replay("take");
  page.replay("talk-over");
  assert.deepEqual(played, [8_000, 8_001], "the main take, then the talk over");
});

test("earlier takes label every cell and name a cold row's model start", async () => {
  const row = (status: string, extra = {}) => ({ type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status, text: null, timings: {}, ...extra } });
  const { page, elements } = pageHarness({}, [row("loading"), row("processing"),
    row("complete", { text: "Clear the screen.", timings: { firstUse: true, processingMs: 90 } })]);
  await page.start();
  assert.deepEqual(elements.get("history-head").children.map((cell: any) => cell.textContent),
    ["Speech in", "Voice", "Transcript", "Press to first words", "Release to text", "Text to first audio", "Release to voice"]);
  await take(page, 100);
  await take(page, 2_000);
  const [cold] = elements.get("history").children;
  assert.deepEqual(cold.children.map((cell: any) => cell.dataset.label),
    ["Speech in", "Voice", "Transcript", "Press to first words", "Release to text", "Text to first audio", "Release to voice"]);
  assert.equal(cold.children[3].children[0].textContent, "–", "a final text only setup shows no first words");
  const releaseToText = cold.children[4].children.map((part: any) => part.textContent);
  assert.equal(releaseToText.length, 2);
  assert.match(releaseToText[1], /^Model start \d/);
});

test("a streaming setup shows live words while held, times them from the press, and the final replaces them", async () => {
  const { page, elements, live } = pageHarness({ transcript: "partials" });
  await page.start();
  assert.equal(elements.get("input-card").textContent, "Local · 1 MB · mlx");
  const clock = performance.now;
  let now = 1_000;
  performance.now = () => now;
  try {
    page.press(1_000);
    await until(() => live.begun.length === 1);
    assert.deepEqual(live.begun, [["mlx-parakeet/v3-greedy", 4_800]], "the take begins with its pre roll");
    page.frame({ level: 0.1, audioTime: 1, samples: new Float32Array(320) });
    assert.equal(live.pushed, 1);
    const takeId = page.run.live.takeId;
    now = 1_100;
    page.liveEvent({ type: "loading", takeId });
    assert.equal(elements.get("status").textContent, "Listening. Loading Parakeet v3 MLX for live words.");
    now = 1_900;
    page.liveEvent({ type: "ready", takeId, cold: true });
    assert.equal(elements.get("status").textContent, "Listening. Release when you finish.");
    page.liveEvent({ type: "partial", takeId: "live-old", stable: "stale", tentative: "", processingMs: 1 });
    assert.equal(elements.get("transcript").textContent, "", "words from another take never show");
    now = 2_250;
    page.liveEvent({ type: "partial", takeId, stable: " Clear the", tentative: " scream", processingMs: 40 });
    assert.equal(elements.get("transcript").textContent, "Clear the scream");
    now = 2_600;
    page.release(2_600);
    assert.equal(live.ended, 1, "release stops the live words before the release tail");
    await until(() => page.run.output);
  } finally { performance.now = clock; }
  assert.equal(elements.get("transcript").textContent, "Clear the screen.", "the final replaces the live words in place");
  const timings = listed(elements.get("input-timings"));
  assert.ok(timings.includes("Press to first words: 1.25 s"), timings.join(" | "));
  assert.ok(timings.includes("Model start: 800 ms"), "the live words' load counts as model start");
  page.liveEvent({ type: "partial", takeId: page.run.live.takeId, stable: "late", tentative: "", processingMs: 1 });
  assert.equal(page.run.live.words, "Clear the scream", "words after the final are ignored");
  page.press(5_000);
  await until(() => page.run.live);
  page.liveEvent({ type: "loading", takeId: page.run.live.takeId });
  page.liveEvent({ type: "ready", takeId: page.run.live.takeId, cold: false });
  assert.equal(page.run.live.loadMs, undefined, "a warm model's load call is no model start");
});

test("a talk over with a streaming setup streams too, and shows its words as you say them", async () => {
  const { page, elements, live } = pageHarness({ holdVoice: true, transcript: "partials" });
  await page.start();
  await take(page, 100);
  let time = 1;
  const frame = (level: number) => page.frame({ level, audioTime: time += 0.02, samples: new Float32Array(320) });
  for (let index = 0; index < 10; index++) frame(0.002);
  for (let index = 0; index < 3; index++) frame(0.2);
  assert.equal(page.phase, "cut-in");
  assert.equal(live.begun.length, 2, "the cut in begins its own live take");
  page.liveEvent({ type: "partial", takeId: page.run.interrupt.live.takeId, stable: "Stop", tentative: " there", processingMs: 30 });
  assert.ok(listed(elements.get("interrupt-timings")).includes("You said: Stop there"));
  assert.equal(elements.get("interrupt-timings").children.at(-1).className, "wide partial");
  while (page.phase === "cut-in") frame(0.002);
  await until(() => page.phase === "idle");
  assert.equal(live.ended, 2);
  assert.equal(page.run.interrupt.text, "Clear the screen.");
  assert.equal(elements.get("interrupt-timings").children.at(-1).className, "wide");
});

test("talk over starts off on every load, and the note says once where it works", () => {
  const html = source("road-test.html");
  const input = html.match(/<input[^>]*id="talk-over"[^>]*>/)![0];
  assert.doesNotMatch(input, /\schecked/, "the switch starts off");
  assert.match(input, /autocomplete="off"/, "a reload never restores a ticked switch");
  assert.equal(html.match(/headphones/gi)?.length, 1);
  assert.match(html, /Talking over works best with headphones; on speakers, other sound can stop the voice\./);
});

test("with talk over off, loud sound during the voice records nothing and the interrupt column shows only the stop", async () => {
  const { page, elements, live } = pageHarness({ holdVoice: true, transcript: "partials" });
  elements.set("talk-over", Object.assign(page.elements["talk-over"], { checked: false }));
  await page.start();
  assert.deepEqual(listed(elements.get("interrupt-timings")), ["Stop to silence: –"]);
  await take(page, 100);
  let time = 1;
  for (let index = 0; index < 40; index++) page.frame({ level: index > 10 ? 0.3 : 0.002, audioTime: time += 0.02, samples: new Float32Array(320) });
  assert.equal(page.phase, "speaking", "no cut in");
  assert.equal(page.mic.recording, false, "no cut in recording");
  assert.equal(live.begun.length, 1, "only the main take streamed");
  assert.equal(page.run.interrupt, undefined, "no talk over detection");
  page.stopVoice(5_000);
  await until(() => page.phase === "idle");
  assert.deepEqual(listed(elements.get("interrupt-timings")), ["Stop to silence: 41 ms"]);
  assert.equal(elements.get("replay-talk-over").hidden, true);
  page.elements["talk-over"].checked = true;
  page.render();
  assert.deepEqual(listed(elements.get("interrupt-timings")), ["Stop to silence: 41 ms", "Voice heard after: –", "Silent after: –"]);
});

test("a take with no words shows no first words, and a voice's first use names its model start", async () => {
  const silent = [
    { type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "complete", text: "", timings: { firstUse: false, processingMs: 40 } } },
  ];
  const { page, elements } = pageHarness({ transcript: "partials" }, silent);
  await page.start();
  const quiet = async (at: number) => {
    page.press(at);
    await new Promise((resolve) => setTimeout(resolve, 0));
    page.release(at + 4_000);
    await until(() => page.run.input && page.phase === "idle");
  };
  await quiet(100);
  assert.equal(elements.get("status").textContent, "Nothing heard. Try again closer to the microphone.");
  assert.ok(listed(elements.get("input-timings")).includes("Press to first words: –"));
  await quiet(9_000);
  const [empty] = elements.get("history").children;
  assert.equal(empty.children[2].children[0].textContent, "Nothing heard");
  assert.equal(empty.children[3].children[0].textContent, "–", "Earlier takes carry no first words either");

  const spoken = pageHarness();
  await spoken.page.start();
  await take(spoken.page, 100);
  assert.ok(listed(spoken.elements.get("output-timings")).includes("Model start: 4.90 s"), "the voice's first use");
  await take(spoken.page, 2_000);
  assert.ok(!listed(spoken.elements.get("output-timings")).some((row: string) => row.startsWith("Model start")), "a warm voice");
  const [cold] = spoken.elements.get("history").children;
  assert.deepEqual(cold.children[5].children.map((part: any) => part.textContent), ["120 ms", "Model start 4.90 s"]);
});

test("the trail marks inserted, replaced and removed words, and counts only changes to words already shown", () => {
  const kinds = (before: string, after: string) => trail.diffWords(before, after).map((segment) =>
    `${segment.kind}:${segment.was.join(" ")}>${segment.words.join(" ")}`);
  assert.deepEqual(kinds("", "Clear the"), ["inserted:>Clear the"]);
  assert.deepEqual(kinds("Clear the scream", "Clear the screen."), ["same:>Clear the", "replaced:scream>screen."]);
  assert.deepEqual(kinds("One, two", "One, two, three."), ["same:>One, two,", "inserted:>three."],
    "case and punctuation alone are no change; the newer form shows");
  assert.deepEqual(kinds("Spawn ten codecs agents", "Spawn ten Codex agents"), ["same:>Spawn ten", "replaced:codecs>Codex", "same:>agents"]);
  assert.deepEqual(kinds("Open the the browser", "Open the browser"), ["same:>Open the", "removed:the>", "same:>browser"]);
  assert.equal(trail.isCorrection(trail.diffWords("Clear", "Clear the")), false, "words added at the end are no correction");
  assert.deepEqual(kinds("seven, eight, n", "seven, eight, nine, ten"), ["same:>seven, eight,", "inserted:>nine, ten"],
    "a word still being written grows; it is no correction");
  assert.equal(trail.isCorrection(trail.diffWords("Clear screen", "Clear the screen")), true, "words put back in the middle are");
  const updates = [{ atMs: 900, words: "Clear" }, { atMs: 1_700, words: "Clear the scream" }, { atMs: 2_500, words: "Clear the screen" }];
  assert.equal(trail.trailSummary(updates), "3 updates, 1 correction");
  assert.equal(trail.trailSummary(updates.slice(0, 1)), "1 update, no corrections");
  const element = (tag: string) => ({ tag, textContent: "", children: [] as any[], className: "", append(...nodes: any[]) { this.children.push(...nodes); } });
  const items = trail.trailItems({ createElement: element }, updates, "Clear the screen.", text.formatDuration);
  const flat = (node: any): string => node.children.length ? node.children.map(flat).join("")
    : node.tag === "ins" ? `[+${node.textContent}]` : node.tag === "del" ? `[-${node.textContent}]` : node.textContent;
  assert.deepEqual(items.map((item: any) => `${item.children[0].textContent} ${flat(item.children[1]).trim()}`), [
    "+900 ms [+Clear]", "+1.70 s Clear [+the scream]", "+2.50 s Clear the [-scream] [+screen]", "Final Clear the screen."]);
});

test("the live words line names the live model, the settings the host applied, and a final model that gives final text only", () => {
  const ultra = { id: "mlx-parakeet/ultra-greedy", label: "Parakeet Ultra MLX" };
  const stream = liveStream(ultra.id)!;
  assert.equal(text.liveLine({ setup: ultra, stream, settings: { updateMs: 800 } }, ultra),
    "Live words: Parakeet Ultra MLX stream, updates every 800 ms, the last 5.12 s may change.");
  assert.equal(text.liveLine({ setup: ultra, stream, settings: { updateMs: 1_600 } },
    { id: "mlx-cleanup/qwen-targeted", label: "Ultra plus Qwen cleanup", baseSetup: ultra.id }),
  "Live words: Parakeet Ultra MLX stream, updates every 1.6 s, the last 5.12 s may change. Ultra plus Qwen cleanup writes the final text.");
  assert.equal(text.liveLine({ setup: { id: "hex/unified", label: "Hex Parakeet Unified" }, stream: liveStream("hex/unified")!,
    settings: { lookaheadMs: 2_080 } }), "Live words: Hex Parakeet Unified stream, 2.08 s lookahead, written words never change.");
  assert.equal(text.liveLine(null), "Live words off.");
  const qwen = { id: "qwen/context", label: "Qwen3 ASR with known words", card: { transcript: "final" } };
  assert.equal(text.liveLine(null, qwen), "Live words off. Qwen3 ASR with known words gives final\u00a0text\u00a0only.");
  assert.equal(text.liveLine(null, { ...ultra, card: { transcript: "partials" } }), "Live words off.");
  assert.equal(text.liveLine({ setup: ultra, stream, settings: { updateMs: 800 } }, qwen),
    "Live words: Parakeet Ultra MLX stream, updates every 800 ms, the last 5.12 s may change. Qwen3 ASR with known words writes the final\u00a0text\u00a0only.",
    "final text only never wraps inside itself");
  assert.equal(text.liveSettingsNote({ stream, settings: { updateMs: 1_600 } }), "Updates every 1.6 s",
    "Earlier takes name the setting by its label");
});

test("live words default to the setup's own stream; another live model and its settings reach the channel, and each figure names its model", async () => {
  const { page, elements, live } = pageHarness({ transcript: "partials" });
  await page.start();
  assert.equal(elements.get("live-model").value, "mlx-parakeet/v3-greedy", "today's pairing");
  assert.deepEqual(elements.get("live-model").options.map((option: any) => option.label), ["Off", "Parakeet v3 MLX", "Hex Parakeet Unified"]);
  assert.deepEqual(elements.get("live-settings").children.map((column: any) => column.children[0].textContent), ["Updates every"]);
  assert.equal(elements.get("live-line").textContent, "Live words: Parakeet v3 MLX stream, updates every 800 ms, the last 5.12 s may change.");
  elements.get("live-model").value = "hex/unified";
  elements.get("live-model").listeners.get("change")!();
  const [lookahead] = elements.get("live-settings").children;
  assert.equal(lookahead.children[0].textContent, "Lookahead");
  lookahead.children[1].value = "480";
  const clock = performance.now;
  let now = 1_000;
  performance.now = () => now;
  try {
    page.press(1_000);
    await until(() => live.begun.length === 1);
    assert.deepEqual(live.begun[0], ["hex/unified", 4_800]);
    assert.equal(JSON.stringify(live.settings[0]), JSON.stringify({ lookaheadMs: 480 }));
    const takeId = page.run.live.takeId;
    page.liveEvent({ type: "stream", takeId, setupId: "hex/unified", settings: { lookaheadMs: 480 } });
    now = 1_900;
    page.liveEvent({ type: "partial", takeId, stable: " Clear the", tentative: "", processingMs: 5 });
    now = 2_300;
    page.liveEvent({ type: "partial", takeId, stable: " Clear the scream", tentative: "", processingMs: 5 });
    assert.equal(elements.get("live-line").textContent,
      "Live words: Hex Parakeet Unified stream, 480 ms lookahead, written words never change. Parakeet v3 MLX writes the final text.");
    assert.equal(elements.get("trail").hidden, false);
    assert.equal(elements.get("trail-summary").textContent, "2 updates, no corrections");
    page.release(2_600);
    await until(() => page.run.output);
  } finally { performance.now = clock; }
  const named = elements.get("input-timings").children.map((row: any) =>
    `${row.children[0].textContent}${row.children[0].children.map((note: any) => ` (${note.textContent})`).join("")}`);
  assert.deepEqual(named.slice(0, 2), ["Press to first words (Hex Parakeet Unified)", "Release to text (Parakeet v3 MLX)"]);
  assert.equal(elements.get("trail-list").children.length, 3, "two updates, then the final");
  await take(page, 5_000);
  const row = elements.get("history").children[0];
  assert.equal(row.children[0].children.map((cell: any) => cell.textContent).join(" | "), "Parakeet v3 MLX | Live words Hex Parakeet Unified · Lookahead 480 ms",
    "an earlier take names its live model and the setting the host applied");
  assert.equal(row.children[2].children.at(-1).children[0].textContent, "2 updates, no corrections", "earlier takes keep the trail");
  elements.get("input-model").listeners.get("change")!();
  assert.equal(elements.get("live-model").value, "mlx-parakeet/v3-greedy", "a new speech in choice restores its own pairing");
});

test("after a take, the live words line follows a new speech in or live words choice, and returns to the take", async () => {
  const { page, elements } = pageHarness({ transcript: "partials" });
  await page.start();
  await take(page, 1_000);
  const taken = "Live words: Parakeet v3 MLX stream, updates every 800 ms, the last 5.12 s may change.";
  assert.equal(elements.get("live-line").textContent, taken);
  elements.get("live-model").value = "off";
  elements.get("live-model").listeners.get("change")!();
  assert.equal(elements.get("live-line").textContent, "Live words off.", "a new live words choice");
  elements.get("live-model").value = "mlx-parakeet/v3-greedy";
  elements.get("live-model").listeners.get("change")!();
  assert.equal(elements.get("live-line").textContent, taken, "the take's own choice again");
  elements.get("input-model").value = "hex/unified";
  elements.get("input-model").listeners.get("change")!();
  assert.equal(elements.get("live-line").textContent,
    "Live words: Hex Parakeet Unified stream, 2.08 s lookahead, written words never change.", "a new speech in choice");
});

test("with live words off nothing streams, no first words figure shows, and the line says so once", async () => {
  const { page, elements, live } = pageHarness({ transcript: "partials" });
  await page.start();
  elements.get("live-model").value = "off";
  elements.get("live-model").listeners.get("change")!();
  assert.equal(elements.get("live-settings").children.length, 0);
  await take(page, 100);
  assert.equal(live.begun.length, 0);
  assert.equal(elements.get("live-line").textContent, "Live words off.");
  assert.equal(elements.get("trail").hidden, true);
  assert.ok(!listed(elements.get("input-timings")).some((row: string) => row.startsWith("Press to first words")));
});

test("Cohere shows its Names setting under Speech in; each take sends it, and the take and earlier takes name the value used", async () => {
  const row = (status: string, extra = {}) => ({ type: "row", row: { setupId: "cohere/transcribe", status, text: null, timings: {}, ...extra } });
  const { page, elements, posted } = pageHarness({}, [row("processing"),
    row("complete", { text: "Close Grok.", timings: { firstUse: false, processingMs: 90 }, settings: { namesBias: 1.25 } })]);
  await page.start();
  assert.equal(elements.get("final-settings").hidden, true, "Parakeet has no final settings");
  elements.get("input-model").value = "cohere/transcribe";
  elements.get("input-model").listeners.get("change")!();
  assert.equal(elements.get("final-settings").hidden, false);
  const [names, ...others] = elements.get("final-settings").children;
  assert.equal(others.length, 0, "one setting");
  const [label, select, note] = names.children;
  assert.equal(label.textContent, "Names");
  assert.equal(note.textContent, "Biases Cohere toward Grok, Claude, Codex, GPT, TM, Hibiki; can turn similar words into these names");
  assert.deepEqual(select.options.map((option: any) => option.label), ["Off", "Gentle", "Strong"]);
  assert.equal(select.value, "0", "Off by default");
  select.value = "1.25";
  await take(page, 1_000);
  const url = posted.find((entry) => entry.startsWith("/prompt-reader/comparisons?"))!;
  assert.equal(new URLSearchParams(url.split("?")[1]).get("settings"), JSON.stringify({ "cohere/transcribe": { namesBias: 1.25 } }));
  assert.equal(elements.get("take-models").textContent, "Cohere Transcribe · Names gentle · Piper Kristin medium",
    "the take names the value the host applied");
  await take(page, 5_000);
  const earlier = elements.get("history").children[0];
  assert.equal(earlier.children[0].children.map((cell: any) => cell.textContent).join(" | "), "Cohere Transcribe | Names gentle");
  elements.get("input-model").value = "mlx-parakeet/v3-greedy";
  elements.get("input-model").listeners.get("change")!();
  assert.equal(elements.get("final-settings").hidden, true);
  assert.equal(page.finalChoice(), null, "a setup without final settings sends none");
});

const qwen = (id: string, settings: unknown, voices: readonly unknown[] = QWEN_SPEAKERS) => ({ id, label: id, provider: { kind: "local" },
  defaultVoice: (voices[0] as { id: string }).id, defaultSpeed: 1, voices, settings, readiness: { state: "ready" } });
const QWEN_MODELS = [qwen("qwen3-tts-1.7b", QWEN_CUSTOM_17B_SETTINGS), qwen("qwen3-tts-0.6b", QWEN_CUSTOM_SETTINGS),
  qwen("qwen3-tts-0.6b-base", QWEN_COPY_SETTINGS, QWEN_COPY_VOICES)];

/** Every node under root, depth first. */
function nodes(root: any): any[] {
  return [root, ...(root.children ?? []).flatMap(nodes)];
}
/** The control in the setting column labelled `label`: its select or input, or for Seed the mode and number pair. */
function control(root: any, label: string): any {
  const column = nodes(root).find((node) => node.className === "setting" && node.children[0]?.textContent === label);
  return column?.children[1];
}
const choose = (select: any, label: string) => {
  const index = select.options.findIndex((option: any) => option.label === label);
  assert.ok(index >= 0, `${label} among ${select.options.map((option: any) => option.label)}`);
  select.value = select.options[index].value;
  select.listeners.get("change")?.();
};
async function pickModel(harness: ReturnType<typeof pageHarness>, id: string) {
  const output = harness.elements.get("output-model");
  output.value = id;
  output.listeners.get("change")();
}

test("Space types in voice settings and the reference transcript; on any other control it still records", async () => {
  const { page, document } = pageHarness({ models: QWEN_MODELS });
  await page.start();
  const calls: string[] = [];
  page.press = () => calls.push("press");
  page.release = () => calls.push("release");
  const space = (tag: string | null) => {
    calls.length = 0;
    const event = { code: "Space", timeStamp: 1, target: { closest: (selector: string) => tag && selector.includes(tag) ? {} : null },
      preventDefault: () => calls.push("prevent") };
    document.listeners.get("keydown")!(event);
    document.listeners.get("keyup")!(event);
    return [...calls];
  };
  for (const tag of ["input", "textarea"]) assert.deepEqual(space(tag), [], `Space types in ${tag}`);
  for (const tag of ["select", "button", null]) {
    assert.deepEqual(space(tag), ["prevent", "press", "prevent", "release"], `Space records from ${tag ?? "the page"}`);
  }
});

test("each Qwen model shows its own controls with notes: Instruction only on 1.7B, First clause split in the main row", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  const root = harness.elements.get("voice-settings");
  assert.equal(root.hidden, true, "Kristin declares no settings");
  await pickModel(harness, "qwen3-tts-1.7b");
  assert.equal(root.hidden, false);
  const [main, sampling] = root.children;
  const labels = (node: any) => nodes(node).filter((entry) => entry.className === "setting").map((entry) => entry.children[0].textContent);
  assert.deepEqual(labels(main), ["Instruction", "Language", "Seed", "First clause split", "Longest part", "Preset"]);
  assert.equal(sampling.tag, "details");
  assert.ok(labels(sampling).includes("Greedy") && labels(sampling).includes("Fine detail top k"));
  assert.equal(control(root, "First clause split").options[Number(control(root, "First clause split").value)].label, "Off");
  assert.ok(nodes(main).filter((entry) => entry.className === "setting").every((entry) => entry.children[2]?.textContent?.trim()),
    "every column has its note");
  await pickModel(harness, "qwen3-tts-0.6b");
  assert.equal(labels(root.children[0])[0], "Language", "0.6B takes no instruction");
  assert.equal(harness.elements.get("voice-copy").hidden, true);
});

test("greedy greys out its sampling; the Soft preset fills controls; changed settings are sent and named on the take row", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  await pickModel(harness, "qwen3-tts-1.7b");
  const root = harness.elements.get("voice-settings");
  choose(control(root, "Preset"), "Soft");
  assert.equal(control(root, "Temperature").options[Number(control(root, "Temperature").value)].label, "0.7");
  choose(control(root, "Greedy"), "On");
  for (const label of ["Temperature", "Top p", "Top k"]) assert.equal(control(root, label).disabled, true, label);
  assert.equal(control(root, "Repetition penalty").disabled, false);
  assert.equal(control(root, "Preset").options.at(-1).label, "Custom", "greedy matches no preset");
  choose(control(root, "Greedy"), "Off (stock)");
  choose(control(root, "Preset"), "Stock");
  choose(control(root, "First clause split"), "On");
  const instruction = control(root, "Instruction");
  instruction.value = "Speak slowly.";
  control(root, "Seed").children[1].value = "7";
  await take(harness.page, 100);
  await until(() => harness.page.run?.output);
  assert.equal(JSON.stringify(harness.spoken[0].settings), JSON.stringify({ instructions: "Speak slowly.", seed: 7, firstClause: true }));
  assert.equal(harness.elements.get("take-models").textContent,
    'Parakeet v3 MLX · qwen3-tts-1.7b, Ryan · instruction "Speak slowly.", seed 7, first clause split on');
});

test("New each reading draws an integer seed per reading and the take row shows it, even when it equals the default", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  await pickModel(harness, "qwen3-tts-0.6b");
  const root = harness.elements.get("voice-settings");
  const [mode, seed] = control(root, "Seed").children;
  choose(mode, "New each reading");
  assert.equal(seed.disabled, true);
  await take(harness.page, 100);
  await until(() => harness.page.run?.output);
  await take(harness.page, 2_000);
  await until(() => harness.spoken.length === 2 && harness.page.run?.output);
  const seeds = harness.spoken.map((request: any) => request.settings?.seed ?? 1);
  assert.ok(seeds.every((value: number) => Number.isSafeInteger(value) && value >= 0));
  assert.match(harness.elements.get("take-models").textContent, new RegExp(`seed ${seeds[1]}$`));
});

test("Read again speaks the latest transcript in the voice chosen now and keeps the last take as an earlier row", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  assert.equal(harness.elements.get("read-again").disabled, true);
  await take(harness.page, 100);
  await until(() => harness.page.run?.output && harness.page.phase === "idle");
  harness.page.render();
  assert.equal(harness.elements.get("read-again").disabled, false);
  await pickModel(harness, "qwen3-tts-0.6b");
  choose(control(harness.elements.get("voice-settings"), "First clause split"), "On");
  harness.elements.get("read-again").listeners.get("click")();
  await until(() => harness.spoken.length === 2 && harness.page.run?.output);
  assert.equal(harness.spoken[1].text, "Clear the screen.");
  assert.equal(harness.spoken[1].selection.model, "qwen3-tts-0.6b");
  assert.equal(JSON.stringify(harness.spoken[1].settings), JSON.stringify({ firstClause: true }));
  assert.equal(harness.elements.get("history").children.length, 1, "the first reading moves to Earlier takes");
});

test("interrupting Read again with a new take preserves the interrupted reading in history", async () => {
  const { page } = pageHarness();
  await page.start();
  await take(page, 100);
  await until(() => page.phase === "idle");
  page.readback.hold = true;
  const reading = page.readAgain();
  await until(() => page.phase === "speaking");
  const repeated = page.run;
  page.press(3_000);
  await reading;
  assert.notEqual(page.run, repeated);
  assert.ok(page.history.includes(repeated), "Read again must remember its own reading after a new take replaces it");
  assert.equal(repeated.output.outcome, "stopped");
  assert.equal(page.phase, "recording", "the old reading cannot reset the new take's phase");
  page.readback.hold = false;
  page.release(3_400);
  await until(() => page.phase === "idle");
});

test("a voice copy model shows the consent note; My recording needs a recording and its transcript before sending", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  await pickModel(harness, "qwen3-tts-0.6b-base");
  const panel = harness.elements.get("voice-copy");
  harness.page.render();
  assert.equal(panel.hidden, false);
  assert.ok(nodes(panel).some((node) => node.textContent?.startsWith("Copy only your own voice or the synthetic references.")));
  const voice = harness.elements.get("output-voice");
  voice.value = "recording";
  harness.page.render();
  const textarea = nodes(panel).find((node) => node.tag === "textarea");
  // A refused reading never speaks, so wait for the status line rather than the voice.
  const refused = async (at: number, message: string) => {
    harness.page.press(at);
    await new Promise((resolve) => setTimeout(resolve, 0));
    harness.page.release(at + 400);
    await until(() => harness.elements.get("status").textContent === message);
  };
  assert.match(textarea.value, /^The window is open/, "a script to read pre fills the transcript");
  await refused(100, "Record a reference first.");
  assert.equal(harness.spoken.length, 0);
  harness.page.voiceCopy.recording = { samples: new Float32Array(48_000), rate: 48_000, id: "a".repeat(64) };
  textarea.value = " ";
  await refused(2_000, "Type the words you read, or choose Copy: Voice only.");
  textarea.value = "The window is open.";
  await take(harness.page, 4_000);
  await until(() => harness.spoken.length === 1);
  assert.equal(JSON.stringify(harness.spoken[0].reference), JSON.stringify({ id: "a".repeat(64), transcript: "The window is open." }));
  assert.equal(typeof harness.spoken[0].restore, "function");
});

test("readback sends settings and a reference, keeps the applied settings, and uploads again once when the host lost it", async () => {
  const { readback, sockets } = readbackHarness();
  let restored = 0;
  const reference = { id: "a".repeat(64), transcript: "Hello." };
  const done = readback.speak({ selection, catalogueRevision: "rev", text: "Hi.", settings: { seed: 3 }, reference,
    restore: async () => { restored++; return { ...reference, id: "b".repeat(64) }; } });
  while (!sockets[0]?.sent.length) await new Promise((resolve) => setTimeout(resolve, 1));
  const socket = sockets[0]!;
  const ask = (index: number) => socket.event({ type: "output-selected", requestId: socket.sent[index].requestId,
    catalogueRevision: "rev", selectionRevision: 1, selection });
  ask(0);
  assert.deepEqual([socket.sent[1].settings, socket.sent[1].reference], [{ seed: 3 }, reference]);
  socket.event({ type: "sample-error", requestId: socket.sent[1].requestId, code: "REFERENCE_MISSING", message: "missing" });
  while (socket.sent.length < 3) await new Promise((resolve) => setTimeout(resolve, 1));
  ask(2);
  assert.equal(socket.sent[3].reference.id, "b".repeat(64));
  socket.event({ type: "sample-accepted", requestId: socket.sent[3].requestId, playbackId: 5, settings: { seed: 3, topP: 1 } });
  socket.event({ type: "sample-error", requestId: socket.sent[3].requestId, code: "REFERENCE_MISSING", message: "missing" });
  const result = await done;
  assert.equal(restored, 1, "only once");
  assert.equal(result.outcome, "error");
  assert.deepEqual(result.settings, { seed: 3, topP: 1 });
});

test("replacing a reading while its playback clear is pending settles both readings", async () => {
  const { readback, sockets, ports } = readbackHarness();
  const first = readback.speak({ selection, catalogueRevision: "rev", text: "First." });
  await until(() => sockets[0]?.sent.length);
  const socket = sockets[0]!;
  const requestId = socket.sent[0].requestId;
  socket.event({ type: "sample-accepted", requestId, playbackId: 1 });
  socket.event({ type: "playback", requestId, action: "start", playbackId: 1, sampleRate: 22_050 });
  const second = readback.speak({ selection, catalogueRevision: "rev", text: "Second." });
  await until(() => socket.sent.some((event) => event.type === "select-output" && event.requestId !== requestId));
  ports[0]!.onmessage!({ data: { kind: "drained", playbackId: 1, audioTime: 10.5 } });
  const next = socket.sent.at(-1).requestId;
  socket.event({ type: "sample-error", requestId: next, code: "SETTINGS_INVALID", message: "bad setting" });
  assert.equal((await second).outcome, "error");
  const result = await Promise.race([first, new Promise<null>((resolve) => setTimeout(() => resolve(null), 1_100))]);
  assert.equal(result?.outcome, "stopped", "replacing a job must resolve the reading it displaced");
});

test("a reading keeps its selected reference and retry transcript when the voice copy panel changes", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  const { page, spoken } = harness;
  await page.start();
  await pickModel(harness, "qwen3-tts-0.6b-base");
  harness.elements.get("output-voice").value = "recording";
  page.render();
  page.voiceCopy.recording = { samples: new Float32Array(48_000), rate: 48_000, id: "a".repeat(64) };
  page.voiceCopy.transcript.value = "The original words.";
  page.transcribe = async () => {
    harness.elements.get("output-voice").value = "ref-10s";
    page.render();
    return { text: "Hello.", finalMs: 1 };
  };
  await take(page, 100);
  await until(() => page.run?.output);
  assert.equal(spoken[0].selection.voice, "recording");
  assert.equal(spoken[0].reference?.id, "a".repeat(64), "the panel's new voice must not replace the take's reference");
  page.voiceCopy.fetch = async () => ({ ok: true, json: async () => ({ id: "b".repeat(64) }) });
  page.voiceCopy.transcript.value = "Edited after sending.";
  assert.equal(JSON.stringify(await spoken[0].restore()), JSON.stringify({ id: "b".repeat(64), transcript: "The original words." }));
  page.voiceCopy.recording = null;
  assert.equal(await spoken[0].restore(), null, "Forget must prevent a retry from restoring that recording");
});

test("voice copy reaches the host through the page's own fetch: Listen on a pick, upload and Forget", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  await pickModel(harness, "qwen3-tts-0.6b-base");
  const copy = harness.page.voiceCopy;
  const pick = harness.elements.get("output-voice").value;
  await copy.hear();
  assert.equal(copy.problem, "");
  assert.deepEqual(harness.played, [2_400], "the pick plays");
  copy.recording = { samples: new Float32Array(48_000), rate: 48_000, id: null };
  assert.equal(await copy.upload(), "d".repeat(64));
  await copy.discard();
  assert.equal(copy.problem, "");
  assert.deepEqual(harness.posted.filter((url) => url.startsWith("/speech-output/")).slice(-3), [
    `/speech-output/picks/qwen3-tts-0.6b-base/${pick}`, "/speech-output/references?sampleRate=48000", `/speech-output/references/${"d".repeat(64)}`]);
});

test("Forget removes a reference even when its upload is still in flight", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  const copy = harness.page.voiceCopy;
  const id = "c".repeat(64);
  const deleted: string[] = [];
  let uploaded!: () => void;
  copy.recording = { samples: new Float32Array(48_000), rate: 48_000, id: null };
  copy.fetch = async (url: string, { method }: { method: string }) => {
    if (method === "DELETE") { deleted.push(url); return { ok: true }; }
    await new Promise<void>((resolve) => { uploaded = resolve; });
    return { ok: true, json: async () => ({ id }) };
  };
  const upload = copy.upload();
  await copy.discard();
  uploaded();
  assert.equal(await upload, null, "an upload that was forgotten cannot become usable");
  assert.deepEqual(deleted, [`/speech-output/references/${id}`]);
  assert.equal(copy.recording, null);
});

test("reference recording reuses its stopped capture graph across repeated recordings", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  const copy = harness.page.voiceCopy;
  for (let index = 0; index < 3; ++index) {
    await copy.start();
    await copy.stop();
  }
  assert.equal(harness.microphones.length, 1, "each capture owns a worklet connected to the long lived context");
  assert.equal(harness.microphones[0].open, false, "the stopped graph retains no microphone stream");
});

test("reference capture waits for the previous worklet flush before it can record again", async (t) => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  const copy = harness.page.voiceCopy;
  t.after(() => clearInterval(copy.ticker));
  await copy.start();
  const capture = copy.capture;
  let flushed!: (take: unknown) => void;
  capture.endTake = () => new Promise((resolve) => { flushed = resolve; });
  const stopped = copy.stop();
  let restarted = false;
  capture.beginTake = () => { restarted = true; };
  await copy.start();
  flushed({ samples: new Float32Array(8_000) });
  await stopped;
  assert.equal(restarted, false, "a pending flush still owns the capture graph");
});

test("reference recording failures stay visible after the panel refreshes", async () => {
  const harness = pageHarness({ models: QWEN_MODELS });
  await harness.page.start();
  await pickModel(harness, "qwen3-tts-0.6b-base");
  harness.elements.get("output-voice").value = "recording";
  harness.page.render();
  const copy = harness.page.voiceCopy;
  const audio = copy.audio;
  copy.audio = async () => { throw new Error("permission denied"); };
  await copy.start();
  assert.match(copy.status.textContent, /Microphone unavailable: permission denied/);
  copy.recording = { samples: new Float32Array(48_000), rate: 48_000, id: null };
  copy.fetch = async () => { throw new Error("connection lost"); };
  await copy.upload();
  harness.page.render();
  assert.match(copy.status.textContent, /Upload failed: connection lost/);
  copy.recording = null;
  copy.audio = audio;
  await copy.start();
  await copy.stop();
  assert.match(copy.status.textContent, /Too short\. Record at least a second\./);
});

test("readings that overlap while the speech socket opens share one connection", async () => {
  const { readback, sockets } = readbackHarness();
  let connected!: () => void;
  const pending = new Promise<void>((resolve) => { connected = resolve; });
  readback.prepare = () => pending;
  const first = readback.speak({ selection, catalogueRevision: "rev", text: "First." });
  const second = readback.speak({ selection, catalogueRevision: "rev", text: "Second." });
  connected();
  await until(() => sockets.some((socket) => socket.sent.length));
  const socket = sockets.find((socket) => socket.sent.length)!;
  socket.event({ type: "sample-error", requestId: socket.sent[0].requestId, code: "SETTINGS_INVALID", message: "bad setting" });
  assert.equal((await first).outcome, "stopped");
  assert.equal((await second).outcome, "error");
  assert.equal(sockets.length, 1, "a superseded preparation must not open a second socket");
});
