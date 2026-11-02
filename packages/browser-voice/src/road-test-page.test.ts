import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as preRoll from "../public/pre-roll.js";
import * as text from "../public/road-test-text.js";
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

test("model cards are plain text: where it runs, size and runtime", () => {
  assert.equal(text.inputCard({ readiness: { ready: true, sizeBytes: 2_509_042_622 }, card: { runtime: "parakeet-mlx 0.5.3 on MLX, Metal GPU" } }),
    "Local · 2.5 GB · parakeet-mlx 0.5.3 on MLX, Metal GPU");
  assert.equal(text.inputCard({ readiness: { ready: false, reason: "runtime or model files need setup" }, card: { runtime: "speech.cpp" } }),
    "Local · size unknown · speech.cpp · needs setup: runtime or model files need setup");
  assert.equal(text.outputCard({ provider: { kind: "local" }, sizeBytes: 63_537_415, runtime: "sherpa-onnx VITS, CPU",
    streaming: "sentence", readiness: { state: "ready" } }), "Local · 64 MB · sherpa-onnx VITS, CPU · streams by sentence");
  assert.equal(text.transcriptMode({ card: { transcript: "final" } }), "Final text only");
  assert.equal(text.transcriptMode({ card: { transcript: "partials" } }), "Partials while you speak");
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

test("mic capture leads a take with pre roll and ends it at the worklet's flush reply", async () => {
  const posted: unknown[] = [];
  class Node { port = { onmessage: undefined as any, postMessage: (data: unknown) => posted.push(data) }; connect(node: unknown) { return node; } }
  const context = vm.createContext({ ...preRoll, navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
    AudioWorkletNode: Node, Float32Array, Promise });
  vm.runInContext(source("capture-constraints.js"), context);
  vm.runInContext(source("mic-capture.js") + "\nglobalThis.Mic = MicCapture;", context);
  const frames: unknown[] = [];
  const audio = { destination: {}, audioWorklet: { async addModule() {} },
    createGain: () => ({ gain: { value: 1 }, connect: (node: unknown) => node }), createMediaStreamSource: () => ({ connect() {}, disconnect() {} }) };
  const mic = new context.Mic({ context: audio, onFrame: (frame: unknown) => frames.push(frame) });
  await mic.start(false);
  const frame = (value: number, flush = false) => mic.node.port.onmessage({ data: { kind: "frame", samples: new Float32Array(320).fill(value), level: value, ...(flush ? { flush } : {}) } });
  for (let index = 0; index < 20; index++) frame(0.01);
  assert.equal(mic.beginTake(), 4_800, "300 ms of idle audio leads the take");
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

/** The page with real gate and text helpers; the mic and voice are scripted fakes. */
function pageHarness(rows: unknown[] = [
  { type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "processing", text: null, timings: {} } },
  { type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "complete", text: "Clear the screen.", timings: { firstUse: false, processingMs: 90 } } },
]) {
  // The real gate schedules its release tail on the window timers.
  (globalThis as any).window ??= { setTimeout, clearTimeout };
  const elements = new Map<string, any>();
  const element = () => ({ textContent: "", value: "", disabled: false, hidden: false, checked: true, className: "", style: {} as Record<string, string>,
    children: [] as any[], listeners: new Map<string, (event?: any) => void>(), classList: { toggle() {} },
    addEventListener(type: string, listener: (event?: any) => void) { this.listeners.set(type, listener); },
    setAttribute() {}, append(...nodes: any[]) { this.children.push(...nodes); }, replaceChildren(...nodes: any[]) { this.children = nodes; } });
  const select = (): any => Object.assign(element(), { options: [] as any[],
    replaceChildren(this: any, ...options: any[]) { this.options = options; this.value = options[0]?.value ?? ""; } });
  const document = { getElementById(id: string) {
    if (!elements.has(id)) elements.set(id, id.endsWith("-model") || id.endsWith("-voice") ? select() : element());
    return elements.get(id);
  }, createElement: element, addEventListener() {} };
  const spoken: any[] = [];
  const posted: string[] = [];
  class Mic { open = true; recording = false; async start() {} beginTake() { this.recording = true; return 0; }
    async endTake() { this.recording = false; return { samples: new Float32Array(8_000).fill(0.1), preRollSamples: 0 }; } }
  class Readback { active = false; playing = false; async prepare() {}
    async speak(request: any) { spoken.push(request); return { outcome: "finished", firstAudioMs: 120, totalMs: 900, samples: new Float32Array(4), sampleRate: 22_050 }; }
    stop() {} }
  const setups = { setups: [{ id: "mlx-parakeet/v3-greedy", label: "Parakeet v3 MLX", card: { runtime: "mlx", transcript: "final" }, readiness: { ready: true, sizeBytes: 1 } }] };
  const voices = { revision: "rev", default: { model: "piper-kristin-medium" }, models: [{ id: "piper-kristin-medium", label: "Piper Kristin medium",
    provider: { kind: "local" }, defaultVoice: "kristin", defaultSpeed: 1, voices: [{ id: "kristin", label: "Kristin" }], readiness: { state: "ready" } }] };
  const fetch = async (url: string) => {
    posted.push(url);
    if (url === "/prompt-reader/setups") return { ok: true, json: async () => setups };
    if (url === "/speech-output/models") return { ok: true, json: async () => voices };
    const body = new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(rows.map((row) => JSON.stringify(row)).join("\n")));
      controller.close();
    } });
    return { ok: true, status: 200, body };
  };
  const context = vm.createContext({ document, fetch, performance, setTimeout, clearTimeout, URLSearchParams, TextEncoder, TextDecoder,
    JSON, Float32Array, Promise, Option: function(label: string, value: string) { return { label, value }; },
    HoldReleaseGate, MicCapture: Mic, VoiceReadback: Readback, TalkOverDetector, ...text,
    AudioContext: class { async resume() {} }, window: { setTimeout, clearTimeout } });
  vm.runInContext(source("road-test.js") + "\nglobalThis.Page = RoadTestPage;", context);
  return { page: new context.Page(), elements, spoken, posted };
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
  assert.equal(elements.get("transcript-mode").textContent, "Final text only");
  const timings = elements.get("output-timings").children.map((row: any) => row.children.map((cell: any) => cell.textContent).join(": "));
  assert.ok(timings.includes("Text to first audio: 120 ms"), timings.join(" | "));
  assert.ok(timings.includes("Text to end: 900 ms"));
  assert.equal(page.phase, "idle");
});

test("model start sums every load phase, so a cleanup row counts its own model after Ultra", async () => {
  const row = (status: string, extra = {}) => ({ type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status, text: null, timings: {}, ...extra } });
  const { page } = pageHarness([row("loading"), row("processing"), row("loading"), row("processing"),
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
