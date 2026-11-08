import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as text from "../public/road-test-text.js";
import * as trail from "../public/live-trail.js";
import { finalSetup } from "./final-settings.ts";
import { liveStream } from "./live-settings.ts";
import { TalkOverDetector } from "../public/talk-over.js";
import { VoiceReadback } from "../public/voice-readback.js";
import { CLEANUP_MODELS as CLEANUP_REGISTRY } from "./text-cleanup.ts";
import { DEFAULT_NAMES_SNAPSHOT, MAX_NAME_BYTES, MAX_NAMES, namesSnapshot, parseNames } from "./names.ts";

/** A page module's source, run in one vm script: its imports and export keywords stripped. */
export const source = (name: string) => readFileSync(new URL(`../public/${name}`, import.meta.url), "utf8")
  .replace(/^import .*;$/gm, "").replace(/^export /gm, "");

export type FetchReply = { ok: boolean; status?: number; body?: ReadableStream; json?(): Promise<unknown> };

/** The cleanups route's catalogue, in the shape of the cleanup contract (design section 3). */
export const CLEANUP_MODELS = CLEANUP_REGISTRY.map((model) => ({ ...model,
  readiness: { ready: true, ...(model.id === "superwhisper/s1-mini" ? { sizeBytes: 1_200_000_000 } : {}) } }));

/** A warm cleanup reply in the contract's shape. */
export function cleanupReply(body: { requestId: string; modelId: string }, text: string, extra = {}): FetchReply {
  return { ok: true, status: 200, json: async () => ({ requestId: body.requestId, modelId: body.modelId, status: "complete", text,
    processingMs: 40, workerWallMs: 45, loadMs: null, firstUse: false, settings: {}, ...extra }) };
}

/** The names route's reply in its contract's shape. */
export function namesReply(names: readonly string[]): FetchReply {
  return { ok: true, status: 200, json: async () => ({ ...namesSnapshot(names), limits: { maxNames: MAX_NAMES, maxNameBytes: MAX_NAME_BYTES } }) };
}

/** Answers each names save as the host does: the canonical list, or 400 with the rule broken. */
export async function canonicalNames(body: { names: unknown }): Promise<FetchReply> {
  try { return namesReply(parseNames(body.names)); }
  catch (error) { return { ok: false, status: 400, json: async () => ({ error: (error as Error).message }) }; }
}

/** Answers each cleanup with its text capitalised and ending in a full stop. */
export async function echoCleanup(body: { requestId: string; modelId: string; text: string }): Promise<FetchReply> {
  return cleanupReply(body, `${body.text[0]!.toUpperCase()}${body.text.slice(1).replace(/\.?$/, ".")}`);
}

/**
 * One fake page context: a socket, a playback worklet port and an output clock.
 * Sockets open on the next microtask; with manualOpen they stay CONNECTING until `open()`.
 */
export function readbackHarness({ manualOpen = false } = {}) {
  const sockets: FakeSocket[] = [];
  const ports: Array<{ messages: any[]; onmessage?: (event: { data: any }) => void; postMessage(data: any): void }> = [];
  class FakeSocket {
    static OPEN = 1;
    readyState = 0;
    sent: any[] = [];
    listeners = new Map<string, Array<(event: any) => void>>();
    constructor() { sockets.push(this); if (!manualOpen) queueMicrotask(() => this.open()); }
    open() { this.readyState = 1; this.fire("open", {}); }
    close() { this.readyState = 3; this.fire("close", {}); }
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

export function packet(playbackId: number, samples: number[]) {
  const bytes = new ArrayBuffer(9 + samples.length * 4);
  const view = new DataView(bytes);
  view.setUint8(0, 1);
  view.setUint32(1, playbackId, true);
  view.setUint32(5, 22_050, true);
  samples.forEach((sample, index) => view.setFloat32(9 + index * 4, sample, true));
  return bytes;
}

export const selection = { schema: 1, provider: { kind: "local" }, model: "piper-kristin-medium", voice: "kristin", speed: 1 };


/**
 * The page with real gate, detector and text helpers; the mic and voice are
 * scripted fakes. With holdVoice the voice plays until stopped, then reports
 * a 41 ms stop; clearing the readback's hold lets later readings finish. Each mic take is one sample longer than the last, so replays
 * show which take they play. With transcript "partials" the setup streams: live records what reaches the live words channel.
 * Cleanup models come from `cleanups`; each cleanup request is recorded and answered by `cleanupReply`, and
 * `stored` seeds the page's localStorage. `comparisonEnd` holds the comparison stream open after its rows.
 * The names route answers `names` and each save with `saveNames`; `namesSaves` records each save's body.
 */
export function pageHarness({ holdVoice = false, transcript = "final", models = [] as unknown[], cleanups = CLEANUP_MODELS as unknown[],
  cleanupReply = echoCleanup, comparisonEnd = Promise.resolve(), stored = {} as Record<string, string>,
  names = namesReply(DEFAULT_NAMES_SNAPSHOT.names) as FetchReply, saveNames = canonicalNames }: { holdVoice?: boolean; transcript?: string; models?: unknown[];
  comparisonEnd?: Promise<void>; names?: FetchReply; saveNames?: (body: any) => Promise<FetchReply>;
  cleanups?: unknown[]; cleanupReply?: (body: any) => Promise<FetchReply>; stored?: Record<string, string> } = {}, rows: unknown[] = [
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
  const cleanupRequests: any[] = [];
  const namesSaves: any[] = [];
  async function fetch(this: unknown, url: string, init?: { method?: string; body?: string }) {
    if (this !== undefined && this !== realm) throw new TypeError("Illegal invocation");
    posted.push(url);
    if (url === "/prompt-reader/cleanups" && init?.method === "POST") {
      const body = JSON.parse(init.body!);
      cleanupRequests.push(body);
      return cleanupReply(body);
    }
    if (url === "/prompt-reader/cleanups") return { ok: true, json: async () => ({ schema: 1, models: cleanups }) };
    if (url === "/road-test/names" && init?.method === "PUT") {
      const body = JSON.parse(init.body!);
      namesSaves.push(body);
      return saveNames(body);
    }
    if (url === "/road-test/names") return names;
    if (url.startsWith("/speech-output/picks/")) return { ok: true, arrayBuffer: async () => new ArrayBuffer(8) };
    if (url.startsWith("/speech-output/references")) return { ok: true, json: async () => (init?.method === "POST" ? { id: "d".repeat(64) } : {}) };
    if (url === "/prompt-reader/setups") return { ok: true, json: async () => setups };
    if (url === "/speech-output/models") return { ok: true, json: async () => voices };
    const body = new ReadableStream({ async start(controller) {
      controller.enqueue(new TextEncoder().encode(rows.map((row) => JSON.stringify(row)).join("\n") + "\n"));
      await comparisonEnd;
      controller.close();
    } });
    return { ok: true, status: 200, body };
  }
  const storage = { items: new Map(Object.entries(stored)), getItem(key: string) { return this.items.get(key) ?? null; },
    setItem(key: string, value: string) { this.items.set(key, value); } };
  const context = vm.createContext({ document, fetch, performance, Math, Number, String, Object, setInterval, clearInterval, setTimeout, clearTimeout, URLSearchParams, TextEncoder, TextDecoder,
    JSON, Float32Array, Promise, Option: function(label: string, value: string) { return { label, value }; },
    HoldReleaseGate, CaptureSession: Mic, VoiceReadback: Readback, LiveWords, TalkOverDetector, ...text, ...trail,
    AudioContext: class { currentTime = 0; sampleRate = 48_000; destination = {}; async resume() {}
      createBuffer(_channels: number, length: number) { return { length, copyToChannel() {} }; }
      async decodeAudioData() { return { sampleRate: 24_000, getChannelData: () => new Float32Array(2_400) }; }
      createBufferSource() { const source = { buffer: { length: 0 }, connect() {}, stop() {}, start() { played.push(source.buffer.length); } }; return source; } },
    window: { setTimeout, clearTimeout }, localStorage: storage });
  realm = vm.runInContext("globalThis", context);
  vm.runInContext(["setting-controls.js", "names.js", "voice-settings.js", "voice-copy.js", "text-cleanup.js", "take-history.js", "road-test.js"].map(source).join("\n")
    + "\nglobalThis.Page = RoadTestPage;", context);
  return { page: new context.Page(), elements, spoken, posted, played, live, document, microphones, cleanups: cleanupRequests, storage, namesSaves };
}

export const until = async (done: () => unknown) => {
  for (let tries = 0; !done(); tries++) {
    if (tries > 200) throw new Error("timed out");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
};
export const listed = (list: any) => list.children.map((row: any) => row.children.map((cell: any) => cell.textContent).join(": "));

/** Holds and releases one take, then waits for the voice to start or finish. */
export async function take(page: any, pressAt: number) {
  page.press(pressAt);
  await new Promise((resolve) => setTimeout(resolve, 0));
  page.release(pressAt + 400);
  await until(() => page.phase === "speaking" || page.run?.output);
}


/** Every node under root, depth first. */
export function nodes(root: any): any[] {
  return [root, ...(root.children ?? []).flatMap(nodes)];
}
/** The control in the setting column labelled `label`: its select or input, or for Seed the mode and number pair. */
export function control(root: any, label: string): any {
  const column = nodes(root).find((node) => node.className === "setting" && node.children[0]?.textContent === label);
  return column?.children[1];
}
export const choose = (select: any, label: string) => {
  const index = select.options.findIndex((option: any) => option.label === label);
  assert.ok(index >= 0, `${label} among ${select.options.map((option: any) => option.label)}`);
  select.value = select.options[index].value;
  select.listeners.get("change")?.();
};
