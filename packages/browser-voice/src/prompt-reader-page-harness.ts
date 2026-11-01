import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as preRoll from "../public/pre-roll.js";

export type PagePort = { onmessage?: (event: { data: unknown }) => void; postMessage(data: unknown): void };

/** Stands in for the worklet: answers each flush with an empty marked frame. */
function echoEmptyFlush(port: PagePort, data: unknown) {
  if ((data as { kind?: string })?.kind !== "flush") return;
  port.onmessage?.({ data: { kind: "frame", samples: new Float32Array(0), level: 0, flush: true } });
}

export type FetchInit = { method?: string; body?: ArrayBuffer | string; signal?: AbortSignal };
export type FetchReply = { ok: boolean; status?: number; body?: ReadableStream; json?(): Promise<unknown>; arrayBuffer?(): Promise<ArrayBuffer> };

export function okReply(body: unknown = {}): FetchReply {
  return { ok: true, json: async () => body, arrayBuffer: async () => new Float32Array([0.5, -0.5]).buffer };
}

/** Execute the shipped page with controlled mic acquisition, worklet messages and fetch replies. */
export function pageHarness({
  worklet = echoEmptyFlush,
  respond = () => Promise.resolve(okReply()),
}: {
  worklet?: (port: PagePort, data: unknown) => void;
  respond?: (url: string, init?: FetchInit) => Promise<FetchReply>;
} = {}) {
  const pending: Array<(stream: unknown) => void> = [];
  const constraints: unknown[] = [];
  const denied: Array<(error: Error) => void> = [];
  const connected: unknown[] = [];
  const played: Array<{ samples: Float32Array }> = [];
  const fetched: string[] = [];
  const posted: unknown[] = [];
  const elements = new Map<string, Record<string, unknown>>();
  const listeners = new Map<string, (event?: unknown) => void>();
  const listen = (target: string) => (type: string, listener: (event?: unknown) => void) => listeners.set(`${target}:${type}`, listener);
  const element = () => ({ children: [] as unknown[], textContent: "", className: "", value: "", append(...nodes: unknown[]) { this.children.push(...nodes); }, replaceChildren(...nodes: unknown[]) { this.children = nodes; }, addEventListener() {} });
  const document = {
    body: { dataset: {} },
    hidden: false,
    getElementById(id: string) {
      if (!elements.has(id)) {
        elements.set(id, { ...element(), addEventListener: listen(id) });
      }
      return elements.get(id);
    },
    createElement: element,
    addEventListener: listen("document"),
  };
  /** Fire a page lifecycle event the reader listens for, such as `document:visibilitychange`. */
  const dispatch = (key: string, event?: unknown) => listeners.get(key)!(event);
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
    ...preRoll, document, AudioContext, AudioWorkletNode, HoldReleaseGate,
    Float32Array, URLSearchParams, AbortController, TextDecoder,
    Option: function(text: string, value: string) { return { text, value }; },
    fetch: (url: string, init?: FetchInit) => {
      fetched.push(url);
      return respond(url, init);
    },
    navigator: { mediaDevices: { getUserMedia: (options: unknown) => {
      constraints.push(options);
      return new Promise((resolve, reject) => { pending.push(resolve); denied.push(reject); });
    } } },
    window: { setTimeout, clearTimeout, addEventListener: listen("window") },
  });
  vm.runInContext(readFileSync(new URL("../public/capture-constraints.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/speech-observations.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(readFileSync(new URL("../public/setup-results.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "").replace(/^export /gm, ""), context);
  vm.runInContext(source, context);
  const reader = new context.Reader();
  reader.comparison.mode = "current-list";
  reader.bind();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this" }];
  return { reader, document, dispatch, pending, denied, connected, constraints, played, fetched, posted, expectedConstraints: vm.runInContext("CAPTURE_AUDIO_CONSTRAINTS", context) };
}

export function mic() {
  const track = { label: "test mic", stopped: false, stop() { this.stopped = true; } };
  return { getTracks: () => [track], getAudioTracks: () => [track], track };
}

export async function waitForRequests(pending: unknown[], count: number) {
  for (let tries = 0; pending.length < count && tries < 20; tries++) await Promise.resolve();
  assert.equal(pending.length, count);
}


/** Record one ready take of 0.4 s through the page's capture path. */
export async function readyTake(harness: ReturnType<typeof pageHarness>) {
  const { reader, pending } = harness;
  const cold = !reader.stream;
  const opening = reader.press();
  if (cold) {
    await waitForRequests(pending, pending.length + 1);
    pending[pending.length - 1]!(mic());
  }
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(0.3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  return reader.take;
}


/** Browser fixtures use the actual immutable manifest, never a second preset table. */
export function comparisonCatalog() {
  const manifest = JSON.parse(readFileSync(new URL("../runtime/setups.json", import.meta.url), "utf8"));
  return { ...manifest, setups: manifest.setups.map((setup: { id: string }) => ({ ...setup, readiness: { ready: false, reason: "test runtime unavailable" } })) };
}

export function enableComparison(reader: any, ids?: string[]) {
  reader.comparison.mode = "compare";
  reader.comparison.catalog = comparisonCatalog();
  reader.comparison.renderPresets();
  if (ids) reader.comparison.selected = new Set(ids);
}

export function streamReply() {
  let stream: ReadableStreamDefaultController | undefined;
  const body = new ReadableStream({ start(controller) { stream = controller; } });
  return { reply: { ok: true, body }, send: (event: unknown) => stream!.enqueue(new TextEncoder().encode(JSON.stringify(event) + "\n")), end: () => stream!.close() };
}

export function rowEvent(requestId: string, setupId: string, status = "complete", text: string | null = "raw words\n") {
  return { type: "row", requestId, comparisonId: "server-job", row: {
    setupId, status, text, pcm: { sampleRate: 16000, channels: 1, sampleCount: 6400, pcmBytes: 25600, pcmSha256: "same-digest" },
    timings: { queueMs: 0, startupMs: null, loadMs: 12, firstUse: true, processingMs: 30, workerWallMs: 32, requestWallMs: 48, serviceWallMs: 49, dependencyProcessingMs: null, deliveryMs: null },
  } };
}

export function acceptStream(stream: ReturnType<typeof streamReply>, requestId: string) {
  stream.send({ type: "accepted", requestId, comparisonId: "server-job", queued: false, setupIds: [], pcm: { sampleCount: 6400, pcmSha256: "same-digest" } });
}
