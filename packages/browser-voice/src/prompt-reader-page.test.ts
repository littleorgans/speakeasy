import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { HoldReleaseGate } from "../public/hold-release.js";
import * as preRoll from "../public/pre-roll.js";
import { feed, loadCapture } from "./capture-worklet-harness.ts";

type PagePort = { onmessage?: (event: { data: unknown }) => void; postMessage(data: unknown): void };

/** Stands in for the worklet: answers each flush with an empty marked frame. */
function echoEmptyFlush(port: PagePort, data: unknown) {
  if ((data as { kind?: string })?.kind !== "flush") return;
  port.onmessage?.({ data: { kind: "frame", samples: new Float32Array(0), level: 0, flush: true } });
}

type FetchInit = { method?: string; body?: ArrayBuffer | string; signal?: AbortSignal };
type FetchReply = { ok: boolean; status?: number; body?: ReadableStream; json?(): Promise<unknown>; arrayBuffer?(): Promise<ArrayBuffer> };

function okReply(body: unknown = {}): FetchReply {
  return { ok: true, json: async () => body, arrayBuffer: async () => new Float32Array([0.5, -0.5]).buffer };
}

/** Execute the shipped page with controlled mic acquisition, worklet messages and fetch replies. */
function pageHarness({
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
  vm.runInContext(readFileSync(new URL("../public/setup-results.js", import.meta.url), "utf8").replace(/^export /gm, ""), context);
  vm.runInContext(source, context);
  const reader = new context.Reader();
  reader.comparison.mode = "current-list";
  reader.bind();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this" }];
  return { reader, document, dispatch, pending, denied, connected, constraints, played, fetched, posted, expectedConstraints: vm.runInContext("CAPTURE_AUDIO_CONSTRAINTS", context) };
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

test("each take drains its worklet tail, so the next take holds none of it", async () => {
  // Page and real worklet, connected through asynchronous ports at 48 kHz.
  let page: PagePort | undefined;
  const { context, Processor } = loadCapture(48_000, (message) =>
    queueMicrotask(() => page?.onmessage?.({ data: message })),
  );
  const processor = new Processor();
  const { reader, pending, posted } = pageHarness({
    worklet: (_port, data) => queueMicrotask(() => processor.port.onmessage?.({ data })),
  });
  const timers: Array<() => void> = [];
  reader.gate.schedule = (callback: () => void) => timers.push(callback);
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const takes: Float32Array[] = [];
  let block = 0;
  for (const value of [1, 0]) {
    const opening = reader.press();
    await waitForRequests(pending, 1);
    if (!takes.length) pending[0]!(mic());
    await opening;
    page = reader.capture.port;
    feed(context, processor, block, (block += 55), value);
    await settle();
    reader.gate.release();
    feed(context, processor, block, (block += 75), value);
    await settle();
    timers.shift()!();
    // Saved audio must stay blocked until the new take's flush reply arrives.
    reader.prompts[0].takes = ["previous.wav"];
    reader.render();
    assert.equal(reader.elements.play.disabled, true);
    await reader.play();
    assert.equal(reader.playback, null);
    await settle();
    takes.push(reader.take.samples);
  }
  // 130 blocks of 128 source samples: 17 frames of 320 plus a 107 sample tail.
  assert.ok(takes[1]!.every((sample) => sample === 0), "second take holds no earlier samples");
  assert.equal(takes[1]!.length, 17 * 320 + 107);
  assert.equal(takes[0]!.length, 17 * 320 + 107, "first take keeps its drained tail");
  assert.ok(takes[0]!.every((sample) => sample === 1));
  assert.equal(posted.filter((data) => (data as { kind?: string }).kind === "flush").length, 2);
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

test("play replays the unsaved take without opening the mic or changing it", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  const opening = reader.press();
  await waitForRequests(pending, 1);
  pending[0]!(mic());
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(0.3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  const take = reader.take;
  const before = Array.from(take.samples);
  await reader.play();
  assert.equal(pending.length, 1, "playback must not request the mic");
  assert.equal(reader.gate.capturing, false);
  assert.equal(reader.take, take);
  assert.deepEqual(Array.from(take.samples), before);
  assert.deepEqual(Array.from(played[0]!.samples), before);
  assert.deepEqual(fetched, ["/prompt-reader/transcriptions"], "only the display transcript is fetched");
  assert.equal(reader.elements.play.disabled, false);
});

test("play on a revisited prompt fetches the latest saved take; T steps to earlier ones", async () => {
  const { reader, pending, played, fetched } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: true, takes: ["utt-p001-a.wav", "utt-p001-b.wav"] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, false);
  assert.equal(reader.elements.playLabel.textContent, "Play take 2 of 2");
  assert.equal(reader.elements.earlier.hidden, false);
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav"]);
  assert.equal(played.length, 1);
  reader.stepTake();
  await reader.play();
  assert.deepEqual(fetched, ["/prompt-reader/takes/utt-p001-b.wav", "/prompt-reader/takes/utt-p001-a.wav"]);
  assert.equal(pending.length, 0, "playback must not request the mic");
  assert.equal(reader.take, null);
});

test("play is disabled on a prompt with no take", () => {
  const { reader } = pageHarness();
  reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  reader.render();
  assert.equal(reader.elements.play.disabled, true);
  assert.equal(reader.elements.earlier.hidden, true);
});

/** Record one ready take of 0.4 s through the page's capture path. */
async function readyTake(harness: ReturnType<typeof pageHarness>) {
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

const settle = () => new Promise((resolve) => setImmediate(resolve));
const heardWords = (reader: { elements: { heardText: { children?: Array<{ textContent: string; className: string }> } } }) =>
  (reader.elements.heardText.children ?? []).map(({ textContent, className }) => [textContent, className]);

test("a ready take shows its transcript under the prompt with differing words marked", async () => {
  const posted: Array<{ url: string; init?: FetchInit }> = [];
  const harness = pageHarness({
    respond: async (url, init) => {
      posted.push({ url, init });
      return okReply({ text: "Read, that THIS!", decodeMs: 155.3, model: "Canary 180M Flash" });
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  const take = await readyTake(harness);
  assert.equal(posted[0]!.url, "/prompt-reader/transcriptions");
  assert.equal(posted[0]!.init!.body, take.samples.buffer, "the take's own samples are sent");
  await settle();
  const { reader } = harness;
  assert.equal(reader.elements.heard.hidden, false);
  assert.deepEqual(heardWords(reader), [["Read,", ""], [" that", "differs"], [" THIS!", ""]]);
  assert.equal(reader.elements.heardNote.textContent, "Canary 180M Flash · 155 ms");
});

test("save works while the transcript is pending and never carries it", async () => {
  const saves: Array<{ url: string; init?: FetchInit }> = [];
  const harness = pageHarness({
    respond: (url, init) => {
      if (url === "/prompt-reader/transcriptions") return new Promise(() => {});
      saves.push({ url, init });
      return Promise.resolve(okReply({ audio: "utt-p001-a.wav" }));
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  const take = await readyTake(harness);
  const { reader } = harness;
  assert.equal(reader.elements.heardNote.textContent, "Transcribing");
  await reader.save();
  assert.equal(saves.length, 1);
  assert.match(saves[0]!.url, /^\/prompt-reader\/recordings\?promptId=p001&device=test\+mic&preRollMs=0$/);
  assert.equal(saves[0]!.init!.body, take.samples.buffer);
  assert.equal(reader.take, null);
  assert.equal(reader.index, 1);
  assert.equal(reader.elements.heard.hidden, true, "the transcript leaves with its take");
});

test("a failed transcript shows a quiet note and save still works", async () => {
  let saved = 0;
  const harness = pageHarness({
    respond: async (url) => {
      if (url === "/prompt-reader/transcriptions") return { ok: false, status: 503, json: async () => ({ error: "child exited" }) };
      saved += 1;
      return okReply({ audio: "utt-p001-a.wav" });
    },
  });
  harness.reader.prompts = [{ id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] }];
  await readyTake(harness);
  await settle();
  const { reader } = harness;
  assert.equal(reader.elements.heard.hidden, false);
  assert.equal(reader.elements.heardNote.textContent, "Transcript unavailable");
  assert.deepEqual(heardWords(reader), []);
  await reader.save();
  assert.equal(saved, 1);
  assert.equal(reader.index, 1);
});

test("a late transcript never lands on a newer take", async () => {
  const replies: Array<(reply: FetchReply) => void> = [];
  const harness = pageHarness({
    respond: () => new Promise((resolve) => replies.push(resolve)),
  });
  const first = await readyTake(harness);
  const second = await readyTake(harness);
  replies[0]!(okReply({ text: "stale words", decodeMs: 1, model: "M" }));
  await settle();
  assert.equal(first.transcript.state, "done");
  assert.equal(second.transcript.state, "pending");
  assert.equal(harness.reader.elements.heardNote.textContent, "Transcribing");
});

test("each take shows its own applied list count and model after later file changes", async () => {
  let count = 2;
  const harness = pageHarness({ respond: async () => okReply({
    text: "read this", decodeMs: 3, model: "en-parakeet-ultra-int8",
    hotwords: { path: "/owner/hotwords.txt", count, revision: String(count) },
  }) });
  const first = await readyTake(harness);
  await settle();
  assert.match(harness.reader.elements.heardHotwords.textContent, /2 entries applied/);
  assert.match(harness.reader.elements.heardNote.textContent, /en-parakeet-ultra-int8/);
  count = 0;
  const second = await readyTake(harness);
  await settle();
  assert.match(harness.reader.elements.heardHotwords.textContent, /No list.*greedy/);
  assert.equal(first.transcript.hotwords.count, 2);
  assert.equal(second.transcript.hotwords.count, 0);
});

test("a rejected list displays exact reasons and keeps the original take saveable", async () => {
  const reason = 'hotword "a☃": unencodable UTF8 coverage at byte 4\nhotword "a :2": reserved native score syntax';
  let saved: ArrayBuffer | string | undefined;
  const harness = pageHarness({ respond: async (url, init) => {
    if (url === "/prompt-reader/transcriptions") return {
      ok: false, status: 503, json: async () => ({ error: reason, model: "en-parakeet-ultra-int8",
        hotwords: { path: "/owner/hotwords.txt", count: 3, revision: "bad" } }),
    };
    saved = init?.body;
    return okReply({ audio: "utt-p001-a.wav" });
  } });
  const take = await readyTake(harness);
  await settle();
  assert.equal(harness.reader.elements.heardError.textContent, reason);
  assert.match(harness.reader.elements.heardHotwords.textContent, /3 entries.*failed/);
  assert.match(harness.reader.elements.heardNote.textContent, /en-parakeet-ultra-int8/);
  assert.equal(harness.reader.take, take);
  await harness.reader.play();
  assert.equal(harness.played[0]?.samples.length, take.samples.length);
  await harness.reader.save();
  assert.equal(saved, take.samples.buffer);
});

const idleFrames = (count: number, value: number) => Array.from({ length: count }, () => new Float32Array(320).fill(value));

test("a take starts with the 300 ms before its press", async () => {
  const saves: string[] = [];
  const harness = pageHarness({
    respond: async (url) => {
      if (url.startsWith("/prompt-reader/recordings")) saves.push(url);
      return okReply({ audio: "utt-a.wav" });
    },
  });
  const { reader, pending } = harness;
  reader.prompts = [
    { id: "p001", stratum: "short", text: "read this", recorded: false, takes: [] },
    { id: "p002", stratum: "short", text: "read that", recorded: false, takes: [] },
  ];
  const opening = reader.press();
  await waitForRequests(pending, 1);
  const device = mic();
  pending[0]!(device);
  await opening;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(1), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(device.track.stopped, false, "the mic stays open after commit");
  assert.equal(reader.take.preRollMs, 0, "the page's first take opens a cold mic");
  await reader.save();
  // Idle frames older than 300 ms fall out of the ring; the idle level stays hidden.
  for (const samples of [...idleFrames(10, 9), ...idleFrames(15, 2)]) reader.onFrame({ kind: "frame", samples, level: 0.4 });
  assert.equal(reader.elements.level.value, 0);
  const second = reader.press();
  await second;
  reader.onFrame({ kind: "frame", samples: new Float32Array(6_400).fill(3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(pending.length, 1, "one getUserMedia across both takes");
  const { samples } = reader.take;
  assert.equal(samples.length, 4_800 + 6_400);
  assert.ok(samples.subarray(0, 4_800).every((sample: number) => sample === 2), "pre roll is the idle ring");
  assert.ok(samples.subarray(4_800).every((sample: number) => sample === 3));
  assert.equal(reader.take.preRollMs, 300);
  await reader.save();
  assert.deepEqual(saves.map((url) => new URLSearchParams(url.split("?")[1]).get("preRollMs")), ["0", "300"]);
});

for (const [event, hide] of [["document:visibilitychange", true], ["window:pagehide", false]] as const) {
  test(`${event.split(":")[1]} stops the warm mic and the next press reopens it cold`, async () => {
    const harness = pageHarness();
    const { reader, pending, document, dispatch } = harness;
    await readyTake(harness);
    const device = reader.stream.getAudioTracks()[0];
    for (const samples of idleFrames(15, 2)) reader.onFrame({ kind: "frame", samples, level: 0.4 });
    document.hidden = hide;
    dispatch(event);
    assert.equal(device.stopped, true);
    document.hidden = false;
    const take = await readyTake(harness);
    assert.equal(pending.length, 2, "the next press opens the mic again");
    assert.equal(take.preRollMs, 0);
    assert.equal(take.samples.length, 6_400, "no idle frame from before the stop leads the take");
  });
}

test("a tap on a warm mic stays too short despite its pre roll", async () => {
  const harness = pageHarness();
  const { reader } = harness;
  await readyTake(harness);
  for (const samples of idleFrames(15, 2)) reader.onFrame({ kind: "frame", samples, level: 0.4 });
  await reader.press();
  reader.onFrame({ kind: "frame", samples: new Float32Array(3_200).fill(3), level: 0.2 });
  reader.gate.cancel();
  reader.finishTake();
  assert.equal(reader.take, null);
  assert.match(reader.elements.status.textContent, /too short/);
});

/** Browser fixtures use the actual immutable manifest, never a second preset table. */
function comparisonCatalog() {
  const manifest = JSON.parse(readFileSync(new URL("../runtime/setups.json", import.meta.url), "utf8"));
  return { ...manifest, setups: manifest.setups.map((setup: { id: string }) => ({ ...setup, readiness: { ready: false, reason: "test runtime unavailable" } })) };
}

function enableComparison(reader: any, ids?: string[]) {
  reader.comparison.mode = "compare";
  reader.comparison.catalog = comparisonCatalog();
  reader.comparison.renderPresets();
  if (ids) reader.comparison.selected = new Set(ids);
}

function streamReply() {
  let stream: ReadableStreamDefaultController | undefined;
  const body = new ReadableStream({ start(controller) { stream = controller; } });
  return { reply: { ok: true, body }, send: (event: unknown) => stream!.enqueue(new TextEncoder().encode(JSON.stringify(event) + "\n")), end: () => stream!.close() };
}

function rowEvent(requestId: string, setupId: string, status = "complete", text: string | null = "raw words\n") {
  return { type: "row", requestId, comparisonId: "server-job", row: {
    setupId, status, text, pcm: { sampleRate: 16000, channels: 1, sampleCount: 6400, pcmBytes: 25600, pcmSha256: "same-digest" },
    timings: { queueMs: 0, startupMs: null, loadMs: 12, firstUse: true, processingMs: 30, workerWallMs: 32, requestWallMs: 48, serviceWallMs: 49, dependencyProcessingMs: null, deliveryMs: null },
  } };
}

function acceptStream(stream: ReturnType<typeof streamReply>, requestId: string) {
  stream.send({ type: "accepted", requestId, comparisonId: "server-job", queued: false, setupIds: [], pcm: { sampleCount: 6400, pcmSha256: "same-digest" } });
}

test("comparison defaults to only v3 in exact manifest order and keeps unavailable identity visible", () => {
  const { reader } = pageHarness();
  enableComparison(reader);
  assert.equal(reader.comparison.checks.length, 5);
  assert.deepEqual(reader.comparison.checks.map((entry: any) => entry.setup.id), comparisonCatalog().setups.map((setup: any) => setup.id));
  assert.deepEqual(reader.comparison.checks.filter((entry: any) => entry.input.checked).map((entry: any) => entry.setup.id), ["mlx-parakeet/v3-greedy"]);
  assert.match(reader.comparison.checks[0].status.textContent, /Needs setup: test runtime unavailable/);
  assert.equal(reader.comparison.checks[0].tags.textContent, "Recommended");
  assert.equal(reader.comparison.checks[2].tags.textContent, "Uses fixed vocabulary");
  assert.match(reader.comparison.checks[3].fact.textContent, /Sherpa Ultra on CPU/);
  assert.equal(reader.comparison.checks.some((entry: any) => /Streams|Accuracy/.test(entry.tags.textContent)), false);
});

test("one canonical buffer is submitted once, rows arrive progressively in selection order, save and play remain independent", async () => {
  const stream = streamReply();
  const posts: Array<{ url: string; init?: FetchInit }> = [];
  const h = pageHarness({ respond: async (url, init) => {
    posts.push({ url, init });
    return url.startsWith("/prompt-reader/comparisons") ? stream.reply : okReply({ audio: "utt-p001-a.wav" });
  } });
  const ids = ["mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy"];
  enableComparison(h.reader, ids);
  const take = await readyTake(h);
  const request = posts[0]!;
  assert.equal(posts.length, 1);
  assert.match(request.url, /^\/prompt-reader\/comparisons\?/);
  assert.equal(request.init?.body, take.samples.buffer);
  const query = new URLSearchParams(request.url.split("?")[1]);
  assert.deepEqual(JSON.parse(query.get("setupIds")!), ids);
  assert.equal(query.has("prompt"), false);
  const requestId = query.get("requestId")!;
  acceptStream(stream, requestId);
  stream.send(rowEvent(requestId, ids[1]!));
  await settle();
  assert.equal(h.reader.comparison.rows[0].result, null);
  assert.equal(h.reader.comparison.rows[1].result.text, "raw words\n");
  assert.equal(h.reader.elements.save.disabled, false);
  assert.equal(h.reader.elements.play.disabled, false);
  await h.reader.play();
  assert.deepEqual(Array.from(h.played[0]!.samples), Array.from(take.samples));
  await h.reader.save();
  assert.equal(posts[1]!.init?.body, take.samples.buffer);
  assert.equal(request.init?.signal?.aborted, true);
  stream.send(rowEvent(requestId, ids[0]!, "complete", "late text"));
  stream.end();
  await settle();
  assert.equal(h.reader.comparison.rows.length, 0);
  assert.equal(h.reader.index, 1);
});

test("selected saved comparison posts only its filename and never decodes or uploads playback samples", async () => {
  const stream = streamReply();
  const posts: Array<{ url: string; init?: FetchInit }> = [];
  const h = pageHarness({ respond: async (url, init) => { posts.push({ url, init }); return stream.reply; } });
  h.reader.prompts[0].takes = ["first.wav", "second.wav"];
  enableComparison(h.reader);
  const running = h.reader.comparison.run();
  await settle();
  const posted = JSON.parse(posts[0]!.init?.body as string);
  assert.deepEqual(posted.source, { kind: "saved", takeName: "second.wav" });
  assert.deepEqual(posted.setupIds, ["mlx-parakeet/v3-greedy"]);
  assert.equal(h.pending.length, 0);
  assert.equal(h.played.length, 0);
  h.reader.stepTake();
  assert.equal(posts[0]!.init?.signal?.aborted, true);
  stream.end();
  await running;
  assert.equal(h.reader.savedTake().name, "first.wav");
});

test("cancel retains completed peers; abrupt EOF does not report all success", async () => {
  const stream = streamReply();
  const h = pageHarness({ respond: async () => stream.reply });
  const ids = ["mlx-parakeet/v3-greedy", "mlx-parakeet/ultra-greedy"];
  enableComparison(h.reader, ids);
  await readyTake(h);
  const job = h.reader.comparison.job;
  acceptStream(stream, job.requestId);
  stream.send(rowEvent(job.requestId, ids[0]!));
  await settle();
  h.reader.comparison.cancel();
  assert.equal(h.reader.comparison.rows[0].result.text, "raw words\n");
  assert.equal(h.reader.comparison.rows[1].phase, "cancelled");
  stream.end();
  await settle();
  assert.equal(h.reader.comparison.rows[1].phase, "cancelled");

  const incomplete = streamReply();
  const second = pageHarness({ respond: async () => incomplete.reply });
  enableComparison(second.reader);
  await readyTake(second);
  acceptStream(incomplete, second.reader.comparison.job.requestId);
  incomplete.end();
  await settle();
  assert.match(second.reader.comparison.message, /incomplete/);
});

test("native controls retain Space, Enter, arrows and letters without reader shortcuts", () => {
  const { reader, dispatch } = pageHarness();
  let actions = 0;
  reader.press = reader.save = reader.play = reader.stepTake = reader.move = () => { actions++; };
  for (const key of [" ", "Enter", "ArrowLeft", "ArrowRight", "p", "t"]) {
    const target = { closest: () => ({}) };
    dispatch("window:keydown", { key, code: key === " " ? "Space" : key, target, preventDefault: () => { actions++; } });
    dispatch("window:keyup", { key, code: key === " " ? "Space" : key, target, preventDefault: () => { actions++; } });
  }
  assert.equal(actions, 0);
});

test("only a reader owned Space keyup releases recording", () => {
  const { reader, dispatch } = pageHarness();
  let releases = 0;
  reader.press = () => { reader.gate.press(); };
  reader.gate.release = () => { releases++; };
  const background = { closest: () => null };
  const event = { key: " ", code: "Space", target: background, preventDefault() {} };
  dispatch("window:keyup", event);
  assert.equal(releases, 0);
  dispatch("window:keydown", event);
  dispatch("window:keyup", { ...event, target: { closest: () => ({}) } });
  assert.equal(releases, 1);
});

test("API candidates follow selectable metadata; selection never uploads, explicit action sends one live take, saved stays denied", async () => {
  const requests: Array<{ url: string; init?: FetchInit }> = [];
  const h = pageHarness({ respond: async (url, init) => {
    requests.push({ url, init });
    if (url === "/speech-providers/session") return okReply({ sessionId: "server-assigned" });
    if (url === "/speech-providers/selection") return okReply({ catalogueRevision: "provider-revision", selectionRevision: 3 });
    return okReply({ config: { schema: 1, provider: { kind: "cartesia", model: "configured-model" } }, result: { text: "raw API text\n", timings: { providerRequestWallMs: { status: "measured", ms: 55, clock: "server", boundary: "provider request" }, providerProcessingMs: { status: "unavailable", reason: "not exposed" }, providerNetworkMs: { status: "unavailable", reason: "not exposed" } } } });
  } });
  enableComparison(h.reader);
  h.reader.comparison.providers = { revision: "provider-revision", input: [
    { id: "api-ready", label: "Configured API", selectable: true, config: { schema: 1, provider: { kind: "cartesia", model: "configured-model" } } },
    { id: "api-hidden", label: "Unconfigured API", selectable: false, config: { schema: 1, provider: { kind: "openai", model: "missing" } } },
  ] };
  h.reader.comparison.renderProviders();
  const provider = h.reader.comparison.elements["input-provider"];
  assert.deepEqual(provider.children.map((option: any) => option.value), ["local", "cartesia"]);
  provider.value = "cartesia";
  h.dispatch("input-provider:change");
  const take = await readyTake(h);
  assert.equal(requests.length, 0, "provider selection and a completed API capture require explicit Transcribe take");
  await h.reader.comparison.run();
  assert.equal(requests.length, 3);
  const session = requests[0]!.init!;
  assert.deepEqual(JSON.parse(session.body as string), {});
  const selection = JSON.parse(requests[1]!.init!.body as string);
  assert.equal(selection.direction, "input");
  assert.equal(selection.config.provider.kind, "cartesia");
  assert.equal(requests[2]!.init!.body, take.samples.buffer);
  assert.equal(h.reader.comparison.rows[0].result.text, "raw API text\n");
  assert.equal(h.reader.comparison.rows[0].view.text.children.map((node: any) => node.textContent).join(""), "raw API text\n", "highlighting preserves the raw whitespace");
  assert.match(h.reader.comparison.rows[0].view.timing.textContent, /Provider processing: Unavailable/);
  h.reader.take = null;
  h.reader.prompts[0].takes = ["saved.wav"];
  await h.reader.comparison.run();
  assert.equal(requests.length, 3, "saved material never reaches API dispatch");
  assert.match(h.reader.comparison.message, /not authorized/);
});

for (const status of ["complete", "cancelled"]) test(`empty/partial output and ${status} comparison termination retain their meanings`, async () => {
  const stream = streamReply();
  const h = pageHarness({ respond: async () => stream.reply });
  enableComparison(h.reader, ["mlx-parakeet/v3-greedy", "mlx-cleanup/qwen-targeted"]);
  await readyTake(h);
  const job = h.reader.comparison.job;
  acceptStream(stream, job.requestId);
  stream.send(rowEvent(job.requestId, "mlx-parakeet/v3-greedy", "complete", ""));
  stream.send(rowEvent(job.requestId, "mlx-cleanup/qwen-targeted", "complete", ""));
  stream.send({ type: "done", requestId: job.requestId, comparisonId: "server-job", status, wallMs: 100 });
  await settle();
  assert.equal(h.reader.comparison.rows[0].view.text.textContent, "No speech detected");
  assert.equal(h.reader.comparison.rows[1].view.text.textContent, "Empty output");
  assert.equal(h.reader.comparison.message, status === "cancelled" ? "Comparison cancelled" : "Comparison finished");
  assert.equal(h.reader.comparison.job, null, "terminal event settles the job without waiting for EOF");
  const failed = rowEvent("next", "mlx-cleanup/qwen-targeted", "error", null);
  (failed.row as any).error = { code: "execution", message: "cleanup failed", rawText: "  partial\n" };
  const next = { requestId: "next", comparisonId: "server-job", controller: new AbortController() };
  h.reader.comparison.job = next;
  h.reader.comparison.rows[1].result = null;
  h.reader.comparison.accept(failed, next);
  assert.equal(h.reader.comparison.rows[1].view.phase.textContent, "Incomplete output");
  assert.equal(h.reader.comparison.rows[1].view.text.textContent, "  partial\n");
  h.reader.comparison.cancel();
});
