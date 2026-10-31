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

type FetchInit = { method?: string; body?: ArrayBuffer };
type FetchReply = { ok: boolean; status?: number; json?(): Promise<unknown>; arrayBuffer?(): Promise<ArrayBuffer> };

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
  const listeners = new Map<string, () => void>();
  const listen = (target: string) => (type: string, listener: () => void) => listeners.set(`${target}:${type}`, listener);
  const document = {
    body: { dataset: {} },
    hidden: false,
    getElementById(id: string) {
      if (!elements.has(id)) {
        elements.set(id, { replaceChildren(...nodes: unknown[]) { this.children = nodes; }, addEventListener() {} });
      }
      return elements.get(id);
    },
    createElement: () => ({ textContent: "", className: "" }),
    addEventListener: listen("document"),
  };
  /** Fire a page lifecycle event the reader listens for, such as `document:visibilitychange`. */
  const dispatch = (key: string) => listeners.get(key)!();
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
    Float32Array, URLSearchParams,
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
  vm.runInContext(readFileSync(new URL("../public/capture-constraints.js", import.meta.url), "utf8").replace("export ", ""), context);
  vm.runInContext(source, context);
  const reader = new context.Reader();
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
