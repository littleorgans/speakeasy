import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { setImmediate, setTimeout } from "node:timers/promises";
import { test } from "node:test";
import { ProviderError } from "@speakeasy/speech-io/providers";
import { OPENAI_SPEED } from "@speakeasy/speech-io";
import {
  DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_OUTPUT_CONFIG, OPENAI_SAMPLE_RATE,
  OpenAITextToSpeech, type OpenAIAudioSegment, type OpenAILimits,
  type OpenAIOptions, type OpenAISocket, type OpenAITTSSession,
} from "./openai.ts";
import { planSegments } from "./stream.ts";

type WireEvent = Record<string, any>;
const PCM = Buffer.from([0, 128, 255, 127, 0, 64, 0, 192]).toString("base64");
const FAKE_KEY = "fake-openai-test-key";

class FakeSocket extends EventEmitter implements OpenAISocket {
  sent: WireEvent[] = [];
  closes = 0;
  onSend?: (event: WireEvent) => void;
  send(data: string): void {
    const event = JSON.parse(data);
    this.sent.push(event);
    this.onSend?.(event);
  }
  close(): void { this.closes++; }
  event(event: WireEvent): void { this.emit("message", Buffer.from(JSON.stringify(event))); }
  created(): void { this.event({ type: "session.created", session: { type: "realtime" } }); }
  updated(overrides: WireEvent = {}): void {
    const update = this.sent.find((event) => event.type === "session.update")!;
    this.event({ type: "session.updated", session: { ...update.session, model: DEFAULT_OPENAI_MODEL, ...overrides } });
  }
  request(): WireEvent { return this.sent.filter((event) => event.type === "response.create").at(-1)!; }
  begin(id = "response-1", metadata = this.request().response.metadata): void {
    this.event({ type: "response.created", response: { id, metadata } });
  }
  part(type: string, values: WireEvent = {}, id = "response-1"): void {
    this.event({ type, response_id: id, item_id: "item-1", content_index: 0, output_index: 0, ...values });
  }
  audio(delta = PCM, id = "response-1"): void { this.part("response.output_audio.delta", { delta }, id); }
  done(status: unknown = "completed", id = "response-1"): void {
    this.event({ type: "response.done", response: { id, status } });
  }
  respond(id = "response-1"): void {
    this.begin(id);
    this.audio(PCM, id);
    this.part("response.output_audio_transcript.done", {
      transcript: this.request().response.input[0].content[0].text,
    }, id);
    this.done("completed", id);
  }
}

function harness(options: Partial<OpenAIOptions> = {}) {
  const socket = new FakeSocket();
  let keys = 0;
  let connections = 0;
  let time = 0;
  const tts = new OpenAITextToSpeech({
    apiKey: () => { keys++; return FAKE_KEY; },
    createSocket: (url, key) => {
      connections++;
      assert.equal(new URL(url).searchParams.get("model"), DEFAULT_OPENAI_MODEL);
      assert.equal(key, FAKE_KEY);
      return socket;
    },
    now: () => time,
    ...options,
  });
  async function open(config = {}, signal?: AbortSignal): Promise<OpenAITTSSession> {
    const opening = tts.open(config, { signal });
    await setImmediate();
    socket.emit("open");
    socket.created();
    time = 5;
    socket.updated();
    return opening;
  }
  return { socket, tts, open, keys: () => keys, connections: () => connections, time: (value: number) => { time = value; } };
}

async function collect(stream: AsyncIterable<OpenAIAudioSegment>): Promise<OpenAIAudioSegment[]> {
  const segments: OpenAIAudioSegment[] = [];
  for await (const segment of stream) segments.push(segment);
  return segments;
}

test("construction and catalogue are inert; setup waits for created then accepted update", async () => {
  const h = harness();
  assert.equal(h.keys(), 0);
  assert.equal(h.connections(), 0);
  assert.equal(DEFAULT_OPENAI_OUTPUT_CONFIG.voice, "marin");
  let ready = false;
  const opening = h.tts.open().then((session) => { ready = true; return session; });
  h.socket.emit("open");
  assert.equal(h.socket.sent.length, 0);
  await setImmediate();
  assert.equal(ready, false);
  h.socket.created();
  const update = h.socket.sent[0]!;
  assert.equal(update.type, "session.update");
  assert.deepEqual(update.session.audio.output, {
    format: { type: "audio/pcm", rate: 24000 }, voice: "marin", speed: 1,
  });
  assert.deepEqual(update.session.output_modalities, ["audio"]);
  await setImmediate();
  assert.equal(ready, false);
  h.socket.updated();
  await (await opening).close();
});

test("buffered audio, returned transcript and measured timings stay separate", async () => {
  const h = harness();
  const session = await h.open({ voice: "cedar", speed: 0.75 });
  const stream = session.speak("Hello there.");
  let yielded = false;
  const first = stream.next().then((value) => { yielded = true; return value; });
  await setImmediate();
  const response = h.socket.request().response;
  assert.equal(response.conversation, "none");
  assert.deepEqual(response.input, [{ type: "message", role: "user", content: [{ type: "input_text", text: "Hello there." }] }]);
  assert.deepEqual(response.output_modalities, ["audio"]);
  assert.match(response.instructions, /aloud without adding commentary/);
  h.socket.begin();
  h.time(12);
  h.socket.audio();
  h.socket.part("response.output_audio_transcript.delta", { delta: "Other " });
  h.socket.part("response.output_audio_transcript.done", { transcript: "Other words." });
  h.socket.part("response.output_audio.done");
  await setImmediate();
  assert.equal(yielded, false);
  h.time(30);
  h.socket.done();
  const segment = (await first).value!;
  assert.equal(segment.sampleRate, OPENAI_SAMPLE_RATE);
  assert.deepEqual([...segment.samples.slice(0, 4)], [-1, 32767 / 32768, 0.5, -0.5]);
  assert.equal(segment.sentence, "Hello there.");
  assert.equal(segment.requestedText, "Hello there.");
  assert.equal(segment.returnedTranscript, "Other words.");
  assert.equal(segment.transcriptDiverged, true);
  assert.equal(segment.voice, "cedar");
  assert.equal(segment.model, DEFAULT_OPENAI_MODEL);
  assert.equal(segment.synthMs, 25);
  assert.equal(segment.readyAtMs, 25);
  assert.equal(segment.providerFirstAudioMs.status, "measured");
  assert.equal(segment.providerFirstAudioMs.status === "measured" && segment.providerFirstAudioMs.ms, 7);
  assert.equal(segment.providerRequestWallMs.status === "measured" && segment.providerRequestWallMs.ms, 25);
  assert.equal(segment.sessionSetupWallMs.status === "measured" && segment.sessionSetupWallMs.ms, 5);
  assert.equal(segment.providerProcessingMs.status, "unavailable");
  assert.equal(segment.providerNetworkMs.status, "unavailable");
  assert.equal((await stream.next()).done, true);
  await session.close();
});

test("fixed text and incremental text reuse the same sentence planner with isolated responses", async () => {
  const text = "Hello, the build finished today. Please wait.";
  for (const incremental of [false, true]) {
    const h = harness();
    const session = await h.open();
    let id = 0;
    h.socket.onSend = (event) => {
      if (event.type === "response.create") h.socket.respond(`r-${++id}`);
    };
    async function* tokens() { for (const token of text.split(/(?<=\s)/)) yield token; }
    const result = await collect(session.speak(incremental ? tokens() : text));
    assert.deepEqual(result.map((segment) => segment.sentence), planSegments(text));
    assert.ok(result.every((segment) => segment.transcriptDiverged === false));
    const requests = h.socket.sent.filter((event) => event.type === "response.create");
    assert.equal(new Set(requests.map((event) => event.response.metadata.speech_segment)).size, result.length);
    assert.ok(requests.every((event) => event.response.conversation === "none" && event.response.input.length === 1));
    assert.equal(h.socket.sent.some((event) => event.type === "conversation.item.create"), false);
    await session.close();
  }
});

test("late response IDs and wrong creation metadata cannot contaminate a sentence", async () => {
  const h = harness();
  const session = await h.open();
  const result = collect(session.speak("Hello."));
  await setImmediate();
  h.socket.begin("old", { speech_segment: "old" });
  h.socket.audio("not-base64", "old");
  h.socket.done("completed", "old");
  h.socket.begin("current");
  h.socket.audio("not-base64", "old");
  h.socket.part("response.output_audio_transcript.done", { transcript: "wrong" }, "old");
  h.socket.done("failed", "old");
  h.socket.audio(PCM, "current");
  h.socket.done("completed", "current");
  const [segment] = await result;
  assert.equal(segment!.returnedTranscript, null);
  assert.equal(segment!.transcriptDiverged, null);
  await session.close();
});

test("completed response output retains the authoritative transcript with matching item identity", async () => {
  const h = harness();
  const session = await h.open();
  const result = collect(session.speak("Hello."));
  await setImmediate();
  h.socket.begin();
  h.socket.part("response.output_audio.delta", { delta: PCM, output_index: 1, content_index: 1 });
  h.socket.part("response.output_audio_transcript.delta", { delta: "partial", output_index: 1, content_index: 1 });
  h.socket.event({ type: "response.done", response: {
    id: "response-1", status: "completed", output: [{}, {
      id: "item-1", content: [{}, { type: "output_audio", transcript: "Hello." }],
    }],
  } });
  const [segment] = await result;
  assert.equal(segment!.returnedTranscript, "Hello.");
  assert.equal(segment!.transcriptDiverged, false);
  await session.close();
});

test("terminal completion is required even after audio.done", async () => {
  for (const status of ["failed", "incomplete", "cancelled", undefined]) {
    const h = harness();
    const session = await h.open();
    const result = collect(session.speak("Hello."));
    const rejection = assert.rejects(result, /Speech provider request failed/);
    await setImmediate();
    h.socket.begin();
    h.socket.audio();
    h.socket.part("response.output_audio.done");
    h.socket.event({ type: "response.done", response: { id: "response-1", status } });
    await rejection;
    assert.equal(h.socket.closes, 1);
    await session.close();
  }
});

test("malformed audio, identity, transcript and event limits fail closed", async () => {
  const cases: Array<{ limits?: Partial<OpenAILimits>; emit: (socket: FakeSocket) => void }> = [
    { emit: (s) => s.audio("!AAA") },
    { emit: (s) => s.audio("AA==") },
    { emit: (s) => s.audio("") },
    { emit: (s) => s.emit("message", "{broken") },
    { emit: (s) => s.event({}) },
    { emit: (s) => s.part("response.output_audio.delta", { delta: PCM, content_index: -1 }) },
    { emit: (s) => { s.audio(); s.part("response.output_audio.delta", { delta: PCM, item_id: "other" }); } },
    { emit: (s) => { s.audio(); s.part("response.output_audio.delta", { delta: PCM, content_index: 1 }); } },
    { emit: (s) => { s.part("response.output_audio.done"); s.audio(); } },
    { emit: (s) => s.part("response.output_audio_transcript.done", { transcript: 42 }) },
    { limits: { maxAudioBytes: 4 }, emit: (s) => s.audio() },
    { limits: { maxAudioBytes: 12 }, emit: (s) => { s.audio(); s.audio(); } },
    { limits: { maxTranscriptChars: 4 }, emit: (s) => s.part("response.output_audio_transcript.delta", { delta: "12345" }) },
    { limits: { maxEventBytes: 1024 }, emit: (s) => s.emit("message", Buffer.alloc(1025, 32)) },
    { limits: { maxEvents: 3 }, emit: (s) => { s.event({ type: "ignored" }); s.event({ type: "ignored" }); s.event({ type: "ignored" }); } },
    { emit: (s) => s.done() },
  ];
  for (const entry of cases) {
    const h = harness({ limits: entry.limits });
    const session = await h.open();
    const result = collect(session.speak("Hello."));
    const rejection = assert.rejects(result, /Speech provider request failed/);
    await setImmediate();
    h.socket.begin();
    entry.emit(h.socket);
    await rejection;
    assert.equal(h.socket.closes, 1);
    await session.close();
  }
});

test("close and abort reject pending audio and block all later dispatch", async () => {
  for (const abort of [false, true]) {
    const h = harness();
    const controller = new AbortController();
    const session = await h.open({}, controller.signal);
    const result = collect(session.speak("Hello. Next. Last."));
    const rejection = assert.rejects(result, /Speech provider request failed/);
    await setImmediate();
    h.socket.begin();
    h.socket.audio();
    if (abort) controller.abort(); else await session.close();
    h.socket.done();
    await rejection;
    await assert.rejects(collect(session.speak("Again.")), /Speech provider request failed/);
    assert.equal(h.socket.sent.filter((event) => event.type === "response.create").length, 1);
    assert.deepEqual(h.socket.sent.at(-1), { type: "response.cancel", response_id: "response-1" });
    assert.equal(h.socket.closes, 1);
  }
});

test("consumer demand gates remote sentences before iterator return or close", async (t) => {
  for (const stop of ["return", "close"] as const) {
    const h = harness();
    const session = await h.open();
    t.after(() => session.close());
    let requests = 0;
    h.socket.onSend = (event) => {
      if (event.type === "response.create" && ++requests === 1) h.socket.respond();
    };
    const stream = session.speak("Hello. Next. Last.");
    assert.equal((await stream.next()).value!.sentence, "Hello.");
    await setImmediate();
    assert.equal(requests, 1);
    if (stop === "return") await stream.return(); else await session.close();
    await setImmediate();
    assert.equal(h.socket.closes, 1);
    assert.equal(requests, 1);
    await session.close();
  }
});

test("closing while the text producer is stalled rejects the pull and suppresses late text", async () => {
  const h = harness();
  const session = await h.open();
  let release!: (result: IteratorResult<string>) => void;
  const tokens: AsyncIterable<string> = {
    [Symbol.asyncIterator]: () => ({
      next: () => new Promise((resolve) => { release = resolve; }),
      return: async () => ({ done: true, value: undefined }),
    }),
  };
  const pending = collect(session.speak(tokens));
  const settled = pending.then(() => true, () => true);
  await setImmediate();
  await session.close();
  const completed = await Promise.race([settled, setTimeout(25, false)]);
  // Release the fake producer even if this regression assertion fails.
  release({ done: false, value: "Late sentence." });
  await assert.rejects(pending, /Speech provider request failed/);
  assert.equal(completed, true);
  assert.equal(h.socket.sent.some((event) => event.type === "response.create"), false);
});

test("iterator return during a pending response cancels without waiting for provider completion", async () => {
  const h = harness();
  const session = await h.open();
  const stream = session.speak("Hello. Next.");
  const pending = stream.next();
  const rejection = assert.rejects(pending, /Speech provider request failed/);
  await setImmediate();
  h.socket.begin();
  assert.equal((await stream.return()).done, true);
  await rejection;
  assert.equal(h.socket.closes, 1);
  assert.equal(h.socket.sent.filter((event) => event.type === "response.create").length, 1);
});

test("selection is checked again before subsequent sentences; raw policy errors stay private", async () => {
  let allowed = true;
  let checks = 0;
  const h = harness({ beforeRequest: () => { checks++; if (!allowed) throw new Error(FAKE_KEY); } });
  const session = await h.open();
  const result = collect(session.speak("Hello. Next."));
  const rejection = assert.rejects(result, (error: Error) => {
    assert.equal(error.message, "Speech provider request failed.");
    assert.doesNotMatch(error.stack!, /fake-openai-test-key/);
    return true;
  });
  await setImmediate();
  h.socket.begin();
  h.socket.audio();
  allowed = false;
  h.socket.done();
  await rejection;
  assert.equal(checks, 3);
  assert.equal(h.socket.sent.filter((event) => event.type === "response.create").length, 1);
  await session.close();
});

test("async authorization is awaited and closing it prevents later dispatch", async () => {
  let checks = 0;
  let release!: () => void;
  const h = harness({ beforeRequest: () => {
    if (++checks === 2) return new Promise<void>((resolve) => { release = resolve; });
    return undefined;
  } });
  const session = await h.open();
  const pending = collect(session.speak("Hello."));
  const rejection = assert.rejects(pending, /Speech provider request failed/);
  await setImmediate();
  assert.equal(checks, 2);
  assert.equal(h.socket.sent.some((event) => event.type === "response.create"), false);
  await session.close();
  release();
  await rejection;
  assert.equal(h.socket.sent.some((event) => event.type === "response.create"), false);
});

test("transport and provider failures expose only safe static errors", async () => {
  for (const mode of ["socket-error", "socket-close", "provider", "send"]) {
    const h = harness();
    const session = await h.open();
    if (mode === "send") h.socket.onSend = () => { throw new Error(FAKE_KEY); };
    const result = collect(session.speak("Hello."));
    const rejection = assert.rejects(result, (error: Error) => {
      assert.equal(error.message, "Speech provider request failed.");
      assert.equal(error.cause, undefined);
      assert.doesNotMatch(error.stack!, /fake-openai-test-key/);
      return true;
    });
    await setImmediate();
    if (mode === "socket-error") h.socket.emit("error", new Error(FAKE_KEY));
    if (mode === "socket-close") h.socket.emit("close");
    if (mode === "provider") h.socket.event({ type: "error", error: { message: FAKE_KEY } });
    await rejection;
    await session.close();
  }
});

test("session and response waits time out without retry", async () => {
  const h = harness({ limits: { sessionTimeoutMs: 10 } });
  await assert.rejects(h.tts.open(), /Speech provider request failed/);
  assert.equal(h.connections(), 1);
  assert.equal(h.socket.closes, 1);
  const r = harness({ limits: { responseTimeoutMs: 10 } });
  const session = await r.open();
  await assert.rejects(collect(session.speak("Hello.")), /Speech provider request failed/);
  assert.equal(r.socket.sent.filter((event) => event.type === "response.create").length, 1);
  await session.close();
});

test("configuration and accepted format are validated before yielding", async () => {
  const h = harness();
  for (const config of [
    { model: "tts-1" }, { voice: 1 }, { voice: "unknown" }, { speed: NaN },
    { speed: 0.2 }, { speed: 1.6 },
    { speed: OPENAI_SPEED.min - 0.001 }, { speed: OPENAI_SPEED.max + 0.001 },
  ]) {
    await assert.rejects(h.tts.open(config), /Speech configuration is unsupported/);
  }
  assert.equal(h.connections(), 0);
  for (const speed of [OPENAI_SPEED.min, OPENAI_SPEED.max]) {
    const boundary = harness();
    const session = await boundary.open({ speed });
    assert.equal(boundary.socket.sent[0]!.session.audio.output.speed, speed);
    await session.close();
  }
  const opening = h.tts.open();
  const rejection = assert.rejects(opening, /Speech provider request failed/);
  h.socket.created();
  h.socket.updated({ audio: { output: { format: { type: "audio/pcm", rate: 16000 }, voice: "marin", speed: 1 } } });
  await rejection;
  assert.equal(h.socket.closes, 1);
});

test("missing, failing or preaborted credential paths never create a socket", async () => {
  for (const [getter, code] of [
    [() => undefined, "PROVIDER_UNAVAILABLE"],
    [() => { throw new Error(FAKE_KEY); }, "PROVIDER_FAILED"],
  ] as const) {
    const h = harness({ apiKey: getter });
    await assert.rejects(h.tts.open(), { code });
    assert.equal(h.connections(), 0);
  }
  const h = harness();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(h.tts.open({}, { signal: controller.signal }), /aborted/);
  assert.equal(h.keys(), 0);
  assert.equal(h.connections(), 0);
});

test("trusted policy codes survive before connection and before each demanded sentence", async (t) => {
  for (const asynchronous of [false, true]) {
    let allowed = false;
    let checks = 0;
    const policy = () => {
      checks++;
      if (!allowed) throw new ProviderError("SELECTION_STALE");
    };
    const h = harness({ beforeRequest: asynchronous ? async () => policy() : policy });
    // Exercise the abortable open path as well as the sentence wait path.
    const controller = new AbortController();
    await assert.rejects(h.tts.open({}, { signal: controller.signal }), { code: "SELECTION_STALE" });
    assert.equal(h.keys(), 0);
    assert.equal(h.connections(), 0);
    allowed = true;
    const session = await h.open();
    t.after(() => session.close());
    h.socket.onSend = (event) => { if (event.type === "response.create") h.socket.respond(); };
    const stream = session.speak("First. Second.");
    assert.equal((await stream.next()).value!.sentence, "First.");
    await setImmediate();
    assert.equal(checks, 3);
    allowed = false;
    await assert.rejects(stream.next(), { code: "SELECTION_STALE" });
    assert.equal(checks, 4);
    assert.equal(h.socket.sent.filter((event) => event.type === "response.create").length, 1);
  }
});

test("unsupported configuration and absent credentials use shared safe codes", async () => {
  const h = harness({ apiKey: () => undefined });
  await assert.rejects(h.tts.open({ voice: "unknown" }), { code: "CONFIG_UNSUPPORTED" });
  await assert.rejects(h.tts.open(), { code: "PROVIDER_UNAVAILABLE" });
  assert.equal(h.connections(), 0);
});
