import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import { createConversationRuntime, type ConversationRuntime, type RuntimeConfig } from "@speakeasy/convo-engine";
import { MercuryChatModel } from "../../llm/src/mercury.ts";
import { LocalSpeechOutputFactory, SpeechOutputFactory } from "@speakeasy/speech-io";
import { WebSocket } from "ws";
import { startBrowserVoiceServer } from "./host.ts";
import { runtimeConfigFor, safeMessage } from "./session.ts";
import { createHostProviderConfig } from "./setup-routes.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { buildSpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_MODELS, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";

test("host serves the room shell and reports health", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  try {
    const health = await fetch(`${host.url}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { ok: true });
    const page = await fetch(host.url);
    assert.equal(page.status, 200);
    assert.match(
      page.headers.get("content-security-policy") ?? "",
      /frame-ancestors 'none'/,
    );
    assert.match(await page.text(), /Speak Easy · Live room/);
  } finally {
    await host.close();
  }
});

test("browser errors redact provider credentials", () => {
  assert.equal(
    safeMessage(new Error("upstream rejected Bearer sk-proj-exampleSecret123")),
    "upstream rejected Bearer [redacted]",
  );
  assert.equal(
    safeMessage("provider csk-exampleSecret123 failed"),
    "provider [redacted] failed",
  );
});

test("host rejects browser sockets from another origin", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice", {
    origin: "https://example.com",
  });
  try {
    const error = await new Promise<Error>((resolve) => socket.once("error", resolve));
    assert.match(error.message, /403/);
  } finally {
    socket.close();
    await host.close();
  }
});

test("browser socket drives one complete transcript and acknowledged playback", async () => {
  const captured: {
    runtime?: RuntimeConfig;
    endpoint?: { mode?: string; minTrailingSilenceMs?: number };
  } = {};
  const host = await startBrowserVoiceServer({
    port: 0,
    createRuntime: async (config) => {
      captured.runtime = config;
      return fakeRuntime((endpoint) => {
        captured.endpoint = endpoint;
      });
    },
  });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const events: Array<Record<string, unknown>> = [];
  const audio: Buffer[] = [];
  socket.on("message", (data, isBinary) => {
    if (isBinary) {
      audio.push(Buffer.from(data as Buffer));
      return;
    }
    const event = JSON.parse(data.toString()) as Record<string, unknown>;
    events.push(event);
    if (event.type === "playback" && event.action === "end") {
      socket.send(
        JSON.stringify({ type: "playback-drained", playbackId: event.playbackId }),
      );
    }
  });

  try {
    await onceOpen(socket);
    socket.send(JSON.stringify({
      type: "start",
      engine: "realtime",
      mode: "hold",
      pauseMs: 700,
      voice: "cedar",
      barge: true,
    }));
    await waitFor(() => events.some((event) => event.type === "session" && event.phase === "ready"));
    socket.send(new Float32Array(320).fill(0.2));
    await waitFor(() => events.some((event) => event.type === "transcript"));
    assert.ok(!events.some((event) => event.type === "metrics"));
    socket.send(JSON.stringify({ type: "commit-input" }));
    await waitFor(() => events.some((event) => event.type === "metrics"));

    const transcripts = events.filter((event) => event.type === "transcript");
    assert.ok(transcripts.some((event) => event.role === "user" && event.text === "hello browser"));
    assert.ok(transcripts.some((event) => event.role === "assistant" && event.text === "Hello back." && event.final === true));
    assert.equal(audio.length, 1);
    assert.ok(events.some((event) => event.type === "playback" && event.action === "start"));
    assert.ok(events.some((event) => event.type === "playback" && event.action === "end"));
    assert.equal(captured.runtime?.voice, "cedar");
    assert.deepEqual(captured.endpoint, { mode: "manual" });
  } finally {
    socket.close();
    await onceClose(socket);
    await host.close();
  }
});

test("natural mode maps the configured pause to eager endpointing", async () => {
  const captured: { endpoint?: unknown } = {};
  const host = await startBrowserVoiceServer({
    port: 0,
    createRuntime: async () => fakeRuntime((endpoint) => {
      captured.endpoint = endpoint;
    }),
  });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice");
  try {
    await onceOpen(socket);
    socket.send(JSON.stringify({
      type: "start",
      engine: "realtime",
      mode: "natural",
      pauseMs: 950,
      voice: "marin",
      barge: true,
    }));
    await waitFor(() => captured.endpoint !== undefined);
    assert.deepEqual(captured.endpoint, {
      mode: "eager",
      minTrailingSilenceMs: 950,
    });
  } finally {
    socket.close();
    await onceClose(socket);
    await host.close();
  }
});

test("browser engine selection maps to isolated responder configurations", () => {
  const common = {
    type: "start" as const,
    mode: "natural" as const,
    pauseMs: 700,
    voice: "cedar" as const,
    barge: true,
  };
  assert.deepEqual(runtimeConfigFor(
    { ...common, engine: "realtime" },
    {
      llmModel: "operator-model",
      ttsEngine: "cartesia",
      voice: "cascade-voice",
    },
  ), {
    llmModel: "operator-model",
    responder: "realtime",
    ttsEngine: "cartesia",
    voice: "cedar",
  });
  assert.deepEqual(runtimeConfigFor(
    { ...common, engine: "mercury-instant" },
    {
      llmModel: "operator-model",
      ttsEngine: "cartesia",
      voice: "cascade-voice",
    },
  ), {
    llmModel: "operator-model",
    responder: "cascade",
    llmProvider: "mercury",
    llmReasoningEffort: "instant",
    ttsEngine: "cartesia",
    voice: "cascade-voice",
  });
  assert.equal(
    runtimeConfigFor({ ...common, engine: "mercury-low" }, {})
      .llmReasoningEffort,
    "low",
  );
});

test("Mercury snapshots optional output while Realtime retains its distinct voice", () => {
  const command = { type: "start" as const, mode: "natural" as const, pauseMs: 700,
    voice: "cedar" as const, barge: false, speechOutput: { ...DEFAULT_LOCAL_OUTPUT } };
  const mercury = runtimeConfigFor({ ...command, engine: "mercury-low" }, {});
  command.speechOutput.voice = "changed after start";
  assert.equal(mercury.speechOutput?.voice, "kristin");
  const realtime = runtimeConfigFor({ ...command, engine: "realtime" }, { speechOutput: DEFAULT_LOCAL_OUTPUT });
  assert.equal(realtime.voice, "cedar");
  assert.equal("speechOutput" in realtime, false);
});

function fakeOutput() {
  const catalogue = buildSpeechOutputCatalogue({ localRegistry: LOCAL_OUTPUT_REGISTRY, local: { default: DEFAULT_LOCAL_OUTPUT,
    models: LOCAL_OUTPUT_MODELS.map((model) => ({ ...model, readiness: { state: "ready" as const, reason: null } })) } });
  const providers = createHostProviderConfig({ credentials: createProviderCredentials(() => ({})),
    readiness: () => [], inputAdapters: [], outputCatalogue: () => catalogue });
  let opens = 0;
  return { providers, catalogue, get opens() { return opens; }, speechOutput: {
    catalogue: () => catalogue,
    quiesce: async () => {},
    createOutput: () => ({ open: async () => {
      opens++;
      return { async *speak() { yield { index: 0, sentence: "hello", samples: new Float32Array([0.1]),
        sampleRate: 22_050, readyAtMs: 1, synthMs: 1, audioDurationMs: 1 }; }, async close() {} };
    } }),
  } };
}

function observe(socket: WebSocket) {
  const events: Array<Record<string, any>> = [];
  socket.on("message", (data, binary) => { if (!binary) events.push(JSON.parse(data.toString())); });
  return events;
}

function roomStart(socket: WebSocket) {
  socket.send(JSON.stringify({ type: "start", engine: "realtime", mode: "hold", pauseMs: 700, voice: "cedar", barge: false }));
}

test("host models and select-output use the combined policy revision while rejecting stale standalone revisions", async () => {
  const output = fakeOutput();
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers, speechOutput: output.speechOutput });
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output");
  const observed = observe(trial);
  try {
    const response = await fetch(host.url + "/speech-output/models");
    const models = await response.json() as { revision: string };
    const combined = output.providers.catalogue().revision;
    assert.notEqual(output.catalogue.revision, combined, "actual standalone and combined snapshots have different hashes");
    assert.equal(models.revision, combined);
    await onceOpen(trial);
    trial.send(JSON.stringify({ type: "select-output", requestId: "select", catalogueRevision: models.revision,
      selection: DEFAULT_LOCAL_OUTPUT }));
    await waitFor(() => observed.some((event) => event.requestId === "select"));
    const selected = observed.find((event) => event.requestId === "select")!;
    assert.equal(selected.type, "output-selected");
    assert.equal(selected.catalogueRevision, models.revision);
    trial.send(JSON.stringify({ type: "speak-sample", requestId: "speak", catalogueRevision: models.revision,
      selectionRevision: selected.selectionRevision, selection: DEFAULT_LOCAL_OUTPUT, text: "hello" }));
    await waitFor(() => observed.some((event) => event.outcome === "produced"));
    trial.send(JSON.stringify({ type: "select-output", requestId: "stale", catalogueRevision: output.catalogue.revision,
      selection: DEFAULT_LOCAL_OUTPUT }));
    await waitFor(() => observed.some((event) => event.requestId === "stale"));
    assert.equal(observed.find((event) => event.requestId === "stale")!.code, "SELECTION_STALE");
    assert.equal(output.opens, 1);
  } finally { trial.close(); await host.close(); }
});

test("models GET is metadata only; idle room permits tryout, and active work excludes room and tryout", async () => {
  const output = fakeOutput();
  let runtimes = 0;
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers, speechOutput: output.speechOutput,
    createRuntime: async () => { runtimes++; return fakeRuntime(); } });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output", { origin: host.url });
  const roomEvents = observe(room);
  const trialEvents = observe(trial);
  const speak = (requestId: string) => trial.send(JSON.stringify({ type: "speak-sample", requestId,
    catalogueRevision: output.providers.catalogue().revision, selection: DEFAULT_LOCAL_OUTPUT, text: "hello" }));
  try {
    assert.equal((await fetch(host.url + "/speech-output/models")).status, 200);
    assert.equal(output.opens, 0);
    await Promise.all([onceOpen(room), onceOpen(trial)]);
    speak("one");
    await waitFor(() => trialEvents.some((event) => event.type === "sample-complete" && event.outcome === "produced"));
    roomStart(room);
    await waitFor(() => roomEvents.some((event) => event.type === "notice"));
    assert.equal(runtimes, 0);
    const accepted = trialEvents.find((event) => event.type === "sample-accepted")!;
    trial.send(JSON.stringify({ type: "playback-drained", playbackId: accepted.playbackId }));
    await waitFor(() => trialEvents.some((event) => event.outcome === "drained"));
    roomStart(room);
    await waitFor(() => roomEvents.some((event) => event.phase === "ready"));
    speak("two");
    await waitFor(() => trialEvents.some((event) => event.type === "sample-error" && event.code === "BUSY"));
    assert.equal(output.opens, 1);
    room.send(JSON.stringify({ type: "stop" }));
    await waitFor(() => roomEvents.filter((event) => event.phase === "stopped").length >= 2);
    speak("three");
    await waitFor(() => output.opens === 2);
  } finally { room.close(); trial.close(); await host.close(); }
});

test("disconnect during runtime creation holds admission until the pending start settles", async () => {
  const output = fakeOutput();
  let finish!: (runtime: ConversationRuntime) => void;
  const pending = new Promise<ConversationRuntime>((resolve) => { finish = resolve; });
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers, speechOutput: output.speechOutput,
    createRuntime: () => pending });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output");
  const roomEvents = observe(room);
  const trialEvents = observe(trial);
  const speak = (requestId: string) => trial.send(JSON.stringify({ type: "speak-sample", requestId,
    catalogueRevision: output.providers.catalogue().revision, selection: DEFAULT_LOCAL_OUTPUT, text: "hello" }));
  try {
    await Promise.all([onceOpen(room), onceOpen(trial)]);
    roomStart(room);
    await waitFor(() => roomEvents.some((event) => event.phase === "loading"));
    room.close();
    await onceClose(room);
    speak("busy");
    await waitFor(() => trialEvents.some((event) => event.code === "BUSY"));
    assert.equal(output.opens, 0);
    finish(fakeRuntime());
    // The next event loop turn lets the cancelled start settle and release.
    await new Promise((resolve) => setImmediate(resolve));
    speak("after");
    await waitFor(() => output.opens === 1);
    assert.ok(!roomEvents.some((event) => event.phase === "ready"));
  } finally { finish(fakeRuntime()); room.close(); trial.close(); await host.close(); }
});

test("tryout cancellation clears delivery immediately while native retirement holds admission and shutdown", async () => {
  const output = fakeOutput();
  let finish!: () => void;
  const retirement = new Promise<void>((resolve) => { finish = resolve; });
  let retiring = false;
  output.speechOutput.quiesce = async () => { retiring = true; await retirement; };
  let runtimes = 0;
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers, speechOutput: output.speechOutput,
    createRuntime: async () => { runtimes++; return fakeRuntime(); } });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output");
  const roomEvents = observe(room);
  const trialEvents = observe(trial);
  try {
    await Promise.all([onceOpen(room), onceOpen(trial)]);
    trial.send(JSON.stringify({ type: "speak-sample", requestId: "native", text: "hello",
      selection: DEFAULT_LOCAL_OUTPUT, catalogueRevision: output.providers.catalogue().revision }));
    await waitFor(() => trialEvents.some((event) => event.outcome === "produced"));
    trial.send(JSON.stringify({ type: "cancel-sample", requestId: "native" }));
    await waitFor(() => retiring && trialEvents.some((event) => event.outcome === "cancelled"));
    assert.ok(trialEvents.some((event) => event.action === "clear"));
    roomStart(room);
    await waitFor(() => roomEvents.some((event) => event.type === "notice"));
    assert.equal(runtimes, 0);
    let closed = false;
    const closing = host.close().then(() => { closed = true; });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(closed, false);
    finish();
    await closing;
  } finally { finish(); room.close(); trial.close(); await host.close(); }
});

test("room stop waits for its runtime quiescence before admitting tryout", async () => {
  const output = fakeOutput();
  let finish!: () => void;
  const retirement = new Promise<void>((resolve) => { finish = resolve; });
  let retiring = false;
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers, speechOutput: output.speechOutput,
    createRuntime: async () => ({ ...fakeRuntime(), quiesce: async () => { retiring = true; await retirement; } }) });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output");
  const roomEvents = observe(room);
  const trialEvents = observe(trial);
  const speak = (requestId: string) => trial.send(JSON.stringify({ type: "speak-sample", requestId, text: "hello",
    selection: DEFAULT_LOCAL_OUTPUT, catalogueRevision: output.providers.catalogue().revision }));
  try {
    await Promise.all([onceOpen(room), onceOpen(trial)]);
    roomStart(room);
    await waitFor(() => roomEvents.some((event) => event.phase === "ready"));
    room.send(JSON.stringify({ type: "stop" }));
    await waitFor(() => retiring);
    speak("waiting");
    await waitFor(() => trialEvents.some((event) => event.code === "BUSY"));
    assert.equal(output.opens, 0);
    finish();
    await waitFor(() => roomEvents.some((event) => event.phase === "stopped"));
    speak("after-retirement");
    await waitFor(() => output.opens === 1);
  } finally { finish(); room.close(); trial.close(); await host.close(); }
});

test("room Stop clears buffered browser audio before shared factory native retirement finishes", async (t) => {
  const output = fakeOutput();
  let finish!: () => void;
  const retirement = new Promise<void>((resolve) => { finish = resolve; });
  let loads = 0;
  let synths = 0;
  const factory = new SpeechOutputFactory({ local: new LocalSpeechOutputFactory({ createSynth: async () => {
    loads++;
    return { numSpeakers: 1, sampleRate: 22_050, synth: async () => {
      if (++synths === 2) await retirement;
      return { samples: Float32Array.of(0.25), sampleRate: 22_050, firstAudioMs: undefined,
        totalSynthMs: 1, audioDurationMs: 1 / 22.05, rtf: 1 };
    } };
  } }) });
  t.mock.method(MercuryChatModel.prototype, "stream", async function* () { yield "One. Two."; });
  const host = await startBrowserVoiceServer({ port: 0, providers: output.providers,
    speechOutput: { catalogue: () => output.catalogue,
      createOutput: (selection, context) => factory.create(selection, { beforeRequest: context.recheck, signal: context.signal }),
      quiesce: (selection) => factory.quiesce(selection), close: () => factory.close() },
    createRuntime: (config) => createConversationRuntime(config, { INCEPTIONLABS_API_KEY: "fake-only" }, fakeRuntime().stt,
      { speechOutputFactory: factory }) });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const trial = new WebSocket(host.url.replace("http", "ws") + "/speech-output");
  const observed = observe(room);
  const trialEvents = observe(trial);
  let packets = 0;
  room.on("message", (_data, binary) => { if (binary) packets++; });
  const speak = (requestId: string) => trial.send(JSON.stringify({ type: "speak-sample", requestId,
    catalogueRevision: output.providers.catalogue().revision, selection: { ...DEFAULT_LOCAL_OUTPUT, voice: 0 }, text: "Trial." }));
  try {
    await Promise.all([onceOpen(room), onceOpen(trial)]);
    room.send(JSON.stringify({ type: "start", engine: "mercury-instant", mode: "hold", pauseMs: 700,
      voice: "cedar", barge: false, speechOutput: DEFAULT_LOCAL_OUTPUT }));
    await waitFor(() => observed.some((event) => event.phase === "ready"));
    room.send(new Float32Array(320).fill(0.2));
    room.send(JSON.stringify({ type: "commit-input" }));
    await waitFor(() => packets === 1 && synths === 2);
    const playback = observed.find((event) => event.type === "playback" && event.action === "start")!;
    room.send(JSON.stringify({ type: "stop" }));
    await waitFor(() => observed.some((event) => event.type === "playback" && event.action === "clear" &&
      event.playbackId === playback.playbackId));
    assert.ok(!observed.some((event) => event.phase === "stopped"));
    speak("busy");
    await waitFor(() => trialEvents.some((event) => event.requestId === "busy" && event.code === "BUSY"));
    finish();
    await waitFor(() => observed.some((event) => event.phase === "stopped"));
    assert.equal(packets, 1, "cancelled native lookahead must not reach the room");
    speak("after-room");
    await waitFor(() => trialEvents.some((event) => event.requestId === "after-room" && event.outcome === "produced"));
    assert.equal(loads, 1, "room and independent trial must reuse the same actual graph");
    assert.equal(synths, 3);
  } finally { finish(); room.close(); trial.close(); await host.close(); }
});

test("retired room recognizers cannot publish into a restarted room", async () => {
  const recognizers: FakeSttSession[] = [];
  const host = await startBrowserVoiceServer({ port: 0, createRuntime: async () => {
    const recognizer = new FakeSttSession();
    recognizers.push(recognizer);
    return { ...fakeRuntime(), stt: { open: async () => recognizer } };
  } });
  const room = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const observed = observe(room);
  try {
    await onceOpen(room);
    roomStart(room);
    await waitFor(() => observed.some((event) => event.phase === "ready"));
    room.send(JSON.stringify({ type: "stop" }));
    await waitFor(() => observed.some((event) => event.phase === "stopped"));
    roomStart(room);
    await waitFor(() => observed.filter((event) => event.phase === "ready").length === 2);
    recognizers[0]!.emit("final", { text: "stale old transcript" });
    recognizers[0]!.emit("error", { err: new Error("stale old error") });
    recognizers[1]!.emit("partial", { text: "current transcript" });
    await waitFor(() => observed.some((event) => event.text === "current transcript"));
    assert.ok(!observed.some((event) => event.text === "stale old transcript" || event.message?.includes("stale old error")),
      "events from the retired loop must be invalidated");
  } finally { room.close(); await host.close(); }
});

function fakeRuntime(
  onOpen: (endpoint: { mode?: string; minTrailingSilenceMs?: number }) => void = () => {},
): ConversationRuntime {
  const session = new FakeSttSession();
  return {
    label: "test speech · test voice",
    stt: {
      open: async (config) => {
        const endpoint = config?.endpoint ?? {};
        session.eager = endpoint.mode !== "manual";
        onOpen(endpoint);
        return session;
      },
    },
    responder: {
      open: async () => ({
        async *respond() {
          yield { type: "token" as const, text: "Hello back.", at: performance.now() };
          yield {
            type: "audio" as const,
            segment: {
              index: 0,
              sentence: "Hello back.",
              samples: new Float32Array(320).fill(0.1),
              sampleRate: 16_000,
              readyAtMs: 1,
              synthMs: 1,
              audioDurationMs: 20,
            },
          };
        },
        interrupt() {},
        async close() {},
      }),
    },
  };
}

class FakeSttSession extends EventEmitter {
  #spoken = false;
  eager = true;
  pushAudio(): void {
    if (this.#spoken) return;
    this.#spoken = true;
    this.emit("partial", { text: "hello" });
    if (this.eager) this.commit();
  }
  flush(): void {
    if (this.#spoken) this.commit();
  }
  commit(): void {
    this.emit("endpoint", {});
    this.emit("final", { text: "hello browser" });
  }
  reset(): void {}
  async end(): Promise<void> {}
}

function onceOpen(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.OPEN) return Promise.resolve();
  return new Promise((resolve, reject) => {
    socket.once("open", resolve);
    socket.once("error", reject);
  });
}

function onceClose(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.CLOSED) return Promise.resolve();
  return new Promise((resolve) => socket.once("close", resolve));
}

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for browser event");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
