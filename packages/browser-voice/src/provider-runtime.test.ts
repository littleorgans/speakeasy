import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import { createConversationRuntime, type ConversationRuntime, type RuntimeConfig } from "@speakeasy/convo-engine";
import { LocalSpeechOutputFactory, CARTESIA_VOICES, decodeWavSamples,
  type OpenAISocket, type OpenAISocketFactory } from "@speakeasy/speech-io";
import type { SpeechOutputConfig } from "@speakeasy/speech-io/providers";
import { WebSocket } from "ws";
import { MercuryChatModel } from "../../llm/src/mercury.ts";
import { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_MODELS } from "../../speech-io/src/tts/catalogue.ts";
import { startBrowserVoiceServer } from "./host.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { createProviderRuntime } from "./provider-runtime.ts";
import { createHostProviderConfig } from "./setup-routes.ts";
import type { BrowserRuntimeContext, RuntimeFactory } from "./session.ts";
import type { ProviderDispatchContext, ProviderDispatchRequest } from "./provider-policy.ts";

const CARTESIA: SpeechOutputConfig = { schema: 1, provider: { kind: "cartesia" },
  model: "sonic-3.6-2026-08-27", voice: "katie", speed: 1 };
const OPENAI: SpeechOutputConfig = { schema: 1, provider: { kind: "openai" },
  model: "gpt-realtime-2.1-mini", voice: "marin", speed: 1 };
const START = { type: "start", engine: "mercury-instant", mode: "hold", pauseMs: 700, voice: "cedar", barge: false };

function fixture(options: { fetch?: typeof globalThis.fetch; createSocket?: OpenAISocketFactory } = {}) {
  const environment: Record<string, string | undefined> = {};
  const requests: Array<{ body: Record<string, any>; signal: AbortSignal | null | undefined }> = [];
  const sockets: OutputSocket[] = [];
  let loads = 0;
  const localOutput = new LocalSpeechOutputFactory({ createSynth: async () => {
    loads++;
    return { numSpeakers: 1, sampleRate: 22_050, synth: async () => ({
      samples: Float32Array.of(0.25), sampleRate: 22_050, firstAudioMs: undefined,
      totalSynthMs: 1, audioDurationMs: 1 / 22.05, rtf: 1,
    }) };
  } });
  const assembly = createProviderRuntime({ credentials: createProviderCredentials(() => environment),
    readiness: () => [], localOutput,
    localCatalogue: { default: DEFAULT_LOCAL_OUTPUT,
      models: LOCAL_OUTPUT_MODELS.map((model) => ({ ...model, readiness: { state: "ready" as const, reason: null } })) },
    createSocket: options.createSocket ?? ((_url, key) => {
      assert.equal(key, "fake-openai");
      const socket = new OutputSocket();
      sockets.push(socket);
      return socket;
    }),
    fetch: options.fetch ?? (async (_url, init) => {
      assert.equal(new Headers(init?.headers).get("authorization"), "Bearer fake-cartesia");
      requests.push({ body: JSON.parse(String(init?.body)), signal: init?.signal });
      return new Response(Float32Array.of(0.5, -0.5).buffer);
    }),
  });
  return { ...assembly, environment, requests, sockets, get loads() { return loads; } };
}
type Assembly = ReturnType<typeof fixture>;

class OutputSocket extends EventEmitter implements OpenAISocket {
  messages: Array<Record<string, any>> = [];
  closed = 0;
  constructor(ready = true) {
    super();
    if (ready) queueMicrotask(() => this.event({ type: "session.created" }));
  }
  event(value: unknown): void { this.emit("message", Buffer.from(JSON.stringify(value))); }
  send(raw: string): void {
    const message = JSON.parse(raw);
    this.messages.push(message);
    queueMicrotask(() => {
      if (message.type === "session.update") {
        this.event({ type: "session.updated", session: { ...message.session, model: OPENAI.model } });
      } else if (message.type === "response.create") {
        const id = `response_${this.messages.length}`;
        this.event({ type: "response.created", response: { id, metadata: message.response.metadata } });
        this.event({ type: "response.output_audio.delta", response_id: id, item_id: "item", output_index: 0,
          content_index: 0, delta: Buffer.from([0, 64, 0, 192]).toString("base64") });
        this.event({ type: "response.output_audio_transcript.done", response_id: id, item_id: "item",
          output_index: 0, content_index: 0, transcript: "Returned sentence." });
        this.event({ type: "response.done", response: { id, status: "completed" } });
      }
    });
  }
  close(): void { this.closed++; }
}

class Recognizer extends EventEmitter {
  turns = 0;
  ended = 0;
  pushAudio(): void {}
  flush(): void {
    this.emit("endpoint", {});
    this.emit("final", { text: `Current turn ${++this.turns}` });
  }
  reset(): void {}
  async end(): Promise<void> { this.ended++; }
}

function emptyRuntime(): ConversationRuntime {
  return { label: "fake room", stt: { open: async () => new Recognizer() },
    responder: { open: async () => ({ async *respond() {}, interrupt() {}, async close() {} }) } };
}

class Peer {
  socket: WebSocket;
  events: Array<Record<string, any>> = [];
  packets: Buffer[] = [];
  constructor(base: string, path = "/voice", drain = true) {
    this.socket = new WebSocket(base.replace("http", "ws") + path);
    this.socket.on("message", (data, binary) => {
      if (binary) { this.packets.push(Buffer.from(data as Buffer)); return; }
      const value = JSON.parse(data.toString());
      this.events.push(value);
      if (drain && value.type === "playback" && value.action === "end") {
        this.send({ type: "playback-drained", playbackId: value.playbackId });
      }
    });
  }
  async open(): Promise<void> {
    if (this.socket.readyState === WebSocket.OPEN) return;
    await new Promise<void>((resolve, reject) => { this.socket.once("open", resolve); this.socket.once("error", reject); });
  }
  send(value: unknown): void { this.socket.send(JSON.stringify(value)); }
  async reply(value: unknown, predicate: (event: Record<string, any>) => boolean): Promise<Record<string, any>> {
    const from = this.events.length;
    this.send(value);
    await until(() => this.events.slice(from).some(predicate));
    return this.events.slice(from).find(predicate)!;
  }
  async select(assembly: Assembly, selection = CARTESIA, requestId = "room_selection") {
    const event = await this.reply({ type: "select-output", requestId, selection,
      catalogueRevision: assembly.providers.catalogue().revision }, (event) => event.requestId === requestId);
    assert.equal(event.type, "output-selected");
    return { speechOutput: event.selection as SpeechOutputConfig, catalogueRevision: event.catalogueRevision as string,
      selectionRevision: event.selectionRevision as number };
  }
  stop(): Promise<Record<string, any>> {
    return this.reply({ type: "stop" }, (event) => event.phase === "stopped");
  }
  async close(): Promise<void> {
    if (this.socket.readyState === WebSocket.CLOSED) return;
    const closed = new Promise<void>((resolve) => this.socket.once("close", resolve));
    this.socket.close();
    await closed;
  }
}

async function until(check: () => boolean): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for fake room operation");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

function hostFor(assembly: Assembly, createRuntime: RuntimeFactory = async () => emptyRuntime(), runtimeConfig?: RuntimeConfig) {
  return startBrowserVoiceServer({ port: 0, ...assembly, createRuntime, runtimeConfig });
}

test("combined provider catalogue derives output and hash from one raw snapshot", async () => {
  const assembly = fixture();
  try {
    const first = assembly.speechOutput.catalogue();
    const second = { ...first, revision: "changed", output: [] };
    let reads = 0;
    const providers = createHostProviderConfig({ credentials: createProviderCredentials(() => ({})),
      readiness: () => [], inputAdapters: [], outputCatalogue: () => ++reads === 1 ? first : second });
    const catalogue = providers.catalogue();
    assert.equal(reads, 1);
    assert.deepEqual(catalogue.output, first.output);
    assert.equal(catalogue.revision, createHash("sha256").update(JSON.stringify({ input: catalogue.input,
      output: first.output, outputRevision: first.revision })).digest("hex"));
  } finally { await assembly.output.close(); }
});

test("actual assembly metadata and selection stay inert for no, one and both fake keys", async () => {
  const assembly = fixture();
  try {
    for (const kinds of [[], ["cartesia"], ["openai"], ["cartesia", "openai"]]) {
      assembly.environment.CARTESIA_API_KEY = kinds.includes("cartesia") ? "fake-cartesia" : undefined;
      assembly.environment.OPENAI_API_KEY = kinds.includes("openai") ? "fake-openai" : undefined;
      const raw = assembly.speechOutput.catalogue();
      const combined = assembly.providers.catalogue();
      assert.deepEqual(raw.models.filter((model) => model.provider.kind !== "local").map((model) => model.provider.kind), kinds);
      assert.deepEqual(combined.input.filter((entry) => entry.config.provider.kind !== "local")
        .map((entry) => entry.config.provider.kind), kinds);
      assert.deepEqual(combined.output, raw.output);
      assert.deepEqual(raw.default, DEFAULT_LOCAL_OUTPUT);
      assert.deepEqual(raw.models.filter((model) => model.provider.kind === "local").map((model) => model.id),
        LOCAL_OUTPUT_MODELS.map((model) => model.id));
      for (const config of [CARTESIA, OPENAI].filter((config) => kinds.includes(config.provider.kind))) {
        const model = raw.models.find((model) => model.provider.kind === config.provider.kind)!;
        assert.deepEqual(model.voices.map((voice) => voice.id), [config.voice]);
        assert.equal(model.id, config.model);
        assert.equal(model.streaming, "sentence");
        assert.match(model.licence, /unverified/i);
        assembly.providers.policy.select({ sessionId: "metadata-session", direction: "output",
          catalogueRevision: combined.revision, config });
      }
    }
    const old = assembly.providers.catalogue().revision;
    assembly.environment.OPENAI_API_KEY = undefined;
    assert.notEqual(assembly.providers.catalogue().revision, old);
    assert.equal(assembly.speechOutput.catalogue().models.some((model) => model.provider.kind === "openai"), false);
    assert.equal(assembly.loads, 0);
    assert.deepEqual(assembly.requests, []);
    assert.deepEqual(assembly.sockets, []);
  } finally { await assembly.output.close(); }
});

test("mounted actual factory drives both trial adapters with exact wires and current policy", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  assembly.environment.OPENAI_API_KEY = "fake-openai";
  const host = await hostFor(assembly);
  const trial = new Peer(host.url, "/speech-output");
  try {
    await trial.open();
    for (const [index, config] of [CARTESIA, OPENAI].entries()) {
      const selected = await trial.select(assembly, config, `trial_select_${index}`);
      assert.equal(assembly.requests.length + assembly.sockets.length, index);
      const produced = await trial.reply({ type: "speak-sample", requestId: `trial_${index}`,
        selection: config, catalogueRevision: selected.catalogueRevision,
        selectionRevision: selected.selectionRevision, text: "Requested sentence." }, (event) => event.outcome === "produced");
      assert.equal(produced.requests.length, 1);
      assert.equal(produced.requests[0].model, config.model);
      assert.equal(produced.requests[0].text, "Requested sentence.");
      await until(() => trial.events.some((event) => event.playbackId === produced.playbackId && event.outcome === "drained"));
    }
    const wire = assembly.requests[0]!.body;
    assert.equal(wire.model_id, CARTESIA.model);
    assert.deepEqual(wire.voice, { id: CARTESIA_VOICES.katie });
    assert.equal(wire.generation_config.speed, 1);
    assert.equal(wire.output_format.sample_rate, 44_100);
    const update = assembly.sockets[0]!.messages.find((message) => message.type === "session.update")!;
    assert.equal(update.session.audio.output.voice, "marin");
    assert.equal(update.session.audio.output.speed, 1);
    const request = assembly.sockets[0]!.messages.find((message) => message.type === "response.create")!;
    assert.equal(request.response.input[0].content[0].text, "Requested sentence.");
    assert.ok(trial.packets.length >= 2);
    assert.equal(assembly.loads, 0);
  } finally { await trial.close(); await host.close(); }
});

test("assembled factory has no ambient credential path for unguarded hosted output", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  assembly.environment.OPENAI_API_KEY = "fake-openai";
  try {
    const session = await assembly.output.create(CARTESIA).open();
    const stream = session.speak("Unauthorized sentence.")[Symbol.asyncIterator]();
    await assert.rejects(stream.next());
    await session.close();
    await assert.rejects(assembly.output.create(OPENAI).open());
    assert.equal(assembly.requests.length + assembly.sockets.length + assembly.loads, 0);
  } finally { await assembly.output.close(); }
});

test("assembly uses both actual file input adapters and preserves the exact PCM view", async () => {
  let calls = 0;
  const samples = Float32Array.of(-0, 1.25, -0.5);
  const assembly = fixture({ fetch: async (_url, init) => {
    calls++;
    assert.ok(init?.body instanceof FormData);
    const file = init.body.get("file");
    assert.ok(file instanceof Blob);
    const bytes = await file.arrayBuffer();
    assert.deepEqual(file.type === "audio/wav" ? decodeWavSamples(new Uint8Array(bytes)) :
      new Float32Array(bytes), samples);
    return Response.json({ type: "transcript", text: "file transcript" });
  } });
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  assembly.environment.OPENAI_API_KEY = "fake-openai";
  try {
    for (const adapter of assembly.providers.inputAdapters) {
      const config = { schema: 1, provider: { kind: adapter.kind, model: adapter.model } };
      const selected = assembly.providers.policy.select({ sessionId: "input-session", direction: "input",
        config, catalogueRevision: assembly.providers.catalogue().revision });
      const result = await assembly.providers.policy.dispatch({ ...selected,
        source: { kind: "live-capture", sessionId: "input-session", operationId: `host_${adapter.kind}` } },
      (context) => adapter.transcribe(samples, context, new AbortController().signal));
      assert.equal((result as { text: string }).text, "file transcript");
    }
    assert.equal(calls, 2);
    assert.equal(assembly.loads, 0);
  } finally { await assembly.output.close(); }
});

test("room rejects missing, borrowed, forged and stale authority before runtime construction", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  assembly.environment.OPENAI_API_KEY = "fake-openai";
  let creations = 0;
  const host = await hostFor(assembly, async () => { creations++; return emptyRuntime(); });
  const room = new Peer(host.url);
  const trial = new Peer(host.url, "/speech-output");
  try {
    await Promise.all([room.open(), trial.open()]);
    const borrowed = await trial.select(assembly);
    await room.reply({ ...START, ...borrowed }, (event) => event.phase === "stopped");
    await room.reply({ ...START, speechOutput: CARTESIA }, (event) => event.phase === "stopped");
    const variants = [
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, selectionRevision: selected.selectionRevision + 1 }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, catalogueRevision: "outdated" }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, speechOutput: { ...CARTESIA, speed: 1.2 } }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, speechOutput: OPENAI }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, speechOutput: { ...CARTESIA, voice: 0 } }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, sessionId: "client-forged" }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, source: { kind: "live-response-text" } }),
      (selected: Awaited<ReturnType<Peer["select"]>>) => ({ ...selected, grantId: "client-forged" }),
    ];
    for (const change of variants) {
      const selected = await room.select(assembly);
      await room.reply({ ...START, ...change(selected) }, (event) => event.phase === "stopped");
      await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
    }
    const old = await room.select(assembly);
    assembly.environment.OPENAI_API_KEY = undefined;
    await room.reply({ ...START, ...old }, (event) => event.phase === "stopped");
    const replacement = await room.select(assembly);
    const invalid = await room.reply({ type: "select-output", requestId: "invalid id",
      selection: CARTESIA, catalogueRevision: replacement.catalogueRevision },
    (event) => event.type === "output-selection-error");
    assert.equal(invalid.requestId, null);
    assert.equal(invalid.code, "COMMAND_INVALID");
    await room.reply({ ...START, ...replacement }, (event) => event.phase === "stopped");
    assert.equal(creations, 0);
    assert.equal(assembly.requests.length + assembly.sockets.length + assembly.loads, 0);
  } finally { await room.close(); await trial.close(); await host.close(); }
});

test("effective hosted defaults cannot bypass selection; local legacy and Realtime stay independent", async () => {
  for (const defaults of [{ speechOutput: CARTESIA }, { ttsEngine: "cartesia" as const }]) {
    const assembly = fixture();
    assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
    const created: RuntimeConfig[] = [];
    const host = await hostFor(assembly, async (config) => { created.push(config); return emptyRuntime(); }, defaults);
    const room = new Peer(host.url);
    try {
      await room.open();
      const rejected = await room.reply(START, (event) => event.phase === "stopped" || event.phase === "ready");
      assert.equal(rejected.phase, "stopped");
      assert.deepEqual(created, []);
      await room.reply({ ...START, speechOutput: DEFAULT_LOCAL_OUTPUT }, (event) => event.phase === "ready");
      assert.deepEqual(created[0]!.speechOutput, DEFAULT_LOCAL_OUTPUT);
      await room.stop();
      await room.reply({ ...START, engine: "realtime" }, (event) => event.phase === "ready");
      assert.equal(created[1]!.voice, "cedar");
      assert.equal(created[1]!.speechOutput, undefined);
      assert.equal(assembly.requests.length + assembly.sockets.length, 0);
    } finally { await room.close(); await host.close(); }
  }
});

test("room requires host policy and output metadata even with forged complete start fields", async () => {
  for (const includePolicy of [false, true]) {
    const assembly = fixture();
    assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
    let creations = 0;
    const host = await startBrowserVoiceServer({ port: 0,
      providers: includePolicy ? assembly.providers : undefined,
      createRuntime: async () => { creations++; return emptyRuntime(); } });
    const room = new Peer(host.url);
    try {
      await room.open();
      const selected = await room.reply({ type: "select-output", requestId: "missing", selection: CARTESIA,
        catalogueRevision: assembly.providers.catalogue().revision }, (event) => event.type === "output-selection-error");
      assert.equal(selected.requestId, includePolicy ? "missing" : null);
      await room.reply({ ...START, speechOutput: CARTESIA, catalogueRevision: assembly.providers.catalogue().revision,
        selectionRevision: 1 }, (event) => event.phase === "stopped");
      assert.equal(creations, 0);
    } finally { await room.close(); await host.close(); await assembly.output.close(); }
  }
});

test("real room runtime forwards trusted context to both adapters across turns and revokes on Stop", async (t) => {
  t.mock.method(MercuryChatModel.prototype, "stream", async function* () { yield "Current sentence."; });
  for (const config of [CARTESIA, OPENAI]) {
    const assembly = fixture();
    assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
    assembly.environment.OPENAI_API_KEY = "fake-openai";
    const contexts: BrowserRuntimeContext[] = [];
    const dispatches: ProviderDispatchRequest[] = [];
    const dispatch = assembly.providers.policy.dispatch.bind(assembly.providers.policy);
    t.mock.method(assembly.providers.policy, "dispatch", function<T>(request: ProviderDispatchRequest,
      invoke: (context: ProviderDispatchContext) => T | Promise<T>) {
      dispatches.push(structuredClone(request));
      return dispatch(request, invoke);
    });
    const recognizer = new Recognizer();
    const host = await hostFor(assembly, (selected, context) => {
      contexts.push(context!);
      return createConversationRuntime(selected, { INCEPTIONLABS_API_KEY: "fake-chat" },
        { open: async () => recognizer }, { speechOutputFactory: assembly.output,
          speechOutputContext: context?.speechOutputContext });
    });
    const room = new Peer(host.url);
    try {
      await room.open();
      const selected = await room.select(assembly, config);
      assert.equal(assembly.requests.length + assembly.sockets.length, 0);
      await room.reply({ ...START, ...selected }, (event) => event.phase === "ready");
      for (let turn = 1; turn <= 2; turn++) {
        await room.reply({ type: "commit-input" }, (event) => event.type === "metrics");
        assert.equal(room.packets.length, turn);
      }
      assert.equal(contexts.length, 1);
      assert.equal(dispatches.length, 1);
      assert.equal(dispatches[0]!.source.kind, "live-response-text");
      assert.ok(dispatches[0]!.source.kind === "live-response-text");
      assert.equal(dispatches[0]!.source.sessionId, dispatches[0]!.sessionId);
      assert.match(dispatches[0]!.source.operationId, /^[0-9a-f-]{36}$/);
      const context = contexts[0]!.speechOutputContext!;
      assert.equal(context.getKey?.(), config.provider.kind === "cartesia" ? "fake-cartesia" : "fake-openai");
      assert.equal(context.signal?.aborted, false);
      if (config.provider.kind === "cartesia") assert.equal(assembly.requests.length, 2);
      else assert.equal(assembly.sockets[0]!.messages.filter((message) => message.type === "response.create").length, 2);
      const busy = await room.reply({ type: "select-output", requestId: "active_selection", selection: config,
        catalogueRevision: selected.catalogueRevision }, (event) => event.requestId === "active_selection");
      assert.equal(busy.code, "BUSY");
      room.send({ ...START, ...selected });
      await context.beforeRequest?.();
      await room.stop();
      assert.equal(context.signal?.aborted, true);
      assert.throws(() => context.getKey?.());
      assert.throws(() => context.beforeRequest?.());
      await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
      assert.equal(contexts.length, 1);
      const fresh = await room.select(assembly, config, "restart");
      assert.notEqual(fresh.selectionRevision, selected.selectionRevision);
      await room.reply({ ...START, ...fresh }, (event) => event.phase === "ready");
      assert.equal(contexts.length, 2);
      assert.notEqual(contexts[1]!.speechOutputContext!.signal, context.signal);
      assert.equal(dispatches[1]!.sessionId, dispatches[0]!.sessionId);
      assert.notDeepEqual(dispatches[1]!.source, dispatches[0]!.source);
    } finally { await room.close(); await host.close(); }
  }
});

test("a removed key blocks the next real room transmission and retires its captured authority", async (t) => {
  t.mock.method(MercuryChatModel.prototype, "stream", async function* () { yield "Current sentence."; });
  for (const config of [CARTESIA, OPENAI]) {
    const assembly = fixture();
    assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
    assembly.environment.OPENAI_API_KEY = "fake-openai";
    let context: BrowserRuntimeContext | undefined;
    const host = await hostFor(assembly, (selected, trusted) => {
      context = trusted;
      return createConversationRuntime(selected, { INCEPTIONLABS_API_KEY: "fake-chat" },
        { open: async () => new Recognizer() }, { speechOutputFactory: assembly.output,
          speechOutputContext: trusted?.speechOutputContext });
    });
    const room = new Peer(host.url);
    try {
      await room.open();
      const selected = await room.select(assembly, config);
      await room.reply({ ...START, ...selected }, (event) => event.phase === "ready");
      await room.reply({ type: "commit-input" }, (event) => event.type === "metrics");
      const count = assembly.requests.length + assembly.sockets.reduce((total, socket) => total +
        socket.messages.filter((message) => message.type === "response.create").length, 0);
      const key = config.provider.kind === "cartesia" ? "CARTESIA_API_KEY" : "OPENAI_API_KEY";
      const saved = assembly.environment[key];
      assembly.environment[key] = undefined;
      await room.reply({ type: "commit-input" }, (event) => event.type === "notice" && event.level === "error");
      assert.equal(context?.speechOutputContext?.signal?.aborted, true);
      assembly.environment[key] = saved;
      assert.throws(() => context?.speechOutputContext?.getKey?.());
      assert.throws(() => context?.speechOutputContext?.beforeRequest?.());
      assert.equal(assembly.requests.length + assembly.sockets.reduce((total, socket) => total +
        socket.messages.filter((message) => message.type === "response.create").length, 0), count);
      await room.stop();
      await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
    } finally { await room.close(); await host.close(); }
  }
});

test("Stop aborts actual OpenAI readiness before awaiting pending room startup", async () => {
  const sockets: OutputSocket[] = [];
  const assembly = fixture({ createSocket: () => { const socket = new OutputSocket(false); sockets.push(socket); return socket; } });
  assembly.environment.OPENAI_API_KEY = "fake-openai";
  let context: BrowserRuntimeContext | undefined;
  const recognizer = new Recognizer();
  const host = await hostFor(assembly, (selected, trusted) => {
    context = trusted;
    return createConversationRuntime(selected, { INCEPTIONLABS_API_KEY: "fake-chat" },
      { open: async () => recognizer }, { speechOutputFactory: assembly.output,
        speechOutputContext: trusted?.speechOutputContext });
  });
  const room = new Peer(host.url);
  try {
    await room.open();
    const selected = await room.select(assembly, OPENAI);
    room.send({ ...START, ...selected });
    await until(() => sockets.length === 1);
    await room.stop();
    assert.equal(context?.speechOutputContext?.signal?.aborted, true);
    assert.equal(sockets[0]!.closed, 1);
    assert.equal(recognizer.ended, 1);
    sockets[0]!.event({ type: "session.created" });
    assert.deepEqual(sockets[0]!.messages, []);
    assert.equal(room.events.some((event) => event.phase === "ready"), false);
    await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
    assert.equal(sockets.length, 1);
  } finally { await room.close(); await host.close(); }
});

test("cancelled creation and its actual retirement retain shared admission and suppress late startup", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  let finish!: (runtime: ConversationRuntime) => void;
  let retire!: () => void;
  const opening = new Promise<ConversationRuntime>((resolve) => { finish = resolve; });
  const retirement = new Promise<void>((resolve) => { retire = resolve; });
  let context: BrowserRuntimeContext | undefined;
  let retiring = false;
  let responderOpens = 0;
  const host = await hostFor(assembly, async (_config, trusted) => { context = trusted; return opening; });
  const room = new Peer(host.url);
  const trial = new Peer(host.url, "/speech-output");
  const speak = (id: string) => trial.reply({ type: "speak-sample", requestId: id, selection: DEFAULT_LOCAL_OUTPUT,
    text: "Local trial.", catalogueRevision: assembly.providers.catalogue().revision },
  (event) => event.requestId === id && (event.type === "sample-error" || event.outcome === "produced"));
  try {
    await Promise.all([room.open(), trial.open()]);
    const selected = await room.select(assembly);
    room.send({ ...START, ...selected });
    await until(() => context !== undefined);
    room.send({ type: "stop" });
    await until(() => context!.speechOutputContext!.signal!.aborted);
    assert.equal((await speak("opening_busy")).code, "BUSY");
    finish({ ...emptyRuntime(), responder: { open: async () => { responderOpens++; throw new Error("late runtime opened"); } },
      quiesce: async () => { retiring = true; await retirement; } });
    await until(() => retiring);
    assert.equal((await speak("retiring_busy")).code, "BUSY");
    assert.equal(responderOpens, 0);
    assert.equal(room.events.some((event) => event.phase === "stopped"), false);
    retire();
    await until(() => room.events.some((event) => event.phase === "stopped"));
    assert.equal((await speak("after_retirement")).outcome, "produced");
    assert.equal(room.events.some((event) => event.phase === "ready"), false);
    assert.throws(() => context!.speechOutputContext!.getKey!());
  } finally { finish(emptyRuntime()); retire(); await room.close(); await trial.close(); await host.close(); }
});

test("busy and failed starts revoke room permits; reconnect cannot borrow a previous connection", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  let creations = 0;
  let failedContext: BrowserRuntimeContext | undefined;
  const host = await hostFor(assembly, async (_config, context) => {
    creations++;
    if (creations === 1) { failedContext = context; throw new Error("fake startup failure"); }
    return emptyRuntime();
  });
  let room = new Peer(host.url);
  const trial = new Peer(host.url, "/speech-output", false);
  try {
    await Promise.all([room.open(), trial.open()]);
    const produced = await trial.reply({ type: "speak-sample", requestId: "occupy", selection: DEFAULT_LOCAL_OUTPUT,
      catalogueRevision: assembly.providers.catalogue().revision, text: "Local trial." }, (event) => event.outcome === "produced");
    const selected = await room.select(assembly);
    await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
    assert.equal(creations, 0);
    await trial.reply({ type: "playback-drained", playbackId: produced.playbackId }, (event) => event.outcome === "drained");
    await room.reply({ ...START, ...selected }, (event) => event.phase === "stopped");
    assert.equal(creations, 0);
    const failing = await room.select(assembly);
    await room.reply({ ...START, ...failing }, (event) => event.phase === "stopped");
    assert.equal(failedContext?.speechOutputContext?.signal?.aborted, true);
    await room.reply({ ...START, ...failing }, (event) => event.phase === "stopped");
    assert.equal(creations, 1);
    const disconnected = await room.select(assembly);
    await room.close();
    room = new Peer(host.url);
    await room.open();
    await room.reply({ ...START, ...disconnected }, (event) => event.phase === "stopped");
    assert.equal(creations, 1);
    const fresh = await room.select(assembly);
    await room.reply({ ...START, ...fresh }, (event) => event.phase === "ready");
    assert.equal(creations, 2);
    assert.equal(assembly.requests.length + assembly.sockets.length, 0);
  } finally { await room.close(); await trial.close(); await host.close(); }
});

test("failed runtime retirement keeps admission, refuses reselection and rejects host shutdown", async () => {
  const assembly = fixture();
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  let retirements = 0;
  const host = await hostFor(assembly, async () => ({ ...emptyRuntime(), quiesce: async () => {
    retirements++;
    throw new Error("fake retirement failed");
  } }));
  const room = new Peer(host.url);
  const trial = new Peer(host.url, "/speech-output");
  try {
    await Promise.all([room.open(), trial.open()]);
    const selected = await room.select(assembly);
    await room.reply({ ...START, ...selected }, (event) => event.phase === "ready");
    await room.reply({ type: "stop" }, (event) => event.type === "notice" && event.level === "error");
    assert.equal(retirements, 1);
    const busy = await trial.reply({ type: "speak-sample", requestId: "retirement_busy", selection: DEFAULT_LOCAL_OUTPUT,
      text: "Local trial.", catalogueRevision: assembly.providers.catalogue().revision }, (event) => event.type === "sample-error");
    assert.equal(busy.code, "BUSY");
    const selection = await room.reply({ type: "select-output", requestId: "retirement_selection", selection: CARTESIA,
      catalogueRevision: assembly.providers.catalogue().revision }, (event) => event.type === "output-selection-error");
    assert.equal(selection.code, "BUSY");
    assert.equal(assembly.loads, 0);
    await assert.rejects(host.close());
  } finally { await room.close(); await trial.close(); await host.close().catch(() => {}); }
});

test("Stop during a later real Cartesia sentence rejects late PCM and old handles after restart", async (t) => {
  t.mock.method(MercuryChatModel.prototype, "stream", async function* () { yield "First sentence. Second sentence."; });
  let requests = 0;
  let late!: (response: Response) => void;
  let pendingSignal: AbortSignal | null | undefined;
  const assembly = fixture({ fetch: async (_url, init) => {
    if (++requests === 2) {
      pendingSignal = init?.signal;
      return new Promise<Response>((resolve) => { late = resolve; });
    }
    return new Response(Float32Array.of(0.25).buffer);
  } });
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  const contexts: BrowserRuntimeContext[] = [];
  const host = await hostFor(assembly, (config, context) => {
    contexts.push(context!);
    return createConversationRuntime(config, { INCEPTIONLABS_API_KEY: "fake-chat" },
      { open: async () => new Recognizer() }, { speechOutputFactory: assembly.output,
        speechOutputContext: context?.speechOutputContext });
  });
  const room = new Peer(host.url);
  try {
    await room.open();
    const selected = await room.select(assembly);
    await room.reply({ ...START, ...selected }, (event) => event.phase === "ready");
    room.send({ type: "commit-input" });
    await until(() => requests === 2 && room.packets.length === 1);
    await room.stop();
    assert.equal(pendingSignal?.aborted, true);
    assert.ok(room.events.some((event) => event.type === "playback" && event.action === "clear"));
    const fresh = await room.select(assembly, CARTESIA, "fresh_after_stop");
    await room.reply({ ...START, ...fresh }, (event) => event.phase === "ready");
    late(new Response(Float32Array.of(0.75).buffer));
    assert.throws(() => contexts[0]!.speechOutputContext!.beforeRequest!());
    assert.throws(() => contexts[0]!.speechOutputContext!.getKey!());
    assert.equal(contexts[1]!.speechOutputContext!.signal!.aborted, false);
    await room.reply({ type: "commit-input" }, (event) => event.type === "metrics");
    assert.equal(requests, 4);
    assert.equal(room.packets.length, 3);
    assert.equal(room.events.filter((event) => event.phase === "ready").length, 2);
  } finally {
    late?.(new Response(Float32Array.of(0.75).buffer));
    await room.close(); await host.close();
  }
});

test("policy failure between room sentences clears delivered audio and blocks later transmission", async (t) => {
  t.mock.method(MercuryChatModel.prototype, "stream", async function* () { yield "First sentence. Second sentence."; });
  let requests = 0;
  const assembly = fixture({ fetch: async () => {
    requests++;
    assembly.environment.CARTESIA_API_KEY = undefined;
    return new Response(Float32Array.of(0.25).buffer);
  } });
  assembly.environment.CARTESIA_API_KEY = "fake-cartesia";
  const host = await hostFor(assembly, (config, context) => createConversationRuntime(config,
    { INCEPTIONLABS_API_KEY: "fake-chat" }, { open: async () => new Recognizer() },
    { speechOutputFactory: assembly.output, speechOutputContext: context?.speechOutputContext }));
  const room = new Peer(host.url, "/voice", false);
  try {
    await room.open();
    const selected = await room.select(assembly);
    await room.reply({ ...START, ...selected }, (event) => event.phase === "ready");
    await room.reply({ type: "commit-input" }, (event) => event.type === "notice" && event.level === "error");
    assert.equal(requests, 1);
    assert.equal(room.packets.length, 1);
    assert.ok(room.events.some((event) => event.type === "playback" && event.action === "clear"));
    await room.stop();
  } finally { await room.close(); await host.close(); }
});
