import assert from "node:assert/strict";
import { test, type TestContext } from "node:test";
import { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_MODELS, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";
import { LocalSpeechOutputFactory } from "../../speech-io/src/tts/factory.ts";
import type { AudioSegment } from "../../speech-io/src/tts/contract.ts";
import { createProviderCredentials } from "./provider-availability.ts";
import { ProviderPolicy } from "./provider-policy.ts";
import { createHostProviderConfig } from "./setup-routes.ts";
import { buildSpeechOutputCatalogue, type ApiOutputModel } from "./speech-output-catalog.ts";
import type { SpeechOutputEvent } from "./speech-output-protocol.ts";
import { SpeechOutputSession, type SpeechOutputContext, type SpeechOutputSessionOptions } from "./speech-output-session.ts";

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(check: () => boolean) {
  for (let i = 0; i < 30 && !check(); ++i) await tick();
  assert.ok(check(), "expected asynchronous observation");
}

const api: ApiOutputModel = {
  id: "approved-api", label: "API", provider: { kind: "cartesia" },
  defaultVoice: "voice", defaultSpeed: 1, speed: { min: 1, max: 1 }, sampleRate: 16_000,
  streaming: "sentence", licence: "Provider terms", voices: [{ id: "voice", label: "Voice", languages: ["en"] }],
  readiness: { state: "ready", reason: null },
};
function audio(index = 0, samples = new Float32Array([0.25, -0.5])): AudioSegment {
  return { index, samples, sentence: `Sentence ${index}.`, sampleRate: 16_000,
    readyAtMs: 1, synthMs: 9_999, audioDurationMs: samples.length / 16 };
}

function harness(t: TestContext, options: {
  createOutput?: SpeechOutputSessionOptions["createOutput"];
  quiesce?: SpeechOutputSessionOptions["quiesce"];
  timeoutMs?: number;
  apiProvider?: "cartesia" | "openai";
  combinedPolicy?: boolean;
} = {}) {
  const events: SpeechOutputEvent[] = [];
  const packets: Uint8Array[] = [];
  const contexts: SpeechOutputContext[] = [];
  const state = { present: true, leaseHeld: false, busy: false, opened: 0, created: 0, closed: 0, releases: 0,
    quiesced: 0, clock: 10, useCombinedRevision: false, spoken: [] as (string | AsyncIterable<string>)[] };
  const catalogue = buildSpeechOutputCatalogue({
    local: { default: DEFAULT_LOCAL_OUTPUT, models: LOCAL_OUTPUT_MODELS.map((model) => ({ ...model,
      readiness: { state: "ready", reason: null },
    })) }, localRegistry: LOCAL_OUTPUT_REGISTRY,
    apiModels: [{ ...api, provider: { kind: options.apiProvider ?? "cartesia" } }], configured: () => state.present,
  });
  const credentials = createProviderCredentials(() => ({ CARTESIA_API_KEY: state.present ? "test-only" : undefined,
    OPENAI_API_KEY: state.present ? "test-only" : undefined }));
  const providers = options.combinedPolicy ? createHostProviderConfig({ credentials, readiness: () => [],
    inputAdapters: [], outputCatalogue: () => catalogue, serverRunId: "run" }) : undefined;
  const policy = providers?.policy ?? new ProviderPolicy({
    serverRunId: "run", catalogue: () => ({ schema: 1, revision: catalogue.revision, input: [], output: catalogue.output }),
    registries: () => ({ input: { localPresets: [], apiModels: [] }, output: catalogue.registry }),
    credentials,
  });
  const serviceCatalogue = () => state.useCombinedRevision && providers
    ? { ...catalogue, revision: providers.catalogue().revision } : catalogue;
  const session = new SpeechOutputSession({
    sessionId: "protected-session", catalogue: serviceCatalogue, policy,
    createOutput: async (config, context) => {
      ++state.created;
      contexts.push(context);
      if (options.createOutput) return options.createOutput(config, context);
      return { open: async () => {
        ++state.opened;
        return { speak: async function* (text: string | AsyncIterable<string>) { state.spoken.push(text); yield audio(); },
          close: async () => { ++state.closed; } };
      } };
    },
    quiesce: async (config) => { ++state.quiesced; await options.quiesce?.(config); },
    acquire: () => {
      if (state.busy || state.leaseHeld) return undefined;
      state.leaseHeld = true;
      return () => { assert.equal(state.leaseHeld, true); state.leaseHeld = false; ++state.releases; };
    },
    sendEvent: (event) => events.push(event), sendAudio: (packet) => packets.push(packet),
    now: () => state.clock, timeoutMs: options.timeoutMs,
  });
  const send = (value: unknown) => session.handle(JSON.stringify(value));
  const speak = (requestId = "trial", selection = catalogue.default, extra = {}) => send({
    type: "speak-sample", requestId, selection, text: "One. Two.", catalogueRevision: serviceCatalogue().revision, ...extra,
  });
  const selectApi = () => {
    const selection = catalogue.output[2].config;
    send({ type: "select-output", requestId: "choose", catalogueRevision: serviceCatalogue().revision, selection });
    const chosen = events.findLast((event) => event.type === "output-selected");
    assert.ok(chosen?.type === "output-selected");
    return { selection, selectionRevision: chosen.selectionRevision };
  };
  const completions = () => events.filter((event) => event.type === "sample-complete");
  t.after(() => session.close());
  return { session, state, catalogue, policy, providers, events, packets, contexts, send, speak, selectApi, completions };
}

test("trial uses shared TTS text session and existing PCM, separates produced from drained and holds work lease", async (t) => {
  const h = harness(t);
  h.speak();
  h.send({ type: "playback-drained", playbackId: 1 });
  await until(() => h.completions().length === 1);
  assert.deepEqual(h.completions().map((event) => event.outcome), ["produced"]);
  assert.equal(h.state.leaseHeld, true);
  assert.equal(h.state.closed, 1);
  assert.deepEqual(h.state.spoken, ["One. Two."]);
  const header = new DataView(h.packets[0].buffer);
  assert.equal(header.getUint8(0), 1);
  assert.equal(header.getUint32(1, true), 1);
  assert.equal(header.getUint32(5, true), 16_000);
  assert.deepEqual(new Float32Array(h.packets[0].slice(9).buffer), audio().samples);
  assert.deepEqual(h.events.filter((event) => event.type === "playback").map((event) => event.action), ["start", "end"]);
  h.send({ type: "playback-drained", playbackId: 999 });
  assert.equal(h.completions().length, 1);
  h.speak("busy");
  assert.equal(h.events.at(-1)?.type, "sample-error");
  h.send({ type: "playback-drained", playbackId: 1 });
  await until(() => h.state.releases === 1);
  assert.deepEqual(h.completions().map((event) => event.outcome), ["produced", "drained"]);
  h.send({ type: "playback-drained", playbackId: 1 });
  h.speak("trial");
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "REQUEST_REUSED"));
  assert.equal(h.state.releases, 1);
});

test("first audio readiness includes admission and open, while compatibility synthMs cannot invent provider timings", async (t) => {
  const opening = deferred<void>();
  const h = harness(t, { createOutput: () => ({ open: async () => {
    await opening.promise;
    return { speak: async function* () { yield audio(); }, close: async () => {} };
  } }) });
  h.speak();
  await until(() => h.state.created === 1);
  h.state.clock = 135;
  opening.resolve();
  await until(() => h.completions().length === 1);
  const result = h.completions()[0];
  assert.deepEqual(result.timing.serverFirstAudioReadyMs, {
    status: "measured", ms: 125, clock: "server",
    boundary: "Admission to first nonempty, nonzero PCM available for host delivery; includes TTS open.",
  });
  for (const key of ["browserFirstAudioMs", "providerRequestWallMs", "providerProcessingMs", "providerNetworkMs"] as const) {
    assert.equal(result.timing[key].status, "unavailable");
    assert.equal(result.segments[0].timing[key].status, "unavailable");
  }
});

test("Stop clears immediately but retains lease through pending operation and native quiescence; late generation cannot deliver", async (t) => {
  const later = deferred<void>();
  const nativeIdle = deferred<void>();
  let closed = 0;
  const h = harness(t, {
    createOutput: () => ({ open: async () => ({
      speak: async function* () { yield audio(); await later.promise; yield audio(1); },
      close: async () => { ++closed; },
    }) }), quiesce: () => nativeIdle.promise,
  });
  h.speak();
  await until(() => h.packets.length === 1);
  h.send({ type: "cancel-sample", requestId: "different" });
  assert.equal(h.completions().length, 0);
  h.send({ type: "cancel-sample", requestId: "trial" });
  assert.ok(h.contexts[0].signal.aborted);
  assert.equal(h.events.at(-2)?.type, "playback");
  assert.equal(h.completions()[0].outcome, "cancelled");
  assert.equal(h.state.leaseHeld, true);
  h.speak("blocked-until-idle");
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "BUSY"));
  const count = h.events.length;
  later.resolve();
  await until(() => h.state.quiesced === 1);
  assert.equal(closed, 1);
  assert.equal(h.state.releases, 0);
  assert.equal(h.packets.length, 1);
  assert.equal(h.events.length, count);
  nativeIdle.resolve();
  await until(() => h.state.releases === 1);
  h.speak("replacement");
  await until(() => h.packets.length === 3);
  const accepted = h.events.filter((event) => event.type === "sample-accepted");
  assert.deepEqual(accepted.map((event) => event.playbackId), [1, 2]);
  h.send({ type: "playback-drained", playbackId: 1 });
  assert.equal(h.state.leaseHeld, true);
});

test("close waits for a late open and closes it, then waits for native quiescence without delivering", async (t) => {
  const opened = deferred<void>();
  const idle = deferred<void>();
  let closes = 0;
  let speaks = 0;
  const h = harness(t, { createOutput: () => ({ open: async () => {
    await opened.promise;
    return { speak: async function* () { ++speaks; yield audio(); }, close: async () => { ++closes; } };
  } }), quiesce: () => idle.promise });
  h.speak();
  await until(() => h.state.created === 1);
  let closed = false;
  const closing = h.session.close().then(() => { closed = true; });
  const eventCount = h.events.length;
  await tick();
  assert.equal(closed, false);
  assert.equal(h.state.releases, 0);
  opened.resolve();
  await until(() => h.state.quiesced === 1);
  assert.equal(closes, 1);
  assert.equal(speaks, 0);
  assert.equal(closed, false);
  idle.resolve();
  await closing;
  h.speak("after-close");
  assert.equal(h.events.length, eventCount);
  assert.equal(h.packets.length, 0);
  assert.equal(h.state.releases, 1);
});

test("API dispatch requires explicit session/config selection; client source and saved grant cannot arm it", async (t) => {
  const h = harness(t);
  const selection = h.catalogue.output[2].config;
  h.speak("unselected", selection);
  await until(() => h.events.some((event) => event.type === "sample-error" && event.code === "SELECTION_REQUIRED"));
  await until(() => !h.state.leaseHeld);
  assert.equal(h.state.created, 0);
  for (const extra of [
    { source: { kind: "saved-take", takeId: "take" } }, { grant: { approved: true } },
    { sessionId: "other" }, { approved: true },
  ]) h.speak("forged", selection, extra);
  assert.equal(h.state.created, 0);
  const chosen = h.selectApi();
  assert.equal(h.state.created, 0);
  h.speak("selected", selection, { selectionRevision: chosen.selectionRevision });
  await until(() => h.packets.length === 1);
  assert.equal(h.state.created, 1);
  assert.throws(() => h.contexts[0].apiKey?.(), "completed production cannot transmit again");
});

test("policy revocation between sentences prevents a later transmission and clears prior playback", async (t) => {
  const later = deferred<void>();
  let transmissions = 0;
  const h = harness(t, { createOutput: (_config, context) => ({ open: async () => ({
    speak: async function* () {
      context.recheck(); ++transmissions; yield audio();
      await later.promise;
      context.recheck(); ++transmissions; yield audio(1);
    }, close: async () => {},
  }) }) });
  const chosen = h.selectApi();
  h.speak("api", chosen.selection, { selectionRevision: chosen.selectionRevision });
  await until(() => h.packets.length === 1);
  h.policy.revoke("protected-session", "output");
  later.resolve();
  await until(() => h.state.releases === 1);
  assert.equal(transmissions, 1);
  assert.equal(h.packets.length, 1);
  assert.ok(h.events.some((event) => event.type === "playback" && event.action === "clear"));
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "SELECTION_REQUIRED"));
});

test("credential disappearance after open is denied before PCM delivery and further provider use", async (t) => {
  const opening = deferred<void>();
  const h = harness(t, { createOutput: () => ({ open: async () => {
    await opening.promise;
    return { speak: async function* () { yield audio(); }, close: async () => {} };
  } }) });
  const chosen = h.selectApi();
  h.speak("key-removed", chosen.selection, { selectionRevision: chosen.selectionRevision });
  await until(() => h.state.created === 1);
  h.state.present = false;
  opening.resolve();
  await until(() => h.state.releases === 1);
  assert.equal(h.packets.length, 0);
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "PROVIDER_UNAVAILABLE"));
  assert.throws(() => h.contexts[0].apiKey?.());
});

test("host busy and stale catalogue reject before TTS construction", async (t) => {
  const h = harness(t);
  h.state.busy = true;
  h.speak();
  h.state.busy = false;
  h.speak("stale", h.catalogue.default, { catalogueRevision: "old" });
  assert.equal(h.state.created, 0);
  assert.equal(h.state.leaseHeld, false);
  assert.deepEqual(h.events.filter((event) => event.type === "sample-error").map((event) => event.code), ["BUSY", "SELECTION_STALE"]);
});

test("empty and silent results cannot claim first audio or a fabricated drain", async (t) => {
  for (const samples of [new Float32Array(), new Float32Array(3)]) {
    const h = harness(t, { createOutput: () => ({ open: async () => ({
      speak: async function* () { yield audio(0, samples); }, close: async () => {},
    }) }) });
    h.speak();
    await until(() => h.completions().length === 1);
    assert.equal(h.completions()[0].timing.serverFirstAudioReadyMs.status, "unavailable");
    assert.deepEqual(h.completions().map((event) => event.outcome), ["produced"]);
    if (samples.length === 0) {
      await until(() => h.state.releases === 1);
      assert.equal(h.events.some((event) => event.type === "playback"), false);
    } else assert.equal(h.state.leaseHeld, true);
  }
});

test("adapter failure is sanitized and invalid PCM is rejected without poisoning subsequent trials", async (t) => {
  let attempts = 0;
  const h = harness(t, { createOutput: () => ({ open: async () => ({
    speak: async function* () {
      if (++attempts === 1) throw new Error("private transport credential detail");
      yield audio(0, new Float32Array([NaN]));
    }, close: async () => {},
  }) }) });
  h.speak("failed");
  await until(() => h.state.releases === 1);
  assert.equal(JSON.stringify(h.events).includes("private"), false);
  h.speak("invalid-pcm");
  await until(() => h.state.releases === 2);
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "AUDIO_INVALID"));
  assert.equal(h.packets.length, 0);
});

test("a drain timeout fails and clears, without reporting playback drained", async (t) => {
  const h = harness(t, { timeoutMs: 15 });
  h.speak();
  await until(() => h.completions().length === 1);
  await new Promise<void>((resolve) => setTimeout(resolve, 25));
  assert.deepEqual(h.completions().map((event) => event.outcome), ["produced"]);
  assert.ok(h.events.some((event) => event.type === "sample-error" && event.code === "TIMEOUT"));
  assert.equal(h.state.releases, 1);
});

test("actual approved factory and sentence pipeline deliver first PCM while fake native lookahead still runs", async (t) => {
  const later = deferred<void>();
  const nativeIdle = deferred<void>();
  const requests: { text: string; sid?: number; speed?: number }[] = [];
  const models: string[] = [];
  const factory = new LocalSpeechOutputFactory({ createSynth: async (model) => {
    models.push(model);
    return { numSpeakers: 1, sampleRate: 22_050, synth: async (request) => {
      requests.push(request);
      if (requests.length === 2) { await later.promise; nativeIdle.resolve(); }
      return { samples: Float32Array.of(0.4, 0.5), sampleRate: 22_050,
        firstAudioMs: undefined, totalSynthMs: 3, audioDurationMs: 2 / 22.05, rtf: 1 };
    } };
  } });
  const h = harness(t, { createOutput: (selection) => factory.create(selection), quiesce: () => nativeIdle.promise });
  t.after(() => factory.close());
  h.speak();
  await until(() => h.packets.length === 1 && requests.length === 2);
  assert.deepEqual(models, ["piper-kristin-medium"]);
  assert.deepEqual(requests.map((request) => [request.text, request.sid, request.speed]), [["One.", 0, 1], ["Two.", 0, 1]]);
  assert.equal(new DataView(h.packets[0].buffer).getUint32(5, true), 22_050);
  assert.equal(h.completions().length, 0);
  h.send({ type: "cancel-sample", requestId: "trial" });
  await tick();
  assert.equal(h.state.leaseHeld, true);
  later.resolve();
  await until(() => h.state.releases === 1);
  assert.equal(h.packets.length, 1);
  assert.deepEqual(h.completions().map((event) => event.outcome), ["cancelled"]);
});

test("provider callback metrics retain each actual request boundary and resolved voice without invented aggregates", async (t) => {
  const requestTiming = {
    providerRequestWallMs: { status: "measured" as const, ms: 41, clock: "server" as const, boundary: "Transport invocation to complete PCM body." },
    providerProcessingMs: { status: "provider-reported" as const, ms: 12, metric: "fake_compute_ms", boundary: "Documented fake sentence compute.", evidenceRef: "fixture-contract" },
    providerNetworkMs: { status: "unavailable" as const, reason: "No independent network interval." },
  };
  const h = harness(t, { createOutput: (config, context) => ({ open: async () => ({
    speak: async function* () {
      context.recheck();
      assert.equal(context.apiKey?.(), "test-only");
      context.observe({ model: config.model, voice: "resolved-voice", text: "One.", timing: requestTiming });
      yield audio();
    }, close: async () => {},
  }) }) });
  const chosen = h.selectApi();
  h.speak("metrics", chosen.selection, { selectionRevision: chosen.selectionRevision });
  await until(() => h.completions().length === 1);
  const result = h.completions()[0];
  assert.deepEqual(result.requests, [{ model: chosen.selection.model, voice: "resolved-voice", text: "One.", timing: requestTiming }]);
  assert.equal(result.timing.providerRequestWallMs.status, "unavailable");
  assert.equal(result.timing.providerProcessingMs.status, "unavailable");
  assert.equal(result.timing.providerNetworkMs.status, "unavailable");
});

test("OpenAI segment observations preserve returned transcript and distinct setup, first PCM and request scopes", async (t) => {
  const wall = { status: "measured" as const, ms: 55, clock: "server" as const, boundary: "response.create through response.done." };
  const first = { ...wall, ms: 18, boundary: "response.create to first nonempty provider PCM event." };
  const setup = { ...wall, ms: 10, boundary: "Socket construction to session.updated." };
  const absent = { status: "unavailable" as const, reason: "Provider exposes no supported metric." };
  const h = harness(t, { apiProvider: "openai", createOutput: (config) => ({ open: async () => ({
    speak: async function* () { yield { ...audio(), provider: "openai", model: config.model, voice: config.voice,
      requestedText: "One.", returnedTranscript: "One!", sessionSetupWallMs: setup, providerFirstAudioMs: first,
      providerRequestWallMs: wall, providerProcessingMs: absent, providerNetworkMs: absent }; },
    close: async () => {},
  }) }) });
  const chosen = h.selectApi();
  h.speak("observations", chosen.selection, { selectionRevision: chosen.selectionRevision });
  await until(() => h.completions().length === 1);
  const result = h.completions()[0];
  assert.deepEqual(result.requests[0], { model: chosen.selection.model, voice: chosen.selection.voice,
    text: "One.", returnedTranscript: "One!", sessionSetupWallMs: setup, providerFirstAudioMs: first,
    timing: { providerRequestWallMs: wall, providerProcessingMs: absent, providerNetworkMs: absent } });
  assert.deepEqual(result.segments[0].timing.providerRequestWallMs, wall);
  assert.equal(result.timing.providerRequestWallMs.status, "unavailable");
});

test("actual combined policy rejects split catalogue revisions and permits selection/local trial only after host projection aligns them", async (t) => {
  const h = harness(t, { combinedPolicy: true });
  const combinedRevision = h.providers!.catalogue().revision;
  assert.notEqual(combinedRevision, h.catalogue.revision);
  for (const catalogueRevision of [h.catalogue.revision, combinedRevision]) {
    h.send({ type: "select-output", requestId: "mismatch", catalogueRevision, selection: h.catalogue.default });
    assert.ok(h.events.at(-1)?.type === "sample-error");
  }
  assert.equal(h.events.some((event) => event.type === "output-selected"), false);
  h.speak("mismatched-local");
  await until(() => h.state.releases === 1);
  assert.equal(h.state.created, 0);
  assert.ok(h.events.filter((event) => event.type === "sample-error").every((event) => event.code === "SELECTION_STALE"));

  // Host owns projecting its combined policy revision into metadata/session.
  h.state.useCombinedRevision = true;
  h.send({ type: "select-output", requestId: "aligned", catalogueRevision: combinedRevision, selection: h.catalogue.default });
  assert.equal(h.events.at(-1)?.type, "output-selected");
  h.speak("aligned-local");
  await until(() => h.completions().length === 1);
  assert.equal(h.state.created, 1);
  assert.equal(h.completions()[0].outcome, "produced");
  h.send({ type: "playback-drained", playbackId: 2 });
  await until(() => h.state.releases === 2);
  assert.equal(h.completions().at(-1)?.outcome, "drained");
  h.send({ type: "select-output", requestId: "old", catalogueRevision: h.catalogue.revision, selection: h.catalogue.default });
  assert.ok(h.events.at(-1)?.type === "sample-error");
});
