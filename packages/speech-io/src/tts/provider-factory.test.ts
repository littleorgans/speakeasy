import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import * as io from "../index.ts";
import * as stt from "../stt.ts";
import { decodeWavSamples, encodeWavFloat32 } from "../bench/wav.ts";
import type { SpeechOutputConfig } from "../provider-config.ts";
import { DEFAULT_LOCAL_OUTPUT } from "./catalogue.ts";
import { LocalSpeechOutputFactory } from "./factory.ts";
import { DEFAULT_OPENAI_OUTPUT_CONFIG, type OpenAIAudioSegment, type OpenAISocket } from "./openai.ts";
import { SpeechOutputFactory } from "./provider-factory.ts";

const cartesia: SpeechOutputConfig = {
  schema: 1, provider: { kind: "cartesia" }, model: "sonic-3.6-2026-08-27", voice: "katie", speed: 1.2,
};

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of stream) items.push(item);
  return items;
}

test("public exports reuse strict audio helpers and expose input/output composition", () => {
  assert.equal(io.decodeWavSamples, decodeWavSamples);
  assert.equal(io.encodeWavFloat32, encodeWavFloat32);
  assert.equal(io.SpeechOutputFactory, SpeechOutputFactory);
  assert.equal(io.LocalSpeechOutputFactory, LocalSpeechOutputFactory);
  assert.equal(typeof io.CartesiaBatchTranscriber, "function");
  assert.equal(typeof io.OpenAIFileTranscriber, "function");
  assert.equal(typeof io.OpenAITextToSpeech, "function");
  assert.equal(stt.SherpaEngine, io.SherpaEngine);
  assert.equal("SpeechOutputFactory" in stt, false);
  const samples = Float32Array.of(-0, 1.25, -1.5, 0.125);
  assert.deepEqual(io.decodeWavSamples(io.encodeWavFloat32(samples, 16_000)), samples);
  const malformed = io.encodeWavFloat32(samples, 16_000);
  new DataView(malformed.buffer).setUint32(4, malformed.length, true);
  assert.throws(() => io.decodeWavSamples(malformed), /RIFF length/);
});

test("local delegation accepts the local factory's model domain without a second preset list", async () => {
  const selections: SpeechOutputConfig[] = [];
  let opens = 0;
  let retired: SpeechOutputConfig | undefined;
  const factory = new SpeechOutputFactory({ local: {
    create: (selection = DEFAULT_LOCAL_OUTPUT) => {
      selections.push(selection);
      return { open: async () => { opens++; return { async *speak() {}, async close() {} }; } };
    },
    quiesce: async (selection) => { retired = selection; },
    close: async () => {},
  } });
  const config: SpeechOutputConfig = { ...DEFAULT_LOCAL_OUTPUT, model: "registered-compatible-model", voice: "registered-voice" };
  const output = factory.create(config);
  config.model = "mutated";
  assert.equal(opens, 0);
  assert.equal(selections[0]?.model, "registered-compatible-model");
  const session = await output.open();
  await session.close();
  await factory.quiesce(selections[0]);
  assert.equal(retired?.model, "registered-compatible-model");
  await assert.rejects(output.open({ voice: "override" }), /fixed/);
  await factory.close();
  await assert.rejects(output.open(), /closed/);
});

test("shared local factory preserves residency and waits for native work after session close", async () => {
  let loads = 0;
  let release!: () => void;
  let started!: () => void;
  const running = new Promise<void>((resolve) => { started = resolve; });
  const pending = new Promise<void>((resolve) => { release = resolve; });
  const local = new LocalSpeechOutputFactory({ createSynth: async () => {
    loads++;
    return { numSpeakers: 54, sampleRate: 24_000, synth: async () => {
      started();
      await pending;
      return { samples: Float32Array.of(0.5, -0.5), sampleRate: 24_000,
        firstAudioMs: undefined, totalSynthMs: 1, audioDurationMs: 1, rtf: 1 };
    } };
  } });
  const factory = new SpeechOutputFactory({ local });
  const selected = { ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: "af_heart" };
  await factory.quiesce(selected);
  const first = factory.create(selected);
  const second = factory.create({ ...selected, voice: "am_michael" });
  assert.equal(loads, 0);
  const [a, b] = await Promise.all([first.open(), second.open()]);
  assert.equal(loads, 1);
  const audio = collect(a.speak("Hello."));
  await running;
  await a.close();
  let quiet = false;
  const retirement = factory.quiesce(selected).then(() => { quiet = true; });
  await setImmediate();
  assert.equal(quiet, false);
  release();
  await retirement;
  assert.deepEqual(await audio, []);
  await b.close();
  await factory.close();
});

test("hosted selection is inert and Cartesia receives exact selection, policy and observations", async () => {
  const calls: string[] = [];
  const observations: unknown[] = [];
  const factory = new SpeechOutputFactory({
    getKey: (kind) => { calls.push(`key:${kind}`); return "fake-key"; },
    fetch: async (_url, init) => {
      calls.push("fetch");
      const body = JSON.parse(String(init.body));
      assert.equal(body.model_id, cartesia.model);
      assert.equal(body.generation_config.speed, cartesia.speed);
      assert.deepEqual(body.voice, { id: io.CARTESIA_VOICES.katie });
      return new Response(Float32Array.of(0.5, -0.5).buffer);
    },
  });
  const config = { ...cartesia, provider: { ...cartesia.provider } };
  const output = factory.create(config, {
    beforeRequest: () => { calls.push("policy"); }, observe: (item) => observations.push(item),
  });
  config.model = "mutation";
  assert.deepEqual(calls, []);
  const session = await output.open();
  assert.deepEqual(calls, []);
  assert.equal((await collect(session.speak("Hello."))).length, 1);
  assert.deepEqual(calls, ["policy", "key:cartesia", "fetch"]);
  assert.equal(observations.length, 1);
  await session.close();
  await factory.quiesce(cartesia);
  await factory.close();
});

test("per operation credentials override host getter and revoked policy prevents dispatch", async () => {
  const keys: string[] = [];
  let requests = 0;
  let allowed = true;
  const factory = new SpeechOutputFactory({
    getKey: () => { throw new Error("must use operation credentials"); },
    fetch: async () => { requests++; return new Response(Float32Array.of(0.5).buffer); },
  });
  const session = await factory.create(cartesia, {
    getKey: () => { keys.push("operation"); return "fake-operation-key"; },
    beforeRequest: () => { if (!allowed) throw new Error("selection revoked"); },
  }).open();
  const audio = session.speak("First sentence. Second sentence.")[Symbol.asyncIterator]();
  assert.equal((await audio.next()).done, false);
  allowed = false;
  await assert.rejects(audio.next());
  assert.equal(requests, 1);
  assert.deepEqual(keys, ["operation"]);
  await session.close();
  await factory.close();
});

test("revocation signal reaches Cartesia and suppresses late transport completion", async () => {
  const controller = new AbortController();
  let dispatched!: () => void;
  const started = new Promise<void>((resolve) => { dispatched = resolve; });
  let finish!: (response: Response) => void;
  let requestSignal: AbortSignal | null | undefined;
  const factory = new SpeechOutputFactory({
    getKey: () => "fake-key",
    fetch: async (_url, init) => {
      requestSignal = init.signal;
      dispatched();
      return new Promise<Response>((resolve) => { finish = resolve; });
    },
  });
  const session = await factory.create(cartesia, { signal: controller.signal }).open();
  const audio = collect(session.speak("Hello."));
  await started;
  controller.abort();
  await assert.rejects(audio);
  assert.equal(requestSignal?.aborted, true);
  finish(new Response(Float32Array.of(0.5).buffer));
  await session.close();
  await factory.close();
});

test("OpenAI uses injected socket and preserves transcript/timing fields", async () => {
  const socket = new EventEmitter() as EventEmitter & OpenAISocket;
  let closed = 0;
  const checks: string[] = [];
  const event = (value: unknown) => socket.emit("message", Buffer.from(JSON.stringify(value)));
  socket.close = () => { closed++; };
  socket.send = (data) => {
    const message = JSON.parse(data);
    queueMicrotask(() => {
      if (message.type === "session.update") {
        event({ type: "session.updated", session: { ...message.session, model: DEFAULT_OPENAI_OUTPUT_CONFIG.model } });
      } else if (message.type === "response.create") {
        event({ type: "response.created", response: { id: "r1", metadata: message.response.metadata } });
        event({ type: "response.output_audio.delta", response_id: "r1", item_id: "i1", output_index: 0,
          content_index: 0, delta: Buffer.from([0, 64, 0, 192]).toString("base64") });
        event({ type: "response.output_audio_transcript.done", response_id: "r1", item_id: "i1",
          output_index: 0, content_index: 0, transcript: "Returned words." });
        event({ type: "response.done", response: { id: "r1", status: "completed" } });
      }
    });
  };
  const factory = new SpeechOutputFactory({ createSocket: (_url, key) => {
    checks.push("socket");
    assert.equal(key, "fake-key");
    queueMicrotask(() => event({ type: "session.created" }));
    return socket;
  } });
  const controller = new AbortController();
  const output = factory.create(DEFAULT_OPENAI_OUTPUT_CONFIG, {
    beforeRequest: () => { checks.push("policy"); },
    getKey: () => { checks.push("key"); return "fake-key"; }, signal: controller.signal,
  });
  assert.deepEqual(checks, []);
  const session = await output.open();
  const segments = await collect(session.speak("Requested words.")) as OpenAIAudioSegment[];
  assert.deepEqual(checks, ["policy", "key", "socket", "policy"]);
  assert.equal(segments[0]?.returnedTranscript, "Returned words.");
  assert.equal(segments[0]?.requestedText, "Requested words.");
  assert.equal(segments[0]?.providerRequestWallMs.status, "measured");
  controller.abort();
  assert.equal(closed, 1);
  await session.close();
  await factory.close();
});

test("missing hosted execution handles never fall back or touch local output", async () => {
  const factory = new SpeechOutputFactory();
  const openai = factory.create(DEFAULT_OPENAI_OUTPUT_CONFIG);
  await assert.rejects(openai.open(), /unavailable/);
  const session = await factory.create(cartesia).open();
  await assert.rejects(collect(session.speak("Hello.")), /unavailable/);
  await session.close();
  assert.throws(() => factory.create({ ...cartesia, provider: { kind: "elevenlabs" } }), /unsupported/);
  await factory.close();
});
