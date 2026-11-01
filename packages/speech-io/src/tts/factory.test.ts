import assert from "node:assert/strict";
import { test } from "node:test";
import { DEFAULT_LOCAL_OUTPUT } from "./catalogue.ts";
import { LocalSpeechOutputFactory } from "./factory.ts";
import type { SynthRequest } from "./synth.ts";

function result(sampleRate: number) {
  return { samples: Float32Array.of(0.5, 0.5), sampleRate, firstAudioMs: undefined, totalSynthMs: 1, audioDurationMs: 1, rtf: 1 };
}
async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of stream) out.push(item);
  return out;
}

test("selected output is inert until open, snapshots config and defaults to Kristin", async () => {
  const requests: SynthRequest[] = [];
  const models: string[] = [];
  const factory = new LocalSpeechOutputFactory({ createSynth: async (model) => {
    models.push(model);
    return { numSpeakers: 1, sampleRate: 22050, synth: async (request) => { requests.push(request); return result(22050); } };
  } });
  const config = { ...DEFAULT_LOCAL_OUTPUT, provider: { ...DEFAULT_LOCAL_OUTPUT.provider } };
  const tts = factory.create(config);
  config.voice = "af_heart";
  assert.deepEqual(models, []);
  const session = await tts.open();
  const audio = await collect(session.speak("Hello. Goodbye."));
  assert.deepEqual(models, ["piper-kristin-medium"]);
  assert.ok(audio.every((segment) => segment.sampleRate === 22050));
  assert.deepEqual(requests.map((request) => [request.sid, request.speed]), [[0, 1], [0, 1]]);
  await session.close();
  assert.throws(() => session.speak("Late."), /closed/);
  await factory.close();
  await assert.rejects(factory.open(), /closed/);
});

test("Kokoro voices reuse one graph and serialize native requests", async () => {
  let loads = 0;
  let active = 0;
  const requests: SynthRequest[] = [];
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => {
    loads += 1;
    return { numSpeakers: 54, sampleRate: 24000, synth: async (request) => {
      active += 1;
      assert.equal(active, 1);
      requests.push(request);
      await new Promise<void>((resolve) => setImmediate(resolve));
      active -= 1;
      return result(24000);
    } };
  } });
  const config = { ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: "af_heart", speed: 1.25 };
  const [heart, michael] = await Promise.all([factory.open(config), factory.open({ ...config, voice: "am_michael" })]);
  await Promise.all([collect(heart.speak("Hello.")), collect(michael.speak("Hi."))]);
  assert.equal(loads, 1);
  assert.deepEqual(requests.map((request) => request.sid), [3, 16]);
  assert.ok(requests.every((request) => request.speed === 1.25));
  await heart.close();
  await michael.close();
  await factory.close();
});

test("failed loads can retry, while wrong bundle metadata fails closed", async () => {
  let attempts = 0;
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => {
    attempts += 1;
    if (attempts === 1) throw new Error("load failed");
    return { numSpeakers: 53, sampleRate: 24000, synth: async () => result(24000) };
  } });
  const config = { ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: "af_heart" };
  await assert.rejects(factory.open(config), /load failed/);
  await assert.rejects(factory.open(config), /pinned bundle/);
  assert.equal(attempts, 2);
  await factory.close();
});

test("closing a session suppresses in flight audio", async () => {
  let release!: () => void;
  const wait = new Promise<void>((resolve) => { release = resolve; });
  let markStarted!: () => void;
  const started = new Promise<void>((resolve) => { markStarted = resolve; });
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => ({
    numSpeakers: 1, sampleRate: 22050,
    synth: async () => { markStarted(); await wait; return result(22050); },
  }) });
  const session = await factory.open();
  const audio = session.speak("Hi.")[Symbol.asyncIterator]();
  const pending = audio.next();
  await started;
  await session.close();
  release();
  assert.equal((await pending).done, true);
  await factory.close();
});
