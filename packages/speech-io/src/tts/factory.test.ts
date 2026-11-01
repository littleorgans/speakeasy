import assert from "node:assert/strict";
import { test } from "node:test";
import { DEFAULT_LOCAL_OUTPUT } from "./catalogue.ts";
import { LocalSpeechOutputFactory } from "./factory.ts";
import type { SynthRequest } from "./synth.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));

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

test("factory close retains native work until pending synthesis settles", async () => {
  const native = deferred<void>();
  const started = deferred<void>();
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => ({
    numSpeakers: 1, sampleRate: 22050,
    synth: async () => { started.resolve(); await native.promise; return result(22050); },
  }) });
  const session = await factory.open();
  const audio = collect(session.speak("Hi."));
  await started.promise;
  await session.close();
  let closed = false;
  const closing = factory.close().then(() => { closed = true; });
  try {
    await nextTurn();
    assert.equal(closed, false, "factory shutdown released a running native call");
    await assert.rejects(factory.open(), /closed/);
  } finally {
    native.resolve();
    await closing;
    assert.deepEqual(await audio, []);
  }
  assert.equal(closed, true);
});

test("quiesce waits for abandoned lookahead even after iterator and session close", async () => {
  const native = deferred<void>();
  const started = deferred<void>();
  let calls = 0;
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => ({
    numSpeakers: 1, sampleRate: 22050,
    synth: async () => {
      if (++calls === 2) { started.resolve(); await native.promise; }
      return result(22050);
    },
  }) });
  const session = await factory.open();
  const audio = session.speak("First. Second.")[Symbol.asyncIterator]();
  await audio.next();
  await started.promise;
  await audio.return?.();
  await session.close();
  let idle = false;
  try {
    const barrier = factory.quiesce().then(() => { idle = true; });
    await nextTurn();
    assert.equal(idle, false, "closed iteration still owns a native synthesis call");
    native.reject(new Error("native failed after cancellation"));
    await barrier;
    assert.equal(idle, true);
  } finally {
    native.resolve();
    await factory.close();
  }
  assert.equal(calls, 2);
});

test("quiesce covers running and queued voices on the same graph without loading another", async () => {
  const firstNative = deferred<void>();
  const secondNative = deferred<void>();
  const started = [deferred<void>(), deferred<void>()];
  const requests: SynthRequest[] = [];
  let loads = 0;
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => {
    ++loads;
    return { numSpeakers: 54, sampleRate: 24000, synth: async (request) => {
      const index = requests.push(request) - 1;
      started[index]!.resolve();
      await [firstNative, secondNative][index]!.promise;
      return result(24000);
    } };
  } });
  const config = { ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: "af_heart" };
  await factory.quiesce(config);
  assert.equal(loads, 0);
  const [heart, bella] = await Promise.all([factory.open(config), factory.open({ ...config, voice: 2 })]);
  const audio = [collect(heart.speak("Heart.")), collect(bella.speak("Bella."))];
  await started[0]!.promise;
  await nextTurn();
  await heart.close();
  await bella.close();
  let idle = false;
  try {
    const barrier = factory.quiesce(config).then(() => { idle = true; });
    await factory.quiesce(); // Unused Kristin must remain unloaded.
    assert.equal(loads, 1);
    firstNative.resolve();
    await started[1]!.promise;
    await nextTurn();
    assert.equal(idle, false, "queued voice still owns the graph");
    secondNative.resolve();
    await barrier;
    assert.deepEqual(requests.map(({ sid }) => sid), [3, 2]);
    assert.deepEqual(await Promise.all(audio), [[], []]);
  } finally {
    firstNative.resolve();
    secondNative.resolve();
    await factory.close();
  }
});

test("shutdown and quiesce await a pending load without opening a closed factory", async () => {
  const loaded = deferred<void>();
  const started = deferred<void>();
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => {
    started.resolve();
    await loaded.promise;
    return { numSpeakers: 1, sampleRate: 22050, synth: async () => result(22050) };
  } });
  const opening = assert.rejects(factory.open(), /closed/);
  await started.promise;
  let idle = false;
  let closed = false;
  const barrier = factory.quiesce().then(() => { idle = true; });
  const closing = factory.close().then(() => { closed = true; });
  const repeatedClose = factory.close();
  try {
    await nextTurn();
    assert.equal(idle, false, "pending load still owns local work");
    assert.equal(closed, false, "shutdown must retain the pending load");
  } finally {
    loaded.resolve();
    await Promise.all([opening, barrier, closing, repeatedClose]);
  }
  assert.equal(idle, true);
  assert.equal(closed, true);
  await factory.quiesce();
});

test("a failed pending load becomes quiescent and does not wedge shutdown", async () => {
  const loaded = deferred<void>();
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => {
    await loaded.promise;
    throw new Error("failed native load");
  } });
  const opening = assert.rejects(factory.open(), /failed native load/);
  const barrier = factory.quiesce();
  const closing = factory.close();
  loaded.resolve();
  await Promise.all([opening, barrier, closing]);
});

test("sessions cannot start more native work after factory shutdown", async () => {
  let calls = 0;
  const factory = new LocalSpeechOutputFactory({ createSynth: async () => ({
    numSpeakers: 1, sampleRate: 22050,
    synth: async () => { ++calls; return result(22050); },
  }) });
  const session = await factory.open();
  await factory.close();
  await assert.rejects(collect(session.speak("Too late.")), /closed/);
  assert.equal(calls, 0);
  await session.close();
});
