import assert from "node:assert/strict";
import { once } from "node:events";
import { test } from "node:test";
import type { FinalEvent, STTErrorEvent, STTSession } from "../contract.ts";
import { Lazy, OfflineLifecycle, type OfflineAdapter } from "./offline-engine.ts";

/** The lifecycle over a fake adapter whose loads are named "model 1", "model 2", ... */

type Deferred<T> = { promise: Promise<T>; resolve(value: T): void; reject(error: unknown): void };

function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

type Fake = {
  preflights: number;
  loads: number;
  failLoads: number;
  released: string[];
  decoded: string[];
  /** When set, each decode waits for it. */
  gate?: Promise<void>;
};

function fakeLifecycle(overrides: Partial<OfflineAdapter<void, string>> = {}) {
  const fake: Fake = { preflights: 0, loads: 0, failLoads: 0, released: [], decoded: [] };
  const lifecycle = new OfflineLifecycle<void, string>({
    name: "FakeEngine",
    maxAudioMs: 1_000,
    silenceGate: undefined,
    prepare: "preflight",
    preflight: async () => {
      fake.preflights += 1;
    },
    load: async () => {
      fake.loads += 1;
      if (fake.failLoads > 0) {
        fake.failLoads -= 1;
        throw new Error("load failed");
      }
      return `model ${fake.loads}`;
    },
    release: async (loaded) => {
      fake.released.push(loaded);
    },
    decoder: (loaded) => async (audio) => {
      await fake.gate;
      fake.decoded.push(`${loaded}:${audio.length}`);
      return { text: loaded, tokens: [] };
    },
    ...overrides,
  });
  return { fake, lifecycle };
}

function decode(session: STTSession, samples = 160): Promise<FinalEvent> {
  session.pushAudio(new Float32Array(samples));
  const final = once(session, "final").then(([event]) => event as FinalEvent);
  session.flush();
  return final;
}

test("a session decodes on the loaded model, capped at maxAudioMs", async () => {
  const { fake, lifecycle } = fakeLifecycle();
  const session = await lifecycle.open();
  session.pushAudio(new Float32Array(32_000));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.text, "model 1");
  assert.deepEqual(fake.decoded, ["model 1:16000"]);
  await assert.rejects(lifecycle.open({ sampleRate: 48_000 }), /FakeEngine expects 16000 Hz/);
});

test("concurrent opens share one load", async () => {
  const { fake, lifecycle } = fakeLifecycle();
  await Promise.all([lifecycle.open(), lifecycle.open(), lifecycle.open()]);
  assert.equal(fake.preflights, 1);
  assert.equal(fake.loads, 1);
});

test("a failed load rejects open and the next open retries from preflight", async () => {
  const { fake, lifecycle } = fakeLifecycle();
  fake.failLoads = 1;
  await assert.rejects(lifecycle.open(), /load failed/);
  assert.equal((await decode(await lifecycle.open())).text, "model 2");
  assert.equal(fake.preflights, 2);
  await lifecycle.open();
  assert.equal(fake.loads, 2);
});

test("prepare follows the readiness policy", async () => {
  const preflight = fakeLifecycle();
  await preflight.lifecycle.prepare();
  await preflight.lifecycle.prepare();
  assert.deepEqual([preflight.fake.preflights, preflight.fake.loads], [2, 0]);
  const load = fakeLifecycle({ prepare: "load" });
  await load.lifecycle.prepare();
  await load.lifecycle.prepare();
  await load.lifecycle.open();
  assert.deepEqual([load.fake.preflights, load.fake.loads], [1, 1]);
});

test("serial decodes run one at a time across sessions, in flush order", async () => {
  let active = 0;
  let maxActive = 0;
  const order: number[] = [];
  const { lifecycle } = fakeLifecycle({
    serial: true,
    decoder: () => async (audio) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      order.push(audio.length);
      if (audio.length === 2) throw new Error("decode failed");
      return { text: String(audio.length), tokens: [] };
    },
  });
  const sessions = await Promise.all([lifecycle.open(), lifecycle.open(), lifecycle.open()]);
  const failed = once(sessions[1]!, "error");
  const finals = [decode(sessions[0]!, 1), decode(sessions[2]!, 3)] as const;
  sessions[1]!.pushAudio(new Float32Array(2));
  sessions[1]!.flush();
  const [first, third] = await Promise.all(finals);
  await failed;
  assert.deepEqual([first.text, third.text], ["1", "3"]);
  assert.deepEqual(order, [1, 3, 2]);
  assert.equal(maxActive, 1);
});

test("release waits for a decode in flight, then frees once; the next open loads again", async () => {
  const { fake, lifecycle } = fakeLifecycle();
  const session = await lifecycle.open();
  const hold = deferred();
  fake.gate = hold.promise;
  const final = decode(session);
  await new Promise((resolve) => setImmediate(resolve));
  const released = lifecycle.release();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(fake.released, []);
  hold.resolve();
  assert.equal((await final).text, "model 1");
  await released;
  await lifecycle.release();
  assert.deepEqual(fake.released, ["model 1"]);
  fake.gate = undefined;
  assert.equal((await decode(await lifecycle.open())).text, "model 2");
});

test("a session from a released lifetime errors without reaching the decoder", async () => {
  const { fake, lifecycle } = fakeLifecycle();
  const session = await lifecycle.open();
  await lifecycle.release();
  session.pushAudio(new Float32Array(160));
  session.flush();
  const [error] = (await once(session, "error")) as [STTErrorEvent];
  assert.match(String(error.err), /FakeEngine: model released/);
  assert.deepEqual(fake.decoded, []);
});

test("a session opened while release runs is stale", async () => {
  const load = deferred<string>();
  const { fake, lifecycle } = fakeLifecycle({ load: () => load.promise });
  const opening = lifecycle.open();
  await new Promise((resolve) => setImmediate(resolve));
  const released = lifecycle.release();
  load.resolve("model");
  await released;
  const session = await opening;
  session.pushAudio(new Float32Array(160));
  session.flush();
  await once(session, "error");
  assert.deepEqual(fake.released, ["model"]);
  assert.deepEqual(fake.decoded, []);
});

test("release during a load that fails resolves, and the newer load survives the failure", async () => {
  const loads = [deferred<string>(), deferred<string>()];
  let calls = 0;
  const { fake, lifecycle } = fakeLifecycle({ load: () => loads[calls++]!.promise });
  const first = assert.rejects(lifecycle.open(), /first load failed/);
  await new Promise((resolve) => setImmediate(resolve));
  const released = lifecycle.release();
  const second = lifecycle.open();
  await new Promise((resolve) => setImmediate(resolve));
  loads[0]!.reject(new Error("first load failed"));
  await released;
  await first;
  loads[1]!.resolve("model 2");
  await second;
  await lifecycle.open();
  assert.equal(calls, 2);
  assert.deepEqual(fake.released, []);
});

test("Lazy shares one load, retries after a failure, and detaches on take", async () => {
  let loads = 0;
  const lazy = new Lazy(async () => {
    loads += 1;
    if (loads === 1) throw new Error("first");
    return loads;
  });
  await assert.rejects(lazy.get(), /first/);
  assert.deepEqual(await Promise.all([lazy.get(), lazy.get()]), [2, 2]);
  assert.equal(await lazy.take(), 2);
  assert.equal(lazy.take(), undefined);
  assert.equal(await lazy.get(), 3);
});
