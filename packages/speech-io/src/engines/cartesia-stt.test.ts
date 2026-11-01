import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { ProviderError } from "@speakeasy/speech-io/providers";
import {
  CARTESIA_STT_MAX_SAMPLES,
  CARTESIA_STT_MAX_RESPONSE_BYTES,
  CartesiaBatchTranscriber,
  type CartesiaSttFetch,
} from "./cartesia-stt.ts";

const FAKE_KEY = "fake-cartesia-test-key";
const pcm = new Float32Array([0, -0, 0.25, -0.5, 1, -1]);
const reply = (extra: Record<string, unknown> = {}) =>
  Response.json({ type: "transcript", text: "  Hello there.\n", ...extra });

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function safeFailure(error: unknown): boolean {
  assert.ok(error instanceof ProviderError);
  assert.equal(error.code, "PROVIDER_FAILED");
  assert.doesNotMatch(JSON.stringify(error), /fake-cartesia-test-key|reflected-private-text/);
  assert.equal(error.cause, undefined);
  return true;
}

test("construction and metadata do not consult keys, clock or transport", () => {
  const unexpected = (): never => { throw new Error("unexpected external work"); };
  const adapter = new CartesiaBatchTranscriber({
    getKey: unexpected, fetch: unexpected, clock: unexpected,
  });
  const metadata = adapter.metadata();
  assert.deepEqual(metadata.config, {
    schema: 1, provider: { kind: "cartesia", model: "ink-whisper" },
  });
  assert.equal(metadata.sampleRate, 16_000);
  assert.equal(metadata.encoding, "pcm_f32le");
  assert.equal(metadata.apiVersion, "2026-08-14");
  assert.equal(metadata.maxSamples, 1_920_000);
  metadata.config.provider = { kind: "openai", model: "mutated" };
  assert.equal(adapter.metadata().config.provider.kind, "cartesia");
});

test("batch uploads only the exact original little endian samples and documented fields", async () => {
  const backing = new Float32Array([99, ...pcm, 99]);
  const samples = backing.subarray(1, backing.length - 1);
  const original = samples.slice();
  let keyReads = 0;
  let calls = 0;
  let clock = 100;
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { keyReads += 1; return FAKE_KEY; },
    clock: () => clock,
    fetch: async (url, init) => {
      calls += 1;
      const target = new URL(url);
      assert.equal(target.origin + target.pathname, "https://api.cartesia.ai/stt");
      assert.deepEqual([...target.searchParams], [["encoding", "pcm_f32le"], ["sample_rate", "16000"]]);
      assert.equal(init.method, "POST");
      assert.equal(init.redirect, "error");
      const headers = new Headers(init.headers);
      assert.equal(headers.get("authorization"), `Bearer ${FAKE_KEY}`);
      assert.equal(headers.get("cartesia-version"), "2026-08-14");
      assert.equal(headers.has("content-type"), false);
      assert.ok(init.signal instanceof AbortSignal);
      assert.ok(init.body instanceof FormData);
      assert.deepEqual([...init.body.keys()].sort(), ["file", "language", "model"]);
      assert.equal(init.body.get("model"), "ink-whisper");
      assert.equal(init.body.get("language"), "en");
      const file = init.body.get("file");
      assert.ok(file instanceof Blob);
      const bytes = new Uint8Array(await file.arrayBuffer());
      assert.equal(bytes.length, original.length * 4);
      const view = new DataView(bytes.buffer);
      original.forEach((sample, index) => assert.ok(Object.is(view.getFloat32(index * 4, true), sample)));
      samples.fill(0.75); // The uploaded snapshot and its identity must stay stable.
      clock = 155;
      return reply({ request_id: "request-1", language: "en", duration: 0.000375,
        processing_ms: 4, network_ms: 7, decodeMs: 8 });
    },
  });
  const result = await adapter.transcribe(samples);
  assert.equal(calls, 1);
  assert.equal(keyReads, 1);
  assert.equal(result.text, "  Hello there.\n");
  assert.equal(result.sampleCount, original.length);
  assert.equal(result.pcmBytes, original.byteLength);
  const bytes = Buffer.alloc(original.byteLength);
  original.forEach((sample, index) => bytes.writeFloatLE(sample, index * 4));
  assert.equal(result.pcmSha256, createHash("sha256").update(bytes).digest("hex"));
  assert.equal(result.sampleRate, 16_000);
  assert.equal(result.channels, 1);
  assert.deepEqual(result.observations, {
    requestId: "request-1", language: "en", audioDurationSeconds: 0.000375,
  });
  assert.equal(result.timings.providerRequestWallMs.status, "measured");
  if (result.timings.providerRequestWallMs.status === "measured") {
    assert.equal(result.timings.providerRequestWallMs.ms, 55);
    assert.match(result.timings.providerRequestWallMs.boundary, /dispatch.*body/i);
  }
  assert.equal(result.timings.providerProcessingMs.status, "unavailable");
  assert.equal(result.timings.providerNetworkMs.status, "unavailable");
  assert.equal("decodeMs" in result, false);
});

test("request wall includes body receipt after response headers", async () => {
  let clock = 10;
  const encoder = new TextEncoder();
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => FAKE_KEY, clock: () => clock,
    fetch: async () => {
      clock = 20;
      return new Response(new ReadableStream({
        pull(controller) {
          clock = 80;
          controller.enqueue(encoder.encode('{"type":"transcript","text":""}'));
          controller.close();
        },
      }));
    },
  });
  const result = await adapter.transcribe(pcm);
  assert.equal(result.text, "");
  assert.deepEqual(result.observations, {});
  assert.equal(result.timings.providerRequestWallMs.status === "measured" &&
    result.timings.providerRequestWallMs.ms, 70);
});

test("invalid or oversized PCM fails before key lookup and dispatch", async () => {
  let externalCalls = 0;
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { externalCalls += 1; return FAKE_KEY; },
    fetch: async () => { externalCalls += 1; return reply(); },
  });
  for (const samples of [new Float32Array(), new Float32Array([NaN]),
    new Float32Array([Infinity]), new Float32Array([-Infinity]),
    new Float32Array(CARTESIA_STT_MAX_SAMPLES + 1)]) {
    await assert.rejects(adapter.transcribe(samples), RangeError);
  }
  await assert.rejects(adapter.transcribe(new Uint8Array([1]) as unknown as Float32Array), RangeError);
  assert.equal(externalCalls, 0);
});

test("finite Float32 values are preserved without clipping or padding", async () => {
  const samples = new Float32Array([2, -2, 1e-40]);
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => FAKE_KEY,
    fetch: async (_url, init) => {
      const file = (init.body as FormData).get("file") as Blob;
      assert.deepEqual(new Float32Array(await file.arrayBuffer()), samples);
      return reply();
    },
  });
  assert.equal((await adapter.transcribe(samples)).sampleCount, 3);
});

test("credentials are resolved per request; missing keys never dispatch", async () => {
  let key: string | undefined;
  let calls = 0;
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => key,
    fetch: async (_url, init) => {
      calls += 1;
      assert.equal(new Headers(init.headers).get("authorization"), `Bearer ${key}`);
      return reply();
    },
  });
  await assert.rejects(adapter.transcribe(pcm), (error: unknown) =>
    error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
  assert.equal(calls, 0);
  key = FAKE_KEY;
  await adapter.transcribe(pcm);
  key = "another-fake-key";
  await adapter.transcribe(pcm);
  assert.equal(calls, 2);
});

test("HTTP, transport, credential and malformed JSON failures are safe and never retried", async () => {
  const reflected = `${FAKE_KEY} reflected-private-text`;
  const failures: CartesiaSttFetch[] = [
    async () => new Response(reflected, { status: 401 }),
    async () => Response.json({ type: "error", message: reflected }),
    async () => { throw new Error(reflected); },
    async () => new Response(`invalid JSON ${reflected}`),
    async () => Response.json(null),
    async () => Response.json({ type: "transcript", text: 42, message: reflected }),
    async () => Response.json({ text: "missing type" }),
  ];
  for (const fetch of failures) {
    let calls = 0;
    const adapter = new CartesiaBatchTranscriber({
      getKey: () => FAKE_KEY,
      fetch: (url, init) => { calls += 1; return fetch(url, init); },
    });
    await assert.rejects(adapter.transcribe(pcm), safeFailure);
    assert.equal(calls, 1);
  }
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { throw new Error(reflected); },
    fetch: async () => { assert.fail("must not dispatch"); },
  });
  await assert.rejects(adapter.transcribe(pcm), safeFailure);
});

test("response size is bounded and malformed optional observations are rejected", async () => {
  let cancelled = false;
  const oversized = new Response(new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(CARTESIA_STT_MAX_RESPONSE_BYTES + 1)); },
    cancel() { cancelled = true; },
  }));
  const responses = [oversized, reply({ duration: -1 }), reply({ duration: "12" }),
    reply({ request_id: { secret: FAKE_KEY } }), reply({ language: ["en"] }),
    new Response(new Uint8Array([0xff, 0xfe]))];
  for (const response of responses) {
    const adapter = new CartesiaBatchTranscriber({ getKey: () => FAKE_KEY, fetch: async () => response });
    await assert.rejects(adapter.transcribe(pcm), safeFailure);
  }
  assert.equal(cancelled, true);
});

test("pre cancellation skips key lookup and cancellation drops late transport results", async () => {
  const parent = new AbortController();
  const pending = deferred<Response>();
  const observed: { signal?: AbortSignal | null } = {};
  let keyReads = 0;
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { keyReads += 1; return FAKE_KEY; },
    fetch: async (_url, init) => { observed.signal = init.signal; return pending.promise; },
  });
  parent.abort(new Error(`${FAKE_KEY} reflected-private-text`));
  await assert.rejects(adapter.transcribe(pcm, { signal: parent.signal }),
    (error: Error) => error.name === "AbortError" && !error.message.includes(FAKE_KEY));
  assert.equal(keyReads, 0);
  const active = new AbortController();
  const result = adapter.transcribe(pcm, { signal: active.signal });
  active.abort(new Error(FAKE_KEY));
  await assert.rejects(result, (error: Error) => error.name === "AbortError" && !error.message.includes(FAKE_KEY));
  assert.equal(observed.signal?.aborted, true);
  pending.resolve(reply());
});

test("cancelling during body read cancels its stream and ignores unsafe abort reasons", async () => {
  const reading = deferred<void>();
  let cancelled = false;
  const parent = new AbortController();
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => FAKE_KEY,
    fetch: async () => new Response(new ReadableStream({
      pull() { reading.resolve(); },
      cancel() { cancelled = true; },
    }, { highWaterMark: 0 })),
  });
  const result = adapter.transcribe(pcm, { signal: parent.signal });
  await reading.promise;
  parent.abort(new Error(FAKE_KEY));
  await assert.rejects(result, (error: Error) => error.name === "AbortError" && !error.message.includes(FAKE_KEY));
  assert.equal(cancelled, true);
});

test("cancellation during response handoff cancels the body before a reader is acquired", async () => {
  const pending = deferred<Response>();
  const parent = new AbortController();
  let cancelledBodies = 0;
  const body = new ReadableStream<Uint8Array>({ cancel() { cancelledBodies += 1; } });
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => FAKE_KEY, fetch: () => pending.promise,
  });
  const result = adapter.transcribe(pcm, { signal: parent.signal });
  pending.resolve(new Response(body));
  queueMicrotask(() => parent.abort(new Error(FAKE_KEY)));
  await assert.rejects(result, (error: Error) =>
    error.name === "AbortError" && !error.message.includes(FAKE_KEY));
  assert.equal(cancelledBodies, 1);
  assert.equal(body.locked, false);
});

test("timeout aborts a non cooperative fetch and safely consumes a late rejection", async () => {
  const pending = deferred<Response>();
  const observed: { signal?: AbortSignal | null } = {};
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => FAKE_KEY, timeoutMs: 10,
    fetch: async (_url, init) => { observed.signal = init.signal; return pending.promise; },
  });
  await assert.rejects(adapter.transcribe(pcm), (error: Error) => error.name === "TimeoutError");
  assert.equal(observed.signal?.aborted, true);
  pending.reject(new Error(`${FAKE_KEY} reflected-private-text`));
});

test("policy key getter may revoke dispatch using shared safe provider errors", async () => {
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { throw new ProviderError("SELECTION_STALE"); },
    fetch: async () => { assert.fail("revoked selection must not dispatch"); },
  });
  await assert.rejects(adapter.transcribe(pcm), (error: unknown) =>
    error instanceof ProviderError && error.code === "SELECTION_STALE");
});

test("revocation during call time credential recheck prevents transmission", async () => {
  const parent = new AbortController();
  const adapter = new CartesiaBatchTranscriber({
    getKey: () => { parent.abort(new Error(FAKE_KEY)); return FAKE_KEY; },
    fetch: async () => { assert.fail("cancelled dispatch must not transmit"); },
  });
  await assert.rejects(adapter.transcribe(pcm, { signal: parent.signal }),
    (error: Error) => error.name === "AbortError" && !error.message.includes(FAKE_KEY));
});

test("invalid clocks never create a fabricated timing observation", async () => {
  for (const end of [NaN, Infinity, 5]) {
    let calls = 0;
    const adapter = new CartesiaBatchTranscriber({
      getKey: () => FAKE_KEY, fetch: async () => reply(),
      clock: () => calls++ === 0 ? 10 : end,
    });
    await assert.rejects(adapter.transcribe(pcm), (error: unknown) =>
      error instanceof ProviderError && error.code === "TIMING_INVALID");
  }
});
