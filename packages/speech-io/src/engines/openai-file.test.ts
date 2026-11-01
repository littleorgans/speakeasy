import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import { ProviderError, parseTimingObservation } from "@speakeasy/speech-io/providers";
import { decodeWavSamples } from "../bench/wav.ts";
import {
  OPENAI_FILE_LIMITS,
  OPENAI_FILE_MODEL,
  OPENAI_FILE_SAMPLE_RATE,
  OpenAIFileTranscriber,
  type OpenAIFileOptions,
} from "./openai-file.ts";

const FAKE_KEY = "fake-openai-key-for-test";
const TAKE = new Float32Array([0.25, -0.75]);

function json(value: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(value), {
    headers: { "content-type": "application/json", ...headers },
  });
}

function transcriber(fetch: NonNullable<OpenAIFileOptions["fetch"]>, timeoutMs?: number) {
  return new OpenAIFileTranscriber({ fetch, getKey: () => FAKE_KEY, timeoutMs });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function safeFailure(error: unknown): boolean {
  assert.ok(error instanceof ProviderError);
  assert.equal(error.code, "PROVIDER_FAILED");
  assert.equal(error.message, "Speech provider request failed.");
  assert.equal(error.cause, undefined);
  assert.ok(!JSON.stringify(error).includes(FAKE_KEY));
  return true;
}

test("completed take uses native multipart with only file, model and stream=false, preserving exact Float32 bits", async () => {
  const bits = new Uint32Array([
    0, 0x80000000, 0x3eaaaaab, 0xbf800000, 0x40000000, 0xc0a00000,
    1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
  ]);
  const backing = new Float32Array(bits.length + 2);
  new Uint32Array(backing.buffer).set(bits, 1);
  const samples = backing.subarray(1, -1);
  const before = new Uint8Array(backing.buffer).slice();
  let calls = 0;
  const engine = transcriber(async (input, init) => {
    calls += 1;
    assert.equal(input, "https://api.openai.com/v1/audio/transcriptions");
    assert.equal(init?.method, "POST");
    assert.equal(init?.redirect, "error");
    assert.ok(init?.signal instanceof AbortSignal);
    assert.deepEqual([...new Headers(init?.headers)], [["authorization", `Bearer ${FAKE_KEY}`]]);
    assert.ok(init?.body instanceof FormData);
    assert.deepEqual([...init.body.keys()].sort(), ["file", "model", "stream"]);
    assert.equal(init.body.get("model"), OPENAI_FILE_MODEL);
    assert.equal(init.body.get("stream"), "false");
    const file = init.body.get("file");
    assert.ok(file instanceof File);
    assert.equal(file.name, "take.wav");
    assert.equal(file.type, "audio/wav");
    const bytes = new Uint8Array(await file.arrayBuffer());
    const view = new DataView(bytes.buffer);
    assert.equal(new TextDecoder().decode(bytes.subarray(0, 4)), "RIFF");
    assert.equal(new TextDecoder().decode(bytes.subarray(8, 12)), "WAVE");
    assert.equal(view.getUint16(20, true), 3, "IEEE float subtype");
    assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getUint32(24, true), OPENAI_FILE_SAMPLE_RATE);
    assert.equal(view.getUint16(34, true), 32);
    assert.equal(view.getUint32(4, true), bytes.length - 8);
    assert.equal(view.getUint32(40, true), samples.byteLength);
    assert.equal(bytes.length, 44 + samples.byteLength);
    bits.forEach((bit, index) => assert.equal(view.getUint32(44 + index * 4, true), bit));
    const decoded = decodeWavSamples(bytes);
    assert.deepEqual(new Uint32Array(decoded.buffer), bits);
    const wire = new Request(input, init);
    assert.match(wire.headers.get("content-type")!, /^multipart\/form-data; boundary=/);
    const multipart = await wire.formData();
    assert.deepEqual([...multipart.keys()].sort(), ["file", "model", "stream"]);
    return json({ text: "  Original punctuation.\n", languages: [{ code: "en", ignored: "extra" }],
      ignored: { secret: FAKE_KEY } });
  });
  const result = await engine.transcribe(samples, Object.assign({ signal: undefined }, {
    prompt: "EXPECTED TRANSCRIPT MUST NOT BE SENT", keywords: ["hint"], language: "en",
  }));
  assert.equal(result.model, OPENAI_FILE_MODEL);
  assert.equal(result.text, "  Original punctuation.\n");
  assert.deepEqual(result.languages, [{ code: "en" }]);
  assert.ok(!JSON.stringify(result).includes(FAKE_KEY));
  assert.deepEqual(new Uint8Array(backing.buffer), before);
  assert.equal(calls, 1);
});

test("request wall includes full body receipt; processing retains request scope and network stays unavailable", async () => {
  let now = 100;
  const engine = new OpenAIFileTranscriber({
    getKey: () => { now = 200; return FAKE_KEY; },
    clock: () => now,
    fetch: async () => {
      now = 215;
      let reads = 0;
      return new Response(new ReadableStream<Uint8Array>({
        pull(controller) {
          reads += 1;
          now = reads === 1 ? 240 : 280;
          if (reads === 1) controller.enqueue(new TextEncoder().encode('{"text":"done"}'));
          else controller.close();
        },
      }), { headers: { "content-type": "application/json; charset=utf-8", "openai-processing-ms": "120.5" } });
    },
  });
  const { timings } = await engine.transcribe(TAKE);
  const wall = timings.providerRequestWallMs;
  assert.equal(wall.status, "measured");
  if (wall.status === "measured") {
    assert.equal(wall.ms, 80);
    assert.equal(wall.clock, "server");
    assert.match(wall.boundary, /complete response body/);
  }
  const processing = timings.providerProcessingMs;
  assert.equal(processing.status, "provider-reported");
  if (processing.status === "provider-reported") {
    assert.equal(processing.ms, 120.5, "never derive or clamp processing to wall");
    assert.equal(processing.metric, "openai-processing-ms");
    assert.match(processing.boundary, /request processing/);
    assert.ok(processing.evidenceRef);
  }
  assert.equal(timings.providerNetworkMs.status, "unavailable");
  Object.values(timings).forEach((observation) => parseTimingObservation(observation));
});

test("processing is optional, finite and nonnegative, including real zero", async () => {
  for (const header of [undefined, "", " ", "NaN", "Infinity", "-1", "12ms", "0x10", "1e999"]) {
    const result = await transcriber(async () => json({ text: "", languages: [] },
      header === undefined ? {} : { "openai-processing-ms": header })).transcribe(TAKE);
    assert.equal(result.text, "");
    assert.deepEqual(result.languages, []);
    assert.equal(result.timings.providerProcessingMs.status, "unavailable");
  }
  for (const header of ["0", " 0.5 ", "1e2"]) {
    const result = await transcriber(async () => json({ text: "ok" },
      { "openai-processing-ms": header })).transcribe(TAKE);
    assert.equal(result.timings.providerProcessingMs.status, "provider-reported");
  }
});

test("constructor has no transport or credential side effects and credentials are obtained separately for each request", async () => {
  let keys = 0;
  let calls = 0;
  const engine = new OpenAIFileTranscriber({
    getKey: () => `${FAKE_KEY}-${++keys}`,
    fetch: async (_input, init) => {
      calls += 1;
      assert.equal(new Headers(init?.headers).get("authorization"), `Bearer ${FAKE_KEY}-${calls}`);
      return json({ text: "ok" });
    },
  });
  assert.equal(keys, 0);
  assert.equal(calls, 0);
  await engine.transcribe(TAKE);
  await engine.transcribe(TAKE);
  assert.equal(keys, 2);
  assert.equal(calls, 2);
});

test("missing keys and invalid takes fail before dispatch without leaking getter errors", async () => {
  let calls = 0;
  const fetch: typeof globalThis.fetch = async () => { calls += 1; return json({ text: "bad" }); };
  for (const key of [undefined, "", " "]) {
    await assert.rejects(new OpenAIFileTranscriber({ fetch, getKey: () => key }).transcribe(TAKE),
      (error: unknown) => error instanceof ProviderError && error.code === "PROVIDER_UNAVAILABLE");
  }
  const engine = transcriber(fetch);
  for (const take of [new Float32Array(), new Float32Array([NaN]), new Float32Array([Infinity]),
    new Float32Array([-Infinity]), new Float32Array(OPENAI_FILE_LIMITS.fileBytes / 4),
    new Float32Array(OPENAI_FILE_LIMITS.fileBytes / 4 + 1)]) {
    await assert.rejects(engine.transcribe(take), safeFailure);
  }
  await assert.rejects(new OpenAIFileTranscriber({ fetch, getKey: () => {
    throw new Error(`getter leaked ${FAKE_KEY}`);
  } }).transcribe(TAKE), safeFailure);
  assert.equal(calls, 0);
});

test("HTTP and transport failures stay safe, discard bodies and never retry", async () => {
  let calls = 0;
  let cancelledBodies = 0;
  for (const status of [401, 429, 500]) {
    const engine = transcriber(async () => {
      calls += 1;
      return new Response(new ReadableStream({ cancel() { cancelledBodies += 1; } }), { status });
    });
    await assert.rejects(engine.transcribe(TAKE), safeFailure);
  }
  await assert.rejects(transcriber(async () => {
    calls += 1;
    throw new Error(`transport failed with ${FAKE_KEY}`);
  }).transcribe(TAKE), safeFailure);
  assert.equal(calls, 4);
  assert.equal(cancelledBodies, 3);
});

test("response schema and UTF8 are validated and provider error details are discarded", async () => {
  for (const value of [null, [], {}, { text: null }, { text: 1 }, { error: { message: FAKE_KEY } },
    { text: "ok", languages: null }, { text: "ok", languages: [null] },
    { text: "ok", languages: [{ code: 1 }] }, { text: "ok", languages: [{ code: "" }] },
    { text: "ok", languages: Array.from({ length: 33 }, () => ({ code: "en" })) },
    { text: "é".repeat(OPENAI_FILE_LIMITS.textBytes / 2 + 1) }]) {
    await assert.rejects(transcriber(async () => json(value)).transcribe(TAKE), safeFailure);
  }
  for (const body of ['{"text":', new Uint8Array([0xff])]) {
    await assert.rejects(transcriber(async () => new Response(body,
      { headers: { "content-type": "application/json" } })).transcribe(TAKE), safeFailure);
  }
  await assert.rejects(transcriber(async () => new Response(`data: ${FAKE_KEY}\n\n`,
    { headers: { "content-type": "text/event-stream" } })).transcribe(TAKE), safeFailure);
});

test("response size is checked from both headers and streamed bytes, including a lying content length", async () => {
  let cancelledBodies = 0;
  for (const length of [String(OPENAI_FILE_LIMITS.responseBytes + 1), "invalid", "1", undefined]) {
    const engine = transcriber(async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new Uint8Array(OPENAI_FILE_LIMITS.responseBytes + 1)); },
      cancel() { cancelledBodies += 1; },
    }), { headers: { "content-type": "application/json",
      ...(length === undefined ? {} : { "content-length": length }) } }));
    await assert.rejects(engine.transcribe(TAKE), safeFailure);
  }
  assert.equal(cancelledBodies, 4);
});

test("JSON and multibyte UTF8 can span body chunks, and a broken body never returns partial success", async () => {
  const bytes = new TextEncoder().encode(JSON.stringify({ text: "界 café 🎤" }));
  const result = await transcriber(async () => new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  }), { headers: { "content-type": "application/json" } })).transcribe(TAKE);
  assert.equal(result.text, "界 café 🎤");
  const broken = new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(new TextEncoder().encode('{"text":"partial"}')); },
    pull(controller) { controller.error(new Error(FAKE_KEY)); },
  });
  await assert.rejects(transcriber(async () => new Response(broken,
    { headers: { "content-type": "application/json" } })).transcribe(TAKE), safeFailure);
  assert.equal(broken.locked, false);
});

test("already cancelled calls never read keys or dispatch and discard the caller's abort reason", async () => {
  let calls = 0;
  const engine = new OpenAIFileTranscriber({
    getKey: () => { calls += 1; return FAKE_KEY; },
    fetch: async () => { calls += 1; return json({ text: "late" }); },
  });
  await assert.rejects(engine.transcribe(TAKE, { signal: AbortSignal.abort(new Error(FAKE_KEY)) }),
    (error: unknown) => error instanceof DOMException && error.name === "AbortError" &&
      !error.message.includes(FAKE_KEY));
  assert.equal(calls, 0);
});

test("cancellation settles even if fetch ignores abort, and cancels a late response body", async () => {
  const pending = deferred<Response>();
  let transportSignal: AbortSignal | undefined;
  let cancelledBodies = 0;
  const engine = transcriber(async (_input, init) => {
    transportSignal = init?.signal ?? undefined;
    return pending.promise;
  });
  const controller = new AbortController();
  const result = engine.transcribe(TAKE, { signal: controller.signal });
  controller.abort(new Error(FAKE_KEY));
  await assert.rejects(result, { name: "AbortError" });
  assert.equal(transportSignal?.aborted, true);
  pending.resolve(new Response(new ReadableStream({ cancel() { cancelledBodies += 1; } })));
  await setImmediate();
  assert.equal(cancelledBodies, 1);
});

test("cancellation during body receipt drops partial text and releases the reader", async () => {
  const controller = new AbortController();
  const reading = deferred<void>();
  let cancelledBodies = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(stream) {
      stream.enqueue(new TextEncoder().encode('{"text":"partial'));
      reading.resolve();
      return new Promise<void>(() => {});
    },
    cancel() { cancelledBodies += 1; },
  });
  const result = transcriber(async () => new Response(body,
    { headers: { "content-type": "application/json" } })).transcribe(TAKE, { signal: controller.signal });
  await reading.promise;
  await setImmediate();
  controller.abort(FAKE_KEY);
  await assert.rejects(result, { name: "AbortError" });
  assert.equal(cancelledBodies, 1);
  assert.equal(body.locked, false);
});

test("one deadline covers header wait and body wait, even for transports ignoring abort", async () => {
  let transportSignal: AbortSignal | undefined;
  await assert.rejects(transcriber(async (_input, init) => {
    transportSignal = init?.signal ?? undefined;
    return new Promise<Response>(() => {});
  }, 10).transcribe(TAKE), { name: "TimeoutError" });
  assert.equal(transportSignal?.aborted, true);
  let cancelledBodies = 0;
  const body = new ReadableStream<Uint8Array>({ cancel() { cancelledBodies += 1; } });
  await assert.rejects(transcriber(async () => new Response(body,
    { headers: { "content-type": "application/json" } }), 10).transcribe(TAKE), { name: "TimeoutError" });
  assert.equal(cancelledBodies, 1);
  assert.equal(body.locked, false);
});
