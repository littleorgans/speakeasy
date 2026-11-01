import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { test } from "node:test";
import { ProviderError } from "@speakeasy/speech-io/providers";
import {
  CARTESIA_BROWSER_MODEL, CARTESIA_SAMPLE_RATE, CARTESIA_SPEED,
} from "@speakeasy/speech-io";
import {
  CartesiaTextToSpeech,
  CARTESIA_VOICES,
  DEFAULT_CARTESIA_VOICE,
  resolveCartesiaVoice,
  type CartesiaOptions,
  type FetchLike,
} from "./cartesia.ts";
import type { AudioSegment } from "./contract.ts";

async function collect(stream: AsyncIterable<AudioSegment>): Promise<AudioSegment[]> {
  const out: AudioSegment[] = [];
  for await (const segment of stream) {
    out.push(segment);
  }
  return out;
}

function responseWithArrayBuffer(arrayBuffer: Response["arrayBuffer"]): Response {
  const response = new Response();
  Object.defineProperty(response, "arrayBuffer", { value: arrayBuffer });
  return response;
}

/** A fetch stub returning raw f32le PCM, capturing the request it saw. */
function stubFetch(
  samples: Float32Array,
  capture?: (url: string, init: RequestInit) => void,
): FetchLike {
  return async (url, init) => {
    capture?.(url, init);
    return new Response(samples.buffer as ArrayBuffer, { status: 200 });
  };
}

test("resolveCartesiaVoice maps names, passes UUIDs, rejects garbage", () => {
  assert.equal(resolveCartesiaVoice("katie"), CARTESIA_VOICES.katie);
  assert.equal(resolveCartesiaVoice("KATIE"), CARTESIA_VOICES.katie);
  const uuid = "f786b574-daa5-4673-aa0c-cbe3e8534c02";
  assert.equal(resolveCartesiaVoice(uuid), uuid);
  for (const voice of ["nope", "constructor", "toString", "__proto__"]) {
    assert.throws(() => resolveCartesiaVoice(voice), { code: "CONFIG_UNSUPPORTED" });
  }
});

test("speak() posts the right request and decodes pcm_f32le samples", async () => {
  let seenUrl = "";
  let seenInit: RequestInit = {};
  const pcm = new Float32Array([0.1, -0.2, 0.3, -0.4]);
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "test-key",
    fetch: stubFetch(pcm, (url, init) => {
      seenUrl = url;
      seenInit = init;
    }),
  });
  const session = await tts.open({ voice: "daniel" });
  const segments = await collect(session.speak("Hello there."));

  assert.match(seenUrl, /\/tts\/bytes$/);
  const headers = seenInit.headers as Record<string, string>;
  assert.equal(headers.authorization, "Bearer test-key");
  assert.equal(headers["cartesia-version"], "2026-08-14");
  const body = JSON.parse(String(seenInit.body));
  assert.equal(body.transcript, "Hello there.");
  assert.equal(body.voice.id, CARTESIA_VOICES.daniel);
  assert.equal(body.output_format.encoding, "pcm_f32le");
  assert.equal(body.output_format.sample_rate, 44100);

  assert.equal(segments.length, 1);
  assert.equal(segments[0]?.sampleRate, 44100);
  assert.ok(segments[0]!.samples.length > 0);
  await session.close();
});

test("open() defaults to the built-in voice", async () => {
  let seenBody = "";
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "k",
    fetch: stubFetch(new Float32Array([0.1]), (_url, init) => {
      seenBody = String(init.body);
    }),
  });
  const session = await tts.open();
  await collect(session.speak("Hi."));
  assert.equal(JSON.parse(seenBody).voice.id, CARTESIA_VOICES[DEFAULT_CARTESIA_VOICE]);
});

test("synth throws a redacted error on HTTP failure", async () => {
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "secret-key-value",
    fetch: async () => new Response("bad voice", { status: 400 }),
  });
  const session = await tts.open();
  await assert.rejects(
    () => collect(session.speak("Hi.")),
    (error: Error) => {
      assert.ok(error instanceof ProviderError);
      assert.equal(error.code, "PROVIDER_FAILED");
      assert.doesNotMatch(error.message, /secret-key-value/);
      return true;
    },
  );
});

test("synth throws when no API key is available", async () => {
  const tts = new CartesiaTextToSpeech({
    apiKey: () => undefined,
    fetch: stubFetch(new Float32Array([0.1])),
  });
  const session = await tts.open();
  await assert.rejects(() => collect(session.speak("Hi.")), { code: "PROVIDER_UNAVAILABLE" });
});

test("forwards the dated model and speed with the current voice payload", async () => {
  let body: Record<string, unknown> = {};
  let version: string | undefined;
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    fetch: stubFetch(new Float32Array([0.2]), (_url, init) => {
      body = JSON.parse(String(init.body));
      version = (init.headers as Record<string, string>)["cartesia-version"];
    }),
  });
  for (const speed of [CARTESIA_SPEED.min, 1.25, CARTESIA_SPEED.max]) {
    const session = await tts.open({ model: CARTESIA_BROWSER_MODEL, speed });
    const segments = await collect(session.speak("Hello."));
    assert.equal(version, "2026-08-14");
    assert.deepEqual(body.voice, { id: CARTESIA_VOICES.katie });
    assert.deepEqual(body.generation_config, { speed });
    assert.equal(body.model_id, "sonic-3.6-2026-08-27");
    assert.equal((body.output_format as Record<string, unknown>).sample_rate, CARTESIA_SAMPLE_RATE);
    assert.equal(segments[0]!.sampleRate, CARTESIA_SAMPLE_RATE);
    await session.close();
  }
});

test("rejects numeric voices, unsupported models and invalid speeds before dispatch", async () => {
  let calls = 0;
  const tts = new CartesiaTextToSpeech({
    apiKey: () => { calls += 1; return "fake-key"; },
    fetch: stubFetch(new Float32Array([0.1])),
  });
  for (const config of [
    { voice: 0 }, { voice: "constructor" }, { model: "sonic-2" },
    { model: "" }, { speed: Number.NaN }, { speed: 0.5 }, { speed: 1.6 },
    { speed: CARTESIA_SPEED.min - 0.001 }, { speed: CARTESIA_SPEED.max + 0.001 },
  ]) {
    await assert.rejects(() => tts.open(config));
  }
  assert.equal(calls, 0);
});

test("consumer break prevents speculative subsequent sentence transmission", async () => {
  let calls = 0;
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    fetch: stubFetch(new Float32Array([0.1]), () => { calls += 1; }),
  });
  const session = await tts.open();
  for await (const _segment of session.speak("First sentence. Second sentence.")) break;
  assert.equal(calls, 1);
  await session.close();
});

test("close aborts the outstanding fetch and drops late PCM", { timeout: 1000 }, async () => {
  let started!: () => void;
  const dispatched = new Promise<void>((resolve) => { started = resolve; });
  let finish!: (response: Response) => void;
  let signal: AbortSignal | null | undefined;
  let calls = 0;
  const tts = new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    fetch: async (_url, init) => {
      calls += 1;
      if (calls > 1) return new Response(new Float32Array([0.2]).buffer);
      signal = init.signal;
      started();
      return new Promise<Response>((resolve) => { finish = resolve; });
    },
  });
  const session = await tts.open();
  const pending = collect(session.speak("First sentence. Second sentence."));
  await dispatched;
  await session.close();
  const aborted = signal?.aborted;
  await assert.rejects(() => pending, { name: "AbortError" });
  finish(new Response(new Float32Array([0.2]).buffer));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(aborted, true);
  assert.equal(calls, 1);
});

test("provider errors never reflect response bodies or transport exceptions", async () => {
  const sensitive = "fake-key private transcript fake-endpoint";
  for (const fetch of [
    async () => new Response(sensitive.repeat(100), { status: 400 }),
    async () => { throw new Error(sensitive); },
  ]) {
    const session = await new CartesiaTextToSpeech({ apiKey: () => "fake-key", fetch }).open();
    await assert.rejects(() => collect(session.speak("Hello.")), (error: Error) => {
      assert.doesNotMatch(error.message, /fake-key|private transcript|fake-endpoint/);
      assert.ok(error.message.length < 200);
      assert.equal(error.cause, undefined);
      return true;
    });
  }
});

test("explicit legacy API version retains its voice payload and existing model default", async () => {
  let body: Record<string, unknown> = {};
  let version: string | undefined;
  const session = await new CartesiaTextToSpeech({
    apiVersion: "2025-04-16",
    apiKey: () => "fake-key",
    fetch: stubFetch(new Float32Array([0.1]), (_url, init) => {
      body = JSON.parse(String(init.body));
      version = (init.headers as Record<string, string>)["cartesia-version"];
    }),
  }).open();
  await collect(session.speak("Hello."));
  assert.equal(version, "2025-04-16");
  assert.equal(body.model_id, "sonic-3.5");
  assert.deepEqual(body.voice, { mode: "id", id: CARTESIA_VOICES.katie });
});

test("constructor and open do not read keys, run policy checks or call transport", async () => {
  let calls = 0;
  const touched = () => { calls += 1; };
  await new CartesiaTextToSpeech({
    apiKey: () => { touched(); return "fake-key"; },
    fetch: async () => { touched(); throw new Error("unexpected transport"); },
    beforeRequest: touched,
    now: () => { touched(); return 0; },
  }).open();
  assert.equal(calls, 0);
});

test("fixed and incremental text use the shared sentence plan with per request keys", async () => {
  async function* tokens() { yield "First sentence. "; yield "Second sentence."; }
  for (const text of ["First sentence. Second sentence.", tokens()]) {
    const sent: string[] = [];
    const headers: string[] = [];
    let keys = 0;
    const session = await new CartesiaTextToSpeech({
      apiKey: () => `fake-key-${++keys}`,
      fetch: stubFetch(new Float32Array([0.2]), (_url, init) => {
        sent.push(JSON.parse(String(init.body)).transcript);
        headers.push((init.headers as Record<string, string>).authorization!);
      }),
    }).open();
    assert.equal((await collect(session.speak(text))).length, 2);
    assert.deepEqual(sent, ["First sentence.", "Second sentence."]);
    assert.deepEqual(headers, ["Bearer fake-key-1", "Bearer fake-key-2"]);
    await session.close();
  }
});

test("request wall includes complete body receipt and excludes policy and callbacks", async () => {
  let clock = 0;
  const completed: Parameters<NonNullable<CartesiaOptions["onRequestComplete"]>>[0][] = [];
  const session = await new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    beforeRequest: () => { clock += 100; },
    now: () => clock,
    fetch: async () => {
      clock += 25;
      return responseWithArrayBuffer(async () => {
        clock += 50;
        return new Float32Array([0.2]).buffer;
      });
    },
    onRequestComplete: (result) => { completed.push(result); clock += 200; },
  }).open({ model: "sonic-3.6-2026-08-27", voice: "DANIEL" });
  const [segment] = await collect(session.speak("Hello."));
  assert.equal(segment?.synthMs, 75);
  assert.equal(segment?.readyAtMs, 375);
  assert.deepEqual(completed, [{
    model: "sonic-3.6-2026-08-27", voice: CARTESIA_VOICES.daniel, text: "Hello.",
    timing: {
      providerRequestWallMs: { status: "measured", ms: 75, clock: "server",
        boundary: "HTTP dispatch to complete response body receipt" },
      providerProcessingMs: { status: "unavailable", reason: "Cartesia bytes exposes no processing metric" },
      providerNetworkMs: { status: "unavailable", reason: "Cartesia bytes exposes no network boundary" },
    },
  }]);
});

test("policy is checked before keys and again before subsequent sentences", async () => {
  for (const allowFirst of [false, true]) {
    let allowed = allowFirst;
    let keys = 0;
    let calls = 0;
    const session = await new CartesiaTextToSpeech({
      apiKey: () => { keys += 1; return "fake-key"; },
      beforeRequest: () => { if (!allowed) throw new ProviderError("SELECTION_STALE"); },
      fetch: stubFetch(new Float32Array([0.1]), () => { calls += 1; }),
    }).open();
    const iterator = session.speak("First sentence. Second sentence.")[Symbol.asyncIterator]();
    if (allowFirst) {
      assert.equal((await iterator.next()).done, false);
      assert.equal(calls, 1);
      allowed = false;
    }
    await assert.rejects(() => iterator.next(), { code: "SELECTION_STALE" });
    assert.equal(keys, Number(allowFirst));
    assert.equal(calls, Number(allowFirst));
  }
});

test("external abort cancels body buffering promptly and suppresses late callbacks", { timeout: 1000 }, async () => {
  const controller = new AbortController();
  let started!: () => void;
  const buffering = new Promise<void>((resolve) => { started = resolve; });
  let finish!: (buffer: ArrayBuffer) => void;
  let signal: AbortSignal | null | undefined;
  let calls = 0;
  let completions = 0;
  const session = await new CartesiaTextToSpeech({
    signal: controller.signal,
    apiKey: () => "fake-key",
    onRequestComplete: () => { completions += 1; },
    fetch: async (_url, init) => {
      calls += 1;
      signal = init.signal;
      return responseWithArrayBuffer(() => {
        started();
        return new Promise<ArrayBuffer>((resolve) => { finish = resolve; });
      });
    },
  }).open();
  const pending = collect(session.speak("First sentence. Second sentence."));
  await buffering;
  controller.abort("private abort reason");
  await assert.rejects(() => pending, { name: "AbortError", message: "Cartesia speech cancelled" });
  assert.equal(signal?.aborted, true);
  finish(new Float32Array([0.1]).buffer);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(calls, 1);
  assert.equal(completions, 0);
});

test("closed or preaborted sessions never dispatch or read keys", async () => {
  for (const closed of [false, true]) {
    let calls = 0;
    const controller = new AbortController();
    const session = await new CartesiaTextToSpeech({
      signal: controller.signal,
      apiKey: () => { calls += 1; return "fake-key"; },
      fetch: stubFetch(new Float32Array([0.1]), () => { calls += 1; }),
    }).open();
    if (closed) { await session.close(); await session.close(); }
    else controller.abort();
    await assert.rejects(() => collect(session.speak("Hello.")), { name: "AbortError" });
    assert.equal(calls, 0);
  }
});

test("malformed, empty and nonfinite PCM fail safely without completing", async () => {
  const buffers = [new ArrayBuffer(0), new ArrayBuffer(3), new Float32Array([Number.NaN]).buffer];
  for (const buffer of buffers) {
    let completed = false;
    const session = await new CartesiaTextToSpeech({
      apiKey: () => "fake-key",
      fetch: async () => new Response(buffer),
      onRequestComplete: () => { completed = true; },
    }).open();
    await assert.rejects(() => collect(session.speak("Hello.")), { code: "PROVIDER_FAILED" });
    assert.equal(completed, false);
  }
});

test("close while a consumer holds a segment cancels prefetch without dispatch", async () => {
  let calls = 0;
  const session = await new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    fetch: stubFetch(new Float32Array([0.1]), () => { calls += 1; }),
  }).open();
  const iterator = session.speak("First sentence. Second sentence.")[Symbol.asyncIterator]();
  assert.equal((await iterator.next()).done, false);
  await session.close();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(calls, 1);
  await assert.rejects(() => iterator.next(), { name: "AbortError" });
});

test("aborting a pending policy check prevents key access and dispatch", { timeout: 1000 }, async () => {
  let keys = 0;
  let begin!: () => void;
  const started = new Promise<void>((resolve) => { begin = resolve; });
  let finish!: () => void;
  const session = await new CartesiaTextToSpeech({
    beforeRequest: () => { begin(); return new Promise<void>((resolve) => { finish = resolve; }); },
    apiKey: () => { keys += 1; return "fake-key"; },
    fetch: async () => { throw new Error("unexpected transport"); },
  }).open();
  const pending = collect(session.speak("Hello."));
  await started;
  await session.close();
  await assert.rejects(() => pending, { name: "AbortError" });
  finish();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(keys, 0);
});

test("completed iterators stay complete and sessions support subsequent turns", async () => {
  const session = await new CartesiaTextToSpeech({
    apiKey: () => "fake-key",
    fetch: stubFetch(new Float32Array([0.2])),
  }).open();
  for (const text of ["Hello.", "Next turn."]) {
    const iterator = session.speak(text)[Symbol.asyncIterator]();
    assert.equal((await iterator.next()).done, false);
    assert.equal((await iterator.next()).done, true);
    assert.equal((await iterator.next()).done, true);
  }
  await session.close();
});

// Explicit live smoke. An ambient key alone must never spend provider quota.
test(
  "live: synthesizes a sentence to real PCM",
  {
    skip:
      process.env.SPEAKEASY_LIVE_CARTESIA !== "1" ||
      !process.env.CARTESIA_API_KEY,
  },
  async () => {
    const tts = new CartesiaTextToSpeech();
    const session = await tts.open({ voice: DEFAULT_CARTESIA_VOICE });
    const start = performance.now();
    const segments = await collect(session.speak("This is a Cartesia smoke test."));
    const ms = performance.now() - start;
    assert.ok(segments.length >= 1);
    assert.ok(segments[0]!.samples.length > 0);
    assert.equal(segments[0]!.sampleRate, 44100);
    console.log(
      `live cartesia: ${segments.length} seg/${segments.reduce((n, s) => n + s.samples.length, 0)} samples in ${ms.toFixed(1)}ms`,
    );
    await session.close();
  },
);
