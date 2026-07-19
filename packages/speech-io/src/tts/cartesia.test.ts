import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { test } from "node:test";
import {
  CartesiaTextToSpeech,
  CARTESIA_VOICES,
  DEFAULT_CARTESIA_VOICE,
  resolveCartesiaVoice,
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
  assert.throws(() => resolveCartesiaVoice("nope"), /unknown Cartesia voice/);
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
  assert.equal(headers["cartesia-version"], "2025-04-16");
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
      assert.match(error.message, /HTTP 400/);
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
  await assert.rejects(() => collect(session.speak("Hi.")), /CARTESIA_API_KEY is not set/);
});

// Gated live smoke: only runs with a real key. Prints timing, never audio.
test(
  "live: synthesizes a sentence to real PCM",
  { skip: !process.env.CARTESIA_API_KEY },
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
