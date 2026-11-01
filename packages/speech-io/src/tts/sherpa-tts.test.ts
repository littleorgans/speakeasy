import assert from "node:assert/strict";
import { test } from "node:test";
import { SherpaTextToSpeech } from "./sherpa-tts.ts";
import {
  planSegments,
  planSegmentsStream,
  synthPipeline,
  type SegmentSynth,
} from "./stream.ts";
import type { SynthResult } from "./synth.ts";

/** A synth that fabricates fixed audio, so the pipeline runs without a model. */
function fakeSynth(): SegmentSynth {
  return {
    async synth(): Promise<SynthResult> {
      const samples = new Float32Array(1600).fill(0.5); // 100ms @ 16kHz
      return {
        samples,
        sampleRate: 16000,
        firstAudioMs: undefined,
        totalSynthMs: 1,
        audioDurationMs: 100,
        rtf: 0.01,
      };
    },
  };
}

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of stream) {
    out.push(item);
  }
  return out;
}

/** Emit a string one delta at a time (worst case for the streaming segmenter). */
async function* charByChar(text: string): AsyncGenerator<string> {
  for (const ch of text) {
    yield ch;
  }
}

async function* once(text: string): AsyncGenerator<string> {
  yield text;
}

const CASES = [
  "",
  "Hi.",
  "promote it to staging",
  "Go to sleep.",
  "one two three four.",
  "one two three four five.",
  "Hello, world. Bye.",
  "The build finished. Two checks are running! Say the word?",
  "Spawn ten agents, then close the pane. Done.",
  "Well— that escalated. Quickly, too.",
];

test("planSegmentsStream on a single delta equals planSegments", async () => {
  for (const text of CASES) {
    const streamed = await collect(planSegmentsStream(once(text)));
    assert.deepEqual(streamed, planSegments(text), `single-delta: ${text}`);
  }
});

test("planSegmentsStream char-by-char equals planSegments", async () => {
  for (const text of CASES) {
    const streamed = await collect(planSegmentsStream(charByChar(text)));
    assert.deepEqual(streamed, planSegments(text), `char-by-char: ${text}`);
  }
});

test("planSegmentsStream emits the aggressive first chunk before the rest", async () => {
  const chunks = await collect(
    planSegmentsStream(once("Spawn ten agents and close everything. Done.")),
  );
  // First sentence is >4 words with no early clause boundary, so it carves at 4.
  assert.equal(chunks[0], "Spawn ten agents and");
  assert.deepEqual(chunks.slice(1), [
    "close everything.",
    "Done.",
  ]);
});

test("planSegmentsStream carves at the first clause boundary within the cap", async () => {
  const chunks = await collect(planSegmentsStream(once("Sure, let me check that.")));
  assert.equal(chunks[0], "Sure,");
  assert.deepEqual(chunks.slice(1), ["let me check that."]);
});

test("synthPipeline yields one indexed segment per chunk", async () => {
  const segments = await collect(
    synthPipeline(planSegmentsStream(once("One. Two. Three.")), fakeSynth()),
  );
  assert.deepEqual(
    segments.map((segment) => segment.sentence),
    ["One.", "Two.", "Three."],
  );
  assert.deepEqual(
    segments.map((segment) => segment.index),
    [0, 1, 2],
  );
  assert.ok(segments.every((segment) => segment.samples.length > 0));
});

test("synthPipeline on an empty stream yields nothing", async () => {
  const segments = await collect(synthPipeline(planSegmentsStream(once("")), fakeSynth()));
  assert.deepEqual(segments, []);
});

test("adapter speak() handles both a string and an incremental token stream", async () => {
  const tts = new SherpaTextToSpeech(async () => ({ ...fakeSynth(), numSpeakers: 1, sampleRate: 16000 }));
  const session = await tts.open({ model: "piper-amy" });

  const fromString = await collect(session.speak("Hello, world. Bye."));
  const fromStream = await collect(
    session.speak(charByChar("Hello, world. Bye.")),
  );

  const sentences = (segments: { sentence: string }[]) =>
    segments.map((segment) => segment.sentence);
  assert.deepEqual(sentences(fromString), ["Hello,", "world.", "Bye."]);
  assert.deepEqual(sentences(fromStream), sentences(fromString));

  await session.close();
});

test("a ready segment arrives before later incremental text", async () => {
  let release!: () => void;
  const later = new Promise<void>((resolve) => { release = resolve; });
  async function* text() {
    yield "First.";
    await later;
    yield "Second.";
  }
  const session = await new SherpaTextToSpeech(async () => ({ ...fakeSynth(), numSpeakers: 1, sampleRate: 16000 })).open();
  const audio = session.speak(text())[Symbol.asyncIterator]();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const first = await Promise.race([
      audio.next(),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("ready audio blocked on later text")), 100); }),
    ]);
    assert.equal(first.value?.sentence, "First.");
  } finally {
    clearTimeout(timer);
    release();
    await audio.return?.();
    await session.close();
  }
});

test("adapter rejects invalid speakers, unresolved names and invalid speed", async () => {
  let loads = 0;
  const tts = new SherpaTextToSpeech(async () => { loads += 1; return { ...fakeSynth(), numSpeakers: 4, sampleRate: 16000 }; });
  for (const voice of [-1, 0.5, NaN, "af_heart"]) await assert.rejects(tts.open({ voice }));
  for (const speed of [0, -1, NaN, Infinity]) await assert.rejects(tts.open({ speed }));
  assert.equal(loads, 0);
  await assert.rejects(tts.open({ voice: 4 }), /speaker/);
  await (await tts.open({ voice: 3 })).close();
});
