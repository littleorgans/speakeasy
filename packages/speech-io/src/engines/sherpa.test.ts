import assert from "node:assert/strict";
import { test } from "node:test";
import type { EmptyEvent, FinalEvent, STTSession } from "../contract.ts";
import { SherpaEngine } from "./sherpa.ts";
import type {
  OnlineRecognizer,
  OnlineRecognizerConfig,
  OnlineStream,
  RecognizerResult,
} from "./sherpa-api.ts";
import { bindingWith, seededModelRoot } from "./test-support.ts";

const MODEL = "en-2023-06-21";

type FakeOnline = {
  configs: OnlineRecognizerConfig[];
  accepted: number[];
  result: RecognizerResult;
  endpoint: boolean;
};

function fakeOnline(): { fake: FakeOnline; Recognizer: new (config: OnlineRecognizerConfig) => OnlineRecognizer } {
  const fake: FakeOnline = { configs: [], accepted: [], result: {}, endpoint: false };
  class Recognizer implements OnlineRecognizer {
    constructor(config: OnlineRecognizerConfig) {
      fake.configs.push(config);
    }
    createStream(): OnlineStream {
      return {
        acceptWaveform: (waveform) => fake.accepted.push(waveform.samples.length),
        inputFinished: () => {},
      };
    }
    isReady(): boolean {
      return false;
    }
    decode(): void {}
    isEndpoint(): boolean {
      return fake.endpoint;
    }
    reset(): void {
      fake.result = {};
      fake.endpoint = false;
    }
    getResult(): RecognizerResult {
      return fake.result;
    }
  }
  return { fake, Recognizer };
}

async function openSession(
  options: { hotwords?: boolean; mode?: "manual" | "eager" } = {},
): Promise<{ fake: FakeOnline; session: STTSession; events: [string, unknown][] }> {
  const { fake, Recognizer } = fakeOnline();
  const engine = new SherpaEngine({
    model: MODEL,
    modelRoot: await seededModelRoot(MODEL),
    binding: bindingWith({ OnlineRecognizer: Recognizer }),
    ...(options.hotwords
      ? { hotwords: { phrases: ["kubectl"], file: "/tmp/hotwords.txt", score: 2 } }
      : {}),
  });
  const session = await engine.open({ endpoint: { mode: options.mode ?? "manual" } });
  const events: [string, unknown][] = [];
  for (const name of ["partial", "final", "empty", "endpoint"]) {
    session.on(name, (event: unknown) => events.push([name, event]));
  }
  return { fake, session, events };
}

test("flush with no decoded text emits empty with the held audio length", async () => {
  const { session, events } = await openSession();
  session.pushAudio(new Float32Array(8_000));
  session.flush();
  assert.deepEqual(events.map(([name]) => name), ["empty"]);
  const empty = events[0]?.[1] as EmptyEvent;
  assert.equal(empty.audioMs, 500);
  assert.equal(typeof empty.decodeMs, "number");
});

test("flush with text emits final detail; padding is not counted as audio", async () => {
  const { fake, session, events } = await openSession();
  session.pushAudio(new Float32Array(16_000));
  fake.result = {
    text: " OPEN FILE",
    tokens: [" OPEN", " FI", "LE"],
    timestamps: [0.1, 0.5, 0.6],
    ys_probs: [-0.1, -0.2, -0.3],
    start_time: 0,
  };
  session.flush();
  assert.deepEqual(events.map(([name]) => name), ["final"]);
  const final = events[0]?.[1] as FinalEvent;
  assert.equal(final.text, "OPEN FILE");
  assert.equal(final.audioMs, 1_000);
  assert.deepEqual(final.tokens, [
    { text: " OPEN", startMs: 100, logProb: -0.1 },
    { text: " FI", startMs: 500, logProb: -0.2 },
    { text: "LE", startMs: 600, logProb: -0.3 },
  ]);
  assert.ok(fake.accepted.length === 2 && (fake.accepted[1] ?? 0) > 16_000);
});

test("token times restart at each utterance after a flush", async () => {
  const { fake, session, events } = await openSession();
  session.pushAudio(new Float32Array(16_000));
  session.flush();
  // The stream clock runs on: 1 s of audio plus 1.2 s of flush padding.
  session.pushAudio(new Float32Array(16_000));
  fake.result = { text: " GO", tokens: [" GO"], timestamps: [0.5], start_time: 2 };
  session.flush();
  const final = events.find(([name]) => name === "final")?.[1] as FinalEvent;
  assert.deepEqual(final.tokens?.map((token) => token.startMs), [300]);
});

test("an endpoint on silence emits endpoint only, never empty", async () => {
  const { fake, session, events } = await openSession({ mode: "eager" });
  fake.endpoint = true;
  session.pushAudio(new Float32Array(1_600));
  assert.deepEqual(events.map(([name]) => name), ["endpoint"]);
});

test("greedy search unless hotwords are passed", async () => {
  const plain = await openSession();
  assert.equal(plain.fake.configs[0]?.decodingMethod, "greedy_search");
  assert.equal(plain.fake.configs[0]?.hotwordsFile, "");
  const biased = await openSession({ hotwords: true });
  assert.equal(biased.fake.configs[0]?.decodingMethod, "modified_beam_search");
  assert.equal(biased.fake.configs[0]?.hotwordsFile, "/tmp/hotwords.txt");
});

test("the engine refuses an offline model and labels with the binding version", async () => {
  const { Recognizer } = fakeOnline();
  const binding = bindingWith({ OnlineRecognizer: Recognizer });
  assert.throws(
    () =>
      new SherpaEngine({
        model: "en-parakeet-tdt-0.6b-v2-int8",
        modelRoot: "/unused",
        binding,
      }),
    /offline/,
  );
  const engine = new SherpaEngine({ model: MODEL, modelRoot: "/unused", binding });
  assert.equal(
    engine.label,
    "sherpa-onnx-node-fake:sherpa-onnx-streaming-zipformer-en-2023-06-21",
  );
});
