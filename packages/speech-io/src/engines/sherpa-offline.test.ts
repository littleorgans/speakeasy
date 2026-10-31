import assert from "node:assert/strict";
import { once } from "node:events";
import { join } from "node:path";
import { test } from "node:test";
import type { EmptyEvent, FinalEvent } from "../contract.ts";
import { SherpaOfflineEngine } from "./sherpa-offline.ts";
import type {
  OfflineRecognizer,
  OfflineRecognizerConfig,
  OfflineStream,
  RecognizerResult,
} from "./sherpa-api.ts";
import { SHERPA_MODELS, type SherpaModelId } from "./sherpa-models.ts";
import { bindingWith, seededModelRoot } from "./test-support.ts";

type FakeOffline = {
  configs: OfflineRecognizerConfig[];
  decoded: number[];
  results: (RecognizerResult | Error)[];
};

async function openSession(
  model: SherpaModelId,
  results: (RecognizerResult | Error)[] = [],
) {
  const fake: FakeOffline = { configs: [], decoded: [], results };
  class Recognizer implements OfflineRecognizer {
    constructor(config: OfflineRecognizerConfig) {
      fake.configs.push(config);
    }
    createStream(): OfflineStream & { samples: number } {
      const stream = {
        samples: 0,
        acceptWaveform: (waveform: { samples: Float32Array }) => {
          stream.samples += waveform.samples.length;
        },
      };
      return stream;
    }
    async decodeAsync(stream: OfflineStream): Promise<RecognizerResult> {
      fake.decoded.push((stream as OfflineStream & { samples: number }).samples);
      const result = fake.results.shift() ?? {};
      if (result instanceof Error) throw result;
      return result;
    }
  }
  const engine = new SherpaOfflineEngine({
    model,
    modelRoot: await seededModelRoot(model),
    binding: bindingWith({ OfflineRecognizer: Recognizer }),
  });
  const session = await engine.open({ endpoint: { mode: "manual" } });
  return { fake, session };
}

test("a silent hold decodes to empty", async () => {
  const { fake, session } = await openSession("en-parakeet-tdt-0.6b-v2-int8", [
    { text: "", tokens: [], timestamps: [] },
  ]);
  session.pushAudio(new Float32Array(3_200));
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 200);
  assert.deepEqual(fake.decoded, [3_200]);
});

test("a hold with no audio is empty without touching the decoder", async () => {
  const { fake, session } = await openSession("en-moonshine-base-int8");
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 0);
  assert.deepEqual(fake.decoded, []);
});

test("audio past maxAudioMs is dropped before decode", async () => {
  const model = "en-moonshine-base-int8";
  const capSamples = (SHERPA_MODELS[model].maxAudioMs * 16_000) / 1_000;
  const { fake, session } = await openSession(model, [{ text: "long" }]);
  const frame = new Float32Array(16_000);
  for (let second = 0; second < 32; second += 1) {
    session.pushAudio(frame);
  }
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.deepEqual(fake.decoded, [capSamples]);
  assert.equal(final.audioMs, SHERPA_MODELS[model].maxAudioMs);
});

test("Parakeet final carries tokens; flushes answer in order", async () => {
  const { session } = await openSession("en-parakeet-tdt-0.6b-v2-int8", [
    {
      text: " Open the file.",
      tokens: [" Open", " the", " file", "."],
      timestamps: [0.08, 0.32, 0.48, 0.8],
      ys_log_probs: [-0.01, -0.02, -0.03, -0.04],
    },
    { text: "" },
  ]);
  const events: [string, unknown][] = [];
  for (const name of ["final", "empty"]) {
    session.on(name, (event: unknown) => events.push([name, event]));
  }
  session.pushAudio(new Float32Array(16_000));
  session.flush();
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  await session.end();
  assert.deepEqual(events.map(([name]) => name), ["final", "empty"]);
  const final = events[0]?.[1] as FinalEvent;
  assert.equal(final.text, "Open the file.");
  assert.equal(final.audioMs, 1_000);
  assert.deepEqual(final.tokens?.map((token) => [token.startMs, token.logProb]), [
    [80, -0.01],
    [320, -0.02],
    [480, -0.03],
    [800, -0.04],
  ]);
});

test("Moonshine final has text and no invented tokens", async () => {
  const { fake, session } = await openSession("en-moonshine-base-int8", [
    { text: " Open the file.", tokens: [" Open", " the", " file", "."], timestamps: [] },
  ]);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.deepEqual(final.tokens, []);
  assert.ok(fake.configs[0]?.modelConfig.moonshine);
  assert.equal(fake.configs[0]?.modelConfig.transducer, undefined);
});

test("Canary transcribes English to English with punctuation", async () => {
  const { fake, session } = await openSession("en-canary-180m-flash-int8", [
    { text: " Open the file.", tokens: [" Open", " the", " file", "."] },
  ]);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.text, "Open the file.");
  assert.deepEqual(final.tokens, []);
  const { modelConfig } = fake.configs[0]!;
  const dir = SHERPA_MODELS["en-canary-180m-flash-int8"].name;
  const { encoder, decoder, ...options } = modelConfig.canary!;
  assert.deepEqual(options, { srcLang: "en", tgtLang: "en", usePnc: 1 });
  assert.ok(encoder.endsWith(join(dir, "encoder.int8.onnx")));
  assert.ok(decoder.endsWith(join(dir, "decoder.int8.onnx")));
  assert.ok(modelConfig.tokens.endsWith(join(dir, "tokens.txt")));
  assert.equal(modelConfig.transducer, undefined);
});

test("Qwen3 ASR loads its tokenizer directory and reports no tokens", async () => {
  const { fake, session } = await openSession("en-qwen3-asr-0.6b-int8-2026-03-25", [
    { text: "Okay, Charles.", tokens: ["Okay", ",", " Charles", "."], timestamps: [] },
  ]);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.text, "Okay, Charles.");
  assert.deepEqual(final.tokens, []);
  const { modelConfig } = fake.configs[0]!;
  const dir = SHERPA_MODELS["en-qwen3-asr-0.6b-int8-2026-03-25"].name;
  const qwen = modelConfig.qwen3Asr!;
  assert.ok(qwen.convFrontend.endsWith(join(dir, "conv_frontend.onnx")));
  assert.ok(qwen.encoder.endsWith(join(dir, "encoder.int8.onnx")));
  assert.ok(qwen.decoder.endsWith(join(dir, "decoder.int8.onnx")));
  assert.ok(qwen.tokenizer.endsWith(join(dir, "tokenizer")));
  assert.equal(modelConfig.tokens, "");
  assert.equal(modelConfig.transducer, undefined);
});

test("Qwen3 ASR reserves generation room for a dense 30 second hold", async () => {
  const model = "en-qwen3-asr-0.6b-int8-2026-03-25";
  const { fake, session } = await openSession(model, [{ text: "purple bicycle" }]);
  session.pushAudio(new Float32Array(30 * 16_000));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.audioMs, 30_000);
  assert.deepEqual(fake.decoded, [30 * 16_000]);
  const qwen = {
    maxNewTokens: 128,
    maxTotalLen: 512,
    ...fake.configs[0]!.modelConfig.qwen3Asr,
  };
  // The native 30 s fixture needs over 128 output tokens. Sherpa makes 13
  // audio tokens per second; reserve the prompt as well as the output budget.
  assert.ok(qwen.maxNewTokens >= 256, "dense speech needs more than the native 128 token default");
  assert.ok(
    qwen.maxTotalLen >= 30 * 13 + 32 + qwen.maxNewTokens,
    "audio and prompt must leave room for the configured generation budget",
  );
});

test("Parakeet Ultra opens from its file set as a NeMo transducer", async () => {
  const { fake, session } = await openSession("en-parakeet-ultra-int8", [
    { text: " Hi.", tokens: [" Hi", "."], timestamps: [0.1, 0.3], ys_log_probs: [-0.1, -0.2] },
  ]);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.deepEqual(final.tokens?.map((token) => token.startMs), [100, 300]);
  const { modelConfig } = fake.configs[0]!;
  const dir = SHERPA_MODELS["en-parakeet-ultra-int8"].name;
  assert.equal(modelConfig.modelType, "nemo_transducer");
  assert.ok(modelConfig.transducer!.encoder.endsWith(join(dir, "encoder.int8.onnx")));
  assert.ok(modelConfig.tokens.endsWith(join(dir, "tokens.txt")));
});

test("a decode failure surfaces as an error event", async () => {
  const { session } = await openSession("en-parakeet-tdt-0.6b-v2-int8", [
    new Error("decoder failed"),
  ]);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [error] = (await once(session, "error")) as [{ err: unknown }];
  assert.equal((error.err as Error).message, "decoder failed");
});

test("the offline engine refuses a streaming model", () => {
  assert.throws(
    () =>
      new SherpaOfflineEngine({
        model: "en-2023-06-21",
        modelRoot: "/unused",
        binding: bindingWith({}),
      }),
    /streaming/,
  );
});
