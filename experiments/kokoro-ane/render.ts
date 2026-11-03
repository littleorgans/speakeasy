import { join } from "node:path";
import { planSegments, trimSilence } from "../../packages/speech-io/src/tts/stream.ts";
import { collect, concat, fail, NativeWorker, runCandidate, WARMUP_TEXT, type Adapter } from "../fast-tts/driver.ts";

/**
 * fast-tts engine: Kokoro v1.0 on mattmireles/kokoro-coreml's KokoroTTS SDK in
 * a resident Swift worker (worker/), the Neural Engine route of the kokoro-ane
 * experiment. The starter bundle has one 15 s acoustic bucket, so every chunk
 * runs at fixed padded shapes. Engines pick the generator (README.md):
 * `hybrid`, the shipped default (ANE convolutions plus Metal), `folded` (all
 * ANE) and `cpu` (every stage on the CPU, the placement control). The SDK's
 * Misaki G2P runs inside the worker, on the clock.
 *
 * Stream mode follows the baselines' chunk policy: speak-easy planSegments, all
 * segments submitted at once, one synthesize call per segment, trimSilence on
 * each, so every yield is one completed segment waveform. Full mode is one
 * synthesize call on the whole text. Cancellation checkpoint: between segments;
 * an active Core ML chain runs to completion.
 *
 * KOKORO_COREML_WORKER names the built worker (README.md Setup).
 */

const ASSETS_DIR = "kokoro-coreml-mattmireles-034f7741";
const SDK_COMMIT = "b23caa862f0585c2c65187d0fc97e0f8311041dc";
const MODEL_REVISION = "mattmireles/kokoro-coreml@034f7741bf938fd6b4b9c75c22bc72cd795f908a (starter bundle, 15 s bucket)";
const SAMPLE_RATE = 24_000;
const VARIANTS = {
  hybrid: "generator convolutions on the ANE, normalization and activations on Metal (GPU); other stages ANE and CPU",
  folded: "row-folded generator on the ANE; other stages ANE and CPU",
  cpu: "every stage on the CPU (KokoroComputePolicy.cpuOnly)",
} as const;
const ENGINES = {
  "kokoro-ane-heart": { voice: "af_heart", variant: "hybrid" },
  "kokoro-ane-folded-heart": { voice: "af_heart", variant: "folded" },
  "kokoro-ane-cpu-heart": { voice: "af_heart", variant: "cpu" },
} as const;

await runCandidate(Object.keys(ENGINES), async ({ engine, root }): Promise<Adapter> => {
  const { voice, variant } = ENGINES[engine as keyof typeof ENGINES];
  const workerPath = process.env.KOKORO_COREML_WORKER ?? fail("set KOKORO_COREML_WORKER to the built worker (README.md)");
  const worker = await NativeWorker.start(workerPath, [join(root, ASSETS_DIR), voice, variant]);
  const synth = (parts: string[]) => collect(worker.synth(parts));
  return {
    engine: {
      id: engine, label: `Kokoro v1.0 Core ML (kokoro-coreml ${variant}) · ${voice}`,
      runtime: `kokoro-coreml ${SDK_COMMIT.slice(0, 8)} KokoroTTS`, provider: `coreml: ${VARIANTS[variant]}`,
      model: "kokoro-v1.0", modelRevision: MODEL_REVISION, voice, sampleRate: SAMPLE_RATE,
      streaming: "resident Swift worker, one KokoroTTS.synthesize call per segment, framed PCM over stdio: segment "
        + "waveforms, not incremental within a segment; full mode is one synthesize call",
      chunkPolicy: "planSegments: first chunk is the first clause or 4 words of sentence 1, then whole sentences, all "
        + "submitted at once; trimSilence trims each chunk and appends a 150 ms gap",
      cancellation: "between segments in the worker (no stop request yet); an active Core ML chain runs to completion",
    },
    versions: { "kokoro-coreml": SDK_COMMIT, swift: "6.4 (Xcode 27.2)" },
    loadDetail: { note: "loadMs is worker spawn plus KokoroTTS.load (manifest and digest checks, vocab, hn-NSF weights); "
      + "Core ML models load lazily on the first synthesis" },
    warmup: async () => void (await synth([WARMUP_TEXT])),
    async *stream(text) {
      const segments = planSegments(text);
      let index = 0;
      for await (const samples of worker.synth(segments)) yield { samples: trimSilence(samples, SAMPLE_RATE), text: segments[index++]! };
    },
    full: async (text) => concat(await synth([text])),
    close: () => worker.close(),
  };
});
