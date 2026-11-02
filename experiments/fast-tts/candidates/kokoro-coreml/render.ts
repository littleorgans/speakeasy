import { join } from "node:path";
import { planSegments, trimSilence } from "../../../../packages/speech-io/src/tts/stream.ts";
import { collect, concat, fail, NativeWorker, runCandidate, WARMUP_TEXT, type Adapter } from "../../driver.ts";

/**
 * fast-tts candidate: Kokoro v1.0 on native Core ML through FluidAudio's
 * KokoroAneManager in a resident Swift worker (worker/). The fallback named by
 * the shortlist: kokoro-coreml itself needs Xcode's Metal compiler for
 * MisakiSwift's MLX dependency (README.md). Engines pick a route: `legacy`, the
 * seven stage chain (CPU and ANE stage policies), or `v3`, the hybrid ANE/GPU
 * generator adapted from kokoro-coreml, which aborts in MPSGraph on macOS 27.0.1
 * (README.md) and stays out of the timed pass. FluidAudio's English frontend
 * (Misaki lexicon, BART G2P fallback) runs inside the worker, on the clock.
 *
 * Stream mode follows the baselines' chunk policy: speak-easy planSegments, all
 * segments submitted at once, one synthesizeDetailed per segment, trimSilence on
 * each, so every yield is one completed segment waveform. Full mode is one
 * synthesizeDetailed call on the whole text. Cancellation checkpoint: between
 * segments; an active Core ML chain runs to completion.
 *
 * KOKORO_ANE_WORKER names the built worker (README.md Setup).
 */

const ASSETS_DIR = "kokoro-ane-fluidaudio";
const FLUIDAUDIO_COMMIT = "e396cf780c4abf8e395087467c0585d3b0d17f68";
// worker/ pins the hub's legacy files to the commit `main` named on 2026-11-03; v3 pins its own.
const MODELS = {
  legacy: "FluidInference/kokoro-82m-coreml@2568276a60ff6e7f312afce3dc9e3f07521cb48d (ANE)",
  v3: "FluidInference/kokoro-82m-coreml@6c0750dd02ef9d981fb82f147d784a3f171755ee (ANE-v3, base 006395f6)",
} as const;
const SAMPLE_RATE = 24_000;
const ENGINES = {
  "kokoro-coreml-heart": { voice: "af_heart", route: "legacy" },
  "kokoro-coreml-michael": { voice: "am_michael", route: "legacy" },
  "kokoro-coreml-v3-heart": { voice: "af_heart", route: "v3" },
} as const;

await runCandidate(Object.keys(ENGINES), async ({ engine, root }): Promise<Adapter> => {
  const { voice, route } = ENGINES[engine as keyof typeof ENGINES];
  const workerPath = process.env.KOKORO_ANE_WORKER ?? fail("set KOKORO_ANE_WORKER to the built worker (README.md)");
  const worker = await NativeWorker.start(workerPath, [join(root, ASSETS_DIR), voice, route]);
  const synth = (parts: string[]) => collect(worker.synth(parts));
  return {
    engine: {
      id: engine, label: `Kokoro v1.0 Core ML (FluidAudio ${route}) · ${voice}`, runtime: `FluidAudio ${FLUIDAUDIO_COMMIT.slice(0, 8)} KokoroAneManager ${route}`,
      provider: `coreml: FluidAudio ${route} per stage compute policies`, model: "kokoro-v1.0", modelRevision: MODELS[route],
      voice, sampleRate: SAMPLE_RATE,
      streaming: "resident Swift worker, one synthesizeDetailed call per segment, framed PCM over stdio: segment "
        + "waveforms, not incremental within a segment; full mode is one synthesizeDetailed call",
      chunkPolicy: "planSegments: first chunk is the first clause or 4 words of sentence 1, then whole sentences, all "
        + "submitted at once; trimSilence trims each chunk and appends a 150 ms gap",
      cancellation: "between segments in the worker (no stop request yet); an active Core ML chain runs to completion",
    },
    versions: { FluidAudio: FLUIDAUDIO_COMMIT, swift: "6.4 (Command Line Tools)" },
    loadDetail: { note: "loadMs is worker spawn plus KokoroAneManager.initialize (Core ML loads of compiled models, G2P assets, voice pack)" },
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
