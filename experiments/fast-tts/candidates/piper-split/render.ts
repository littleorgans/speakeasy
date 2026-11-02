import { join } from "node:path";
import { createTtsFrontend } from "../../../../packages/speech-io/src/tts/frontend.ts";
import { LOCAL_TTS_MODELS } from "../../../../packages/speech-io/src/tts/models.ts";
import { planSegments, splitSentences } from "../../../../packages/speech-io/src/tts/stream.ts";
import { collect, concat, fail, NativeWorker, runCandidate, WARMUP_TEXT, type Adapter } from "../../driver.ts";

/**
 * fast-tts candidate: Piper Kristin medium split into encoder and decoder, run
 * by libpiper (piper1-gpl C API) in a resident worker (worker.c). The text
 * frontend is speak-easy's own: createTtsFrontend, as TtsSynth builds it for
 * Kristin (Piper Plus G2P, RP mapping for espeak voice "en"), on the clock in
 * this process. Its phoneme codepoints reach libpiper as text with
 * phoneme_type "text", so libpiper maps them to ids and never runs eSpeak.
 *
 * Engines differ only in the parts sent: `piper-split-t4` sends whole sentences
 * (splitSentences) and lets the decoder chunking give the early first audio;
 * `piper-split-carved-t4` sends the baselines' planSegments. Each yield is one
 * decoder chunk as libpiper returns it, untrimmed. Cancellation checkpoint:
 * between decoder chunks inside the worker. libpiper fixes 4 intra op threads
 * here (performance cores, at most 4), so compare with piper-kristin-t4.
 *
 * PIPER_SPLIT_WORKER names the compiled worker (README.md Setup).
 */

const SPLIT_DIR = "piper-kristin-split-e7229f7d";
const VOICE_DIR = "vits-piper-en_US-kristin-medium";
const PIPER_COMMIT = "10880972db9e0aca101fed350eadf8f49deefe9f";
const ENGINES = {
  "piper-split-t4": {
    label: "Piper Kristin medium split decoder · sentences", parts: splitSentences,
    policy: "splitSentences: whole sentences, no carved first chunk",
  },
  "piper-split-carved-t4": {
    label: "Piper Kristin medium split decoder · planSegments", parts: planSegments,
    policy: "planSegments: first part is the first clause or 4 words of sentence 1, then whole sentences",
  },
} as const;

await runCandidate(Object.keys(ENGINES), async ({ engine, root }): Promise<Adapter> => {
  const spec = ENGINES[engine as keyof typeof ENGINES];
  const workerPath = process.env.PIPER_SPLIT_WORKER ?? fail("set PIPER_SPLIT_WORKER to the compiled worker (README.md)");
  const kristin = LOCAL_TTS_MODELS["piper-kristin-medium"];
  const frontend = await createTtsFrontend({ tokens: join(root, VOICE_DIR, "tokens.txt"), lang: kristin.lang ?? "en", family: kristin.family }, root, false);
  // libpiper finds <stem>.encoder.onnx and <stem>.decoder.onnx next to this path.
  const worker = await NativeWorker.start(workerPath, [join(root, SPLIT_DIR, "en_US-kristin-medium.onnx"), join(root, SPLIT_DIR, "en_US-kristin-medium.onnx.json")]);

  /** One worker part per frontend sentence; text phoneme_type lowercases and NFD normalizes, so require both to be no-ops. */
  const phonemeParts = (text: string) => (frontend.request(text, 0, 1)?.generationConfig?.phonemeCodepoints ?? []).map((codepoints) => {
    const part = String.fromCodePoint(...codepoints);
    if (part.toLowerCase() !== part || part.normalize("NFD") !== part) throw new Error(`phonemes change under libpiper text normalization: ${part}`);
    return part;
  });
  async function* stream(text: string) {
    for (const piece of spec.parts(text)) {
      for await (const samples of worker.synth(phonemeParts(piece))) yield { samples, text: piece };
    }
  }
  return {
    engine: {
      id: engine, label: `${spec.label} · 4 threads`, runtime: `libpiper ${String(worker.ready.piper)} (piper1-gpl ${PIPER_COMMIT.slice(0, 8)}), onnxruntime 1.22.0`,
      provider: "cpu", model: "piper-kristin-medium", modelRevision: `rhasspy/piper-voices@e7229f7d split; sha256:${kristin.files![0]!.sha256.slice(0, 12)}`,
      voice: "kristin", sampleRate: 22050, threads: 4,
      streaming: "libpiper split voice: one encoder pass per part, then overlapping decoder chunks, so audio is "
        + "incremental within a sentence; full mode drains the same chunks into one",
      chunkPolicy: `${spec.policy}; chunks are libpiper's decoder chunks (first 32 latent frames, 372 ms, 12 context frames `
        + "each side, later 24 to 256 frames sized from decoder speed), untrimmed, no added gaps",
      cancellation: "between decoder chunks in the worker (no stop request yet; a host stops by not sending the next "
        + "part, the current part runs to completion)",
    },
    versions: { libpiper: String(worker.ready.piper), "piper1-gpl": PIPER_COMMIT, onnxruntime: "1.22.0" },
    loadDetail: { note: "loadMs is the Piper Plus frontend (CMU dict parse) plus worker spawn and piper_create (encoder and decoder sessions)" },
    warmup: async () => void (await collect(stream(WARMUP_TEXT))),
    stream,
    full: async (text) => concat((await collect(stream(text))).map((chunk) => chunk.samples)),
    close: () => worker.close(),
  };
});
