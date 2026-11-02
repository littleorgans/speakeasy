import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { planSegments, trimSilence } from "../../../../packages/speech-io/src/tts/stream.ts";
import { fail, runCandidate, WARMUP_TEXT, type Adapter } from "../driver.ts";

/**
 * fast-tts candidate: Supertonic 3 ONNX through the upstream Node example's
 * helper.js (onnxruntime-node, cpu), in process. Stream mode follows the
 * baselines' chunk policy: speak-easy planSegments, one `_infer` call per
 * segment, trimSilence on each, so every yield is one completed segment
 * waveform. Full mode is upstream `TextToSpeech.call` (its own sentence
 * grouping up to 300 characters, 0.3 s joins). Steps and speed stay fixed.
 * Cancellation checkpoint: between segments; an active `_infer` runs to the end.
 *
 * SUPERTONIC_NODE names the directory holding the pinned helper.js and its
 * node_modules (README.md Setup).
 */

const SNAPSHOT = "supertonic-3-3cadd1ee";
const REVISION = "Supertone/supertonic-3@3cadd1ee6394adea1bd021217a0e650ede09a323";
const HELPER_COMMIT = "1e9799e964ea4c0dad7cde993b65c3c813a7b373";
const TOTAL_STEP = 8; // upstream example default
const SPEED = 1.05; // upstream default
// Threads as the baselines: 2 (speak-easy's DEFAULT_NUM_THREADS) unless named. Under
// load an untimed probe found 2 threads faster than onnxruntime's default (README.md).
const ENGINES = {
  "supertonic3-f1": { voice: "F1", threads: 2 },
  "supertonic3-m1": { voice: "M1", threads: 2 },
  "supertonic3-f1-t4": { voice: "F1", threads: 4 },
} as const;

type Style = unknown;
type TextToSpeech = {
  sampleRate: number;
  _infer(texts: string[], langs: string[], style: Style, totalStep: number, speed: number): Promise<{ wav: number[]; duration: number[] }>;
  call(text: string, lang: string, style: Style, totalStep: number, speed: number): Promise<{ wav: number[]; duration: number[] }>;
};
type Ort = { InferenceSession: { create(path: string, options?: Record<string, unknown>): Promise<unknown> } };
type Helper = {
  loadTextToSpeech(onnxDir: string, useGpu: boolean): Promise<TextToSpeech>;
  loadVoiceStyle(paths: string[], verbose: boolean): Style;
};

await runCandidate(Object.keys(ENGINES), async ({ engine, root }): Promise<Adapter> => {
  const { voice, threads } = ENGINES[engine as keyof typeof ENGINES];
  const nodeDir = process.env.SUPERTONIC_NODE ?? fail("set SUPERTONIC_NODE to the Supertonic Node env (README.md)");
  // helper.js creates its sessions with empty options; set threads on the same module instance it imports.
  const ort = (await import(pathToFileURL(join(nodeDir, "node_modules", "onnxruntime-node", "dist", "index.js")).href)) as Ort;
  const create = ort.InferenceSession.create.bind(ort.InferenceSession);
  ort.InferenceSession.create = (path, options) => create(path, { ...options, intraOpNumThreads: threads, interOpNumThreads: 1 });
  const helper = (await import(pathToFileURL(join(nodeDir, "helper.js")).href)) as Helper;
  const dir = join(root, SNAPSHOT);
  const tts = await helper.loadTextToSpeech(join(dir, "onnx"), false);
  const style = helper.loadVoiceStyle([join(dir, "voice_styles", `${voice}.json`)], false);
  const rate = tts.sampleRate;
  // `wav` is padded to the latent length; the duration predictor gives the speech length.
  const infer = async (text: string) => {
    const { wav, duration } = await tts._infer([text], ["en"], style, TOTAL_STEP, SPEED);
    return Float32Array.from(wav.slice(0, Math.floor(duration[0]! * rate)));
  };
  const ortVersion = packageVersion(nodeDir, "onnxruntime-node");
  return {
    engine: {
      id: engine, label: `Supertonic 3 · ${voice} · ${TOTAL_STEP} steps · ${threads} threads`, runtime: `onnxruntime-node ${ortVersion}`,
      provider: "cpu", model: "supertonic-3", modelRevision: REVISION, voice, sampleRate: rate,
      steps: TOTAL_STEP, speed: SPEED, threads,
      streaming: "one _infer call per segment (8 flow steps, then the vocoder): segment waveforms, not incremental "
        + "within a segment; full mode is one upstream TextToSpeech.call",
      chunkPolicy: "planSegments: first chunk is the first clause or 4 words of sentence 1, then whole sentences; "
        + "each waveform cut to its predicted duration, then trimSilence trims it and appends a 150 ms gap",
      cancellation: "between segments; an active _infer runs its eight steps and the vocoder to completion",
    },
    versions: { "onnxruntime-node": ortVersion, "supertonic-helper": HELPER_COMMIT },
    loadDetail: { note: "loadMs is four InferenceSession.create calls (intra op threads set, inter op 1) plus unicode indexer and voice style load" },
    warmup: async () => void (await infer(WARMUP_TEXT)),
    async *stream(text) {
      for (const segment of planSegments(text)) yield { samples: trimSilence(await infer(segment), rate), text: segment };
    },
    full: async (text) => {
      // Trimmed as upstream example_onnx.js does: total duration including its 0.3 s joins.
      const { wav, duration } = await tts.call(text, "en", style, TOTAL_STEP, SPEED);
      return Float32Array.from(wav.slice(0, Math.floor(duration[0]! * rate)));
    },
    close: async () => {},
  };
});

function packageVersion(nodeDir: string, name: string): string {
  return (JSON.parse(readFileSync(join(nodeDir, "node_modules", name, "package.json"), "utf8")) as { version: string }).version;
}
