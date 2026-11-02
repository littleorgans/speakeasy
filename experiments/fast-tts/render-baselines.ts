import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getLocalOutputReadiness, resolveLocalOutput, type LocalOutputModelId } from "../../packages/speech-io/src/tts/catalogue.ts";
import { LOCAL_TTS_MODELS } from "../../packages/speech-io/src/tts/models.ts";
import { SherpaTextToSpeech } from "../../packages/speech-io/src/tts/sherpa-tts.ts";
import { TtsSynth, writeWav } from "../../packages/speech-io/src/tts/synth.ts";
import { round, runCandidate, type Adapter } from "./driver.ts";

/**
 * fast-tts baseline engines through speak-easy's own path: TtsSynth (sherpa, cpu
 * provider, Piper Plus frontend) behind SherpaTextToSpeech, whose speak() runs the
 * shared sentence pipeline. No onProgress anywhere. driver.ts runs the flags, cold
 * and warm shapes and result JSON (CONTRACT.md); summarize.py derives the metrics.
 *
 * cold: load, then one run of one --text (default status) with no warmup.
 * warm: load, one discarded warmup synth, then --repeats passes over the texts.
 * Adapted from experiments/audio8-tts/render-baselines.ts (audio8-tts thread).
 */

// threads unset: TtsSynth's product default (DEFAULT_NUM_THREADS in synth.ts).
const ENGINES = {
  "piper-kristin": { label: "Piper Kristin medium", model: "piper-kristin-medium", voice: "kristin" },
  "piper-kristin-t4": { label: "Piper Kristin medium", model: "piper-kristin-medium", voice: "kristin", threads: 4 },
  "kokoro-heart": { label: "Kokoro v1.0 · af_heart", model: "kokoro-v1.0", voice: "af_heart" },
  "kokoro-heart-t4": { label: "Kokoro v1.0 · af_heart", model: "kokoro-v1.0", voice: "af_heart", threads: 4 },
  "kokoro-michael": { label: "Kokoro v1.0 · am_michael", model: "kokoro-v1.0", voice: "am_michael" },
} as const satisfies Record<string, { label: string; model: LocalOutputModelId; voice: string; threads?: number }>;

// Resolved off the clock, before any load.
const RESOLVED = Object.fromEntries(Object.entries(ENGINES).map(([id, engine]: [string, { label: string; model: LocalOutputModelId; voice: string; threads?: number }]) =>
  [id, { engine, ...resolveLocalOutput({ schema: 1, provider: { kind: "local" }, model: engine.model, voice: engine.voice, speed: 1 }) }]));

await runCandidate(Object.keys(ENGINES), async ({ engine: engineId, root }): Promise<Adapter> => {
  const { engine, voice, speed } = RESOLVED[engineId]!;
  const synth = await TtsSynth.create(engine.model, engine.threads, { root, allowDownload: false });
  const session = await new SherpaTextToSpeech(async () => synth).open({ model: engine.model, voice: voice.speakerId, speed });
  return {
    engine: {
      id: engineId,
      label: `${engine.label} · ${synth.numThreads} threads${engine.threads === undefined ? ", product default" : ""}`,
      runtime: `sherpa-onnx-node ${packageVersion("sherpa-onnx-node")}`, provider: "cpu",
      model: engine.model, modelRevision: `sha256:${LOCAL_TTS_MODELS[engine.model].files[0].sha256.slice(0, 12)}`,
      voice: engine.voice, sampleRate: synth.sampleRate, threads: synth.numThreads,
      streaming: "speak-easy sentence pipeline (SherpaTextToSpeech.speak, depth 1): segment waveforms, "
        + "not incremental within a segment; full mode is one generateAsync call",
      chunkPolicy: "planSegments: first chunk is the first clause or 4 words of sentence 1, then whole sentences; "
        + "trimSilence trims each chunk and appends a 150 ms gap",
      cancellation: "between segments (the host returns the generator); a depth 1 lookahead synth already "
        + "started runs to completion",
    },
    versions: { "sherpa-onnx-node": packageVersion("sherpa-onnx-node") },
    loadDetail: { nativeLoadMs: round(synth.loadMs), note: "loadMs is TtsSynth.create (digest verify, native load, frontend) plus session open" },
    // TtsSynth.warmup returns its own synth time.
    warmup: () => synth.warmup(),
    // The shared sentence pipeline: each segment as it reaches the caller.
    async *stream(text) {
      for await (const segment of session.speak(text)) yield { samples: segment.samples, text: segment.sentence };
    },
    // Whole text in one synth call: one chunk at the end.
    full: async (text) => (await synth.synth({ text, sid: voice.speakerId, speed })).samples,
    close: () => session.close(),
    writeClip: (path, samples, sampleRate) => writeWav(path, { samples, sampleRate }),
  };
}, async ({ engine: engineId, root }) => {
  // Integrity check off the clock; TtsSynth.create verifies the same digests again inside loadMs.
  const { engine } = RESOLVED[engineId]!;
  const readiness = await getLocalOutputReadiness(engine.model, root);
  if (readiness.state !== "ready") throw new Error(`${engine.model} not ready: ${readiness.reason}`);
});

function packageVersion(name: string): string {
  const path = join(import.meta.dirname, "../../packages/speech-io/node_modules", name, "package.json");
  return (JSON.parse(readFileSync(path, "utf8")) as { version: string }).version;
}
