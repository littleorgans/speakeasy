import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { performance } from "node:perf_hooks";
import { getLocalOutputReadiness, resolveLocalOutput, type LocalOutputModelId } from "../../packages/speech-io/src/tts/catalogue.ts";
import { SherpaTextToSpeech } from "../../packages/speech-io/src/tts/sherpa-tts.ts";
import { TtsSynth, writeWav } from "../../packages/speech-io/src/tts/synth.ts";

/**
 * Render the cohort with the two TM local voices through speak-easy's own path:
 * TtsSynth (sherpa, cpu provider, Piper Plus frontend) behind SherpaTextToSpeech,
 * whose speak() runs the shared sentence pipeline. No onProgress anywhere.
 *
 * Per engine: verify pinned payloads, cold load, one warmup synth, then `runs`
 * passes over every text. TTFA is the first segment's readyAtMs; RTF is the last
 * segment's readyAtMs over total audio. The first pass writes the WAV.
 */

const ENGINES = [
  { engine: "kristin", model: "piper-kristin-medium", voice: "kristin" },
  { engine: "kokoro-heart", model: "kokoro-v1.0", voice: "af_heart" },
] as const satisfies readonly { engine: string; model: LocalOutputModelId; voice: string }[];

type Cohort = { texts: { id: string; text: string }[]; reference: { id: string; text: string } };

const { values } = parseArgs({
  options: {
    root: { type: "string" },
    out: { type: "string" },
    json: { type: "string" },
    runs: { type: "string", default: "1" },
    engine: { type: "string", multiple: true },
  },
});
if (!values.root || !values.out || !values.json) {
  throw new Error("usage: node experiments/audio8-tts/render-baselines.ts --root <models/tts> --out <clips dir> --json <timings.json> [--runs N] [--engine kristin|kokoro-heart]");
}
const runs = Number(values.runs);
const cohort = JSON.parse(await readFile(join(import.meta.dirname, "cohort.json"), "utf8")) as Cohort;
const selected = ENGINES.filter((entry) => !values.engine || values.engine.includes(entry.engine));

const results = [];
for (const entry of selected) {
  const readiness = await getLocalOutputReadiness(entry.model, values.root);
  if (readiness.state !== "ready") throw new Error(`${entry.model} not ready: ${readiness.reason}`);
  const { voice, speed } = resolveLocalOutput({ schema: 1, provider: { kind: "local" }, model: entry.model, voice: entry.voice, speed: 1 });

  const coldStart = performance.now();
  const synth = await TtsSynth.create(entry.model, undefined, { root: values.root, allowDownload: false });
  const coldOpenMs = performance.now() - coldStart;
  const warmupMs = await synth.warmup();
  const session = await new SherpaTextToSpeech(async () => synth).open({ model: entry.model, voice: voice.speakerId, speed });

  const texts = [];
  for (const item of [...cohort.texts, cohort.reference]) {
    const passes = [];
    for (let run = 0; run < runs; run += 1) {
      const segments = [];
      for await (const segment of session.speak(item.text)) segments.push(segment);
      const audioMs = segments.reduce((sum, segment) => sum + segment.audioDurationMs, 0);
      const lastReadyMs = segments.at(-1)!.readyAtMs;
      passes.push({
        ttfaMs: segments[0]!.readyAtMs, lastReadyMs, audioMs, rtf: lastReadyMs / audioMs,
        segments: segments.length, synthMs: segments.reduce((sum, segment) => sum + segment.synthMs, 0),
      });
      if (run === 0) {
        const samples = new Float32Array(segments.reduce((sum, segment) => sum + segment.samples.length, 0));
        let offset = 0;
        for (const segment of segments) {
          samples.set(segment.samples, offset);
          offset += segment.samples.length;
        }
        await writeWav(join(values.out, `${item.id}--${entry.engine}.wav`), { samples, sampleRate: synth.sampleRate });
      }
    }
    texts.push({ id: item.id, runs: passes });
  }
  await session.close();
  results.push({
    engine: entry.engine, model: entry.model, voice: entry.voice, threads: synth.numThreads,
    sampleRate: synth.sampleRate, coldOpenMs, nativeLoadMs: synth.loadMs, warmupMs, texts,
  });
  console.log(`${entry.engine}: rendered ${texts.length} texts x ${runs} runs`);
}
await writeFile(values.json, `${JSON.stringify({ schema: 1, runs, engines: results }, null, 2)}\n`);
