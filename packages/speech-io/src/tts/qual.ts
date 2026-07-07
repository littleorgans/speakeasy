import { join } from "node:path";
import { formatMs } from "../bench/format.ts";
import { parseTtsModelId, TTS_MODELS, type TtsModelId } from "./models.ts";
import {
  buildTimingReport,
  streamSpeech,
  type SpeechSegment,
} from "./stream.ts";
import {
  BUILD_STATUS_PARAGRAPH,
  printColumns,
  QUAL_SENTENCES,
  TTS_RESULTS_DIR,
} from "./sweep.ts";
import { TtsSynth, writeWav } from "./synth.ts";

/**
 * Streamable-quality sweep: the deliverable that finds the highest-fidelity
 * voice that still streams on CPU. For every registry model it measures RTF on
 * the long build-status paragraph at 2 and 4 threads, writes the two A/B
 * samples (qual-<model>-0.wav short, qual-<model>-1.wav paragraph), and for any
 * model whose best RTF clears the streaming bar proves first-audio latency plus
 * ahead-of-playback through the real sentence pipeline in stream.ts.
 *
 * Reuses TtsSynth, writeWav, streamSpeech/buildTimingReport, and the sweep's
 * sentence corpus + table printer; adds no download or synth logic of its own.
 * Streaming stays on the stable no-callback path (see synth.ts).
 */

/** Below this RTF the sentence pipeline stays ahead of playback on CPU. */
const STREAMABLE_RTF = 0.8;
/** Thread counts benchmarked per model; the second is the best CPU config. */
const THREAD_COUNTS = [2, 4] as const;

type QualRow = {
  modelId: TtsModelId;
  sampleRate: number;
  voices: number;
  rtfByThreads: Map<number, number>;
  streamable: boolean;
  timeToFirstAudioMs?: number;
  aheadOfPlayback?: boolean;
};

await runQualSweep(parseArgs(process.argv.slice(2)));

async function runQualSweep(models: TtsModelId[]): Promise<void> {
  console.log("speak-easy TTS streamable-quality sweep");
  console.log(
    `models=${models.join(",")} threads=${THREAD_COUNTS.join("/")} streamable-rtf<${STREAMABLE_RTF} out=${TTS_RESULTS_DIR}`,
  );

  const rows: QualRow[] = [];
  for (const modelId of models) {
    rows.push(await evaluateModel(modelId));
  }

  printSummary(rows);
}

async function evaluateModel(modelId: TtsModelId): Promise<QualRow> {
  console.log(`\nmodel=${modelId} (${TTS_MODELS[modelId].note})`);
  const rtfByThreads = new Map<number, number>();
  let best: TtsSynth | undefined;

  for (const threads of THREAD_COUNTS) {
    const synth = await TtsSynth.create(modelId, threads);
    // Write the A/B wavs once (audio is thread-count invariant); the two
    // synths double as the session warmup for this thread count.
    if (best === undefined) {
      await writeQualSamples(synth);
    }
    await synth.synth({ text: "Warm up." });
    const measured = await synth.synth({ text: BUILD_STATUS_PARAGRAPH });
    rtfByThreads.set(threads, measured.rtf);
    console.log(
      `model=${modelId} threads=${threads} load=${formatMs(synth.loadMs)} synth=${formatMs(measured.totalSynthMs)} audio=${formatMs(measured.audioDurationMs)} rtf=${measured.rtf.toFixed(3)}`,
    );
    best = synth;
  }

  const bestRtf = Math.min(...rtfByThreads.values());
  const row: QualRow = {
    modelId,
    sampleRate: best!.sampleRate,
    voices: best!.numSpeakers,
    rtfByThreads,
    streamable: bestRtf < STREAMABLE_RTF,
  };

  if (row.streamable) {
    const timing = await measureStreaming(best!);
    row.timeToFirstAudioMs = timing.timeToFirstAudioMs;
    row.aheadOfPlayback = timing.aheadOfPlayback;
    console.log(
      `model=${modelId} stream(threads=${best!.numThreads}) first-audio=${formatMs(timing.timeToFirstAudioMs)} ahead-of-playback=${timing.aheadOfPlayback ? "yes" : "no"}`,
    );
  }
  return row;
}

/** Synth the short + paragraph A/B pair to qual-<model>-<idx>.wav. */
async function writeQualSamples(synth: TtsSynth): Promise<void> {
  for (const [index, sentence] of QUAL_SENTENCES.entries()) {
    const result = await synth.synth({ text: sentence.text });
    const wavPath = join(TTS_RESULTS_DIR, `qual-${synth.modelId}-${index}.wav`);
    await writeWav(wavPath, result);
    console.log(`wrote ${wavPath} (${sentence.id})`);
  }
}

/** Drive the real streaming pipeline (stream.ts) and time first audio. */
async function measureStreaming(synth: TtsSynth) {
  const segments: SpeechSegment[] = [];
  for await (const segment of streamSpeech(BUILD_STATUS_PARAGRAPH, { synth })) {
    segments.push(segment);
  }
  return buildTimingReport(segments);
}

function printSummary(rows: QualRow[]): void {
  console.log("\nstreamable-quality summary:");
  const header = [
    "model",
    "rtf@2thr",
    "rtf@4thr",
    "streamable?",
    "first-audio",
    "ahead",
    "sample-rate",
    "voices",
  ];
  const table = rows.map((row) => [
    row.modelId,
    (row.rtfByThreads.get(2) ?? Number.NaN).toFixed(3),
    (row.rtfByThreads.get(4) ?? Number.NaN).toFixed(3),
    row.streamable ? "yes" : "no",
    row.timeToFirstAudioMs === undefined
      ? "n/a"
      : formatMs(row.timeToFirstAudioMs),
    row.aheadOfPlayback === undefined ? "n/a" : row.aheadOfPlayback ? "yes" : "no",
    `${row.sampleRate}Hz`,
    String(row.voices),
  ]);
  printColumns([header, ...table]);
}

function parseArgs(args: string[]): TtsModelId[] {
  const all = Object.keys(TTS_MODELS) as TtsModelId[];
  const selected: TtsModelId[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg !== "--model") {
      throw new Error(
        `Unknown argument ${arg}\nusage: node src/tts/qual.ts [--model <id>]...\n       --model ids: ${all.join(", ")}`,
      );
    }
    const value = args[index + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --model`);
    }
    selected.push(parseTtsModelId(value));
    index += 1;
  }
  return selected.length > 0 ? selected : all;
}
