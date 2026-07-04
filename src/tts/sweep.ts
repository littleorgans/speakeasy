import { join } from "node:path";
import { formatMs, formatOptionalMs } from "../bench/format.ts";
import { median, medianOptional } from "../bench/stats.ts";
import { TTS_MODELS, type TtsModelId } from "./models.ts";
import { TtsSynth, writeWav, type SynthResult } from "./synth.ts";

/**
 * TTS head-to-head sweep: synth a fixed domain-flavored sentence set on every
 * model, write the wavs for listening, and report timing (total synth, RTF;
 * first-audio prints n/a because the 1.13.3 streaming callback is unusable,
 * see synth.ts).
 */

export const TTS_RESULTS_DIR = join(process.cwd(), "results", "tts");

export type SweepSentence = { id: string; text: string };

/** Multi-sentence sample shared by the sweep and the streaming demo. */
export const BUILD_STATUS_PARAGRAPH =
  "The build finished on the staging channel. Two checks are still running. Say the word and I will promote it.";

/** Short pangram: dense phoneme coverage for hearing a voice's timbre. */
const PANGRAM = "The quick brown fox jumps over the lazy dog.";

export const SWEEP_SENTENCES: readonly SweepSentence[] = [
  {
    id: "domain",
    text: "Deploy the littleorgans agent and restart the chrome pane.",
  },
  { id: "digits", text: "Meeting at 3 45 pm, gate 12." },
  {
    id: "director",
    text: "Open the transport matters pane and switch to chrome.",
  },
  { id: "neutral", text: PANGRAM },
  {
    // Multi-sentence: the long-text case the streaming pipeline (stream.ts)
    // sentence-chunks to get first audio out early.
    id: "paragraph",
    text: BUILD_STATUS_PARAGRAPH,
  },
];

/**
 * Two-utterance A/B set for the streamable-quality sweep (qual.ts): a short
 * pangram then the long build-status paragraph the user flagged as grating.
 * Written to qual-<model>-0.wav (short) and qual-<model>-1.wav (paragraph).
 */
export const QUAL_SENTENCES: readonly SweepSentence[] = [
  { id: "short", text: PANGRAM },
  { id: "paragraph", text: BUILD_STATUS_PARAGRAPH },
];

/** Sentence used for the multi-voice variety wavs (kokoro). */
const VARIETY_SENTENCE = SWEEP_SENTENCES[0]!;
/** Preferred speaker ids for variety wavs; clamped to the model's range. */
const VARIETY_SIDS = [1, 5, 9];

type UtteranceRow = {
  modelId: TtsModelId;
  label: string;
  sid: number;
  wavPath: string;
  result: SynthResult;
};

export type SweepOptions = {
  models: TtsModelId[];
  speed: number;
};

export async function runTtsSweep(options: SweepOptions): Promise<void> {
  console.log("speak-easy TTS sweep");
  console.log(
    `models=${options.models.join(",")} speed=${options.speed} sentences=${SWEEP_SENTENCES.length} out=${TTS_RESULTS_DIR}`,
  );

  const rows: UtteranceRow[] = [];
  for (const modelId of options.models) {
    rows.push(...(await sweepModel(modelId, options.speed)));
  }

  printAggregate(rows, options.models);
}

async function sweepModel(
  modelId: TtsModelId,
  speed: number,
): Promise<UtteranceRow[]> {
  const synth = await TtsSynth.create(modelId);
  console.log(
    `model=${modelId} (${TTS_MODELS[modelId].note}) load=${formatMs(synth.loadMs)} sample-rate=${synth.sampleRate}Hz voices=${synth.numSpeakers}`,
  );

  // One unmeasured warmup so onnxruntime session priming does not skew the
  // per-utterance medians; its timing is still printed as the cold number.
  const cold = await synth.synth({ text: "Warm up.", speed });
  console.log(
    `model=${modelId} cold-synth total=${formatMs(cold.totalSynthMs)}`,
  );

  const rows: UtteranceRow[] = [];
  for (const [index, sentence] of SWEEP_SENTENCES.entries()) {
    rows.push(
      await synthToWav(synth, {
        modelId,
        label: sentence.id,
        sid: 0,
        speed,
        text: sentence.text,
        wavPath: join(TTS_RESULTS_DIR, `${modelId}-${index}.wav`),
      }),
    );
  }

  for (const sid of varietySids(synth.numSpeakers)) {
    rows.push(
      await synthToWav(synth, {
        modelId,
        label: `${VARIETY_SENTENCE.id}-voice${sid}`,
        sid,
        speed,
        text: VARIETY_SENTENCE.text,
        wavPath: join(TTS_RESULTS_DIR, `${modelId}-voice${sid}.wav`),
      }),
    );
  }

  return rows;
}

async function synthToWav(
  synth: TtsSynth,
  input: {
    modelId: TtsModelId;
    label: string;
    sid: number;
    speed: number;
    text: string;
    wavPath: string;
  },
): Promise<UtteranceRow> {
  const result = await synth.synth({
    text: input.text,
    sid: input.sid,
    speed: input.speed,
  });
  await writeWav(input.wavPath, result);
  console.log(
    [
      `model=${input.modelId}`,
      `utt=${input.label}`,
      `sid=${input.sid}`,
      `first-audio=${formatOptionalMs(result.firstAudioMs)}`,
      `total=${formatMs(result.totalSynthMs)}`,
      `audio=${formatMs(result.audioDurationMs)}`,
      `rtf=${result.rtf.toFixed(3)}`,
      `wav=${input.wavPath}`,
    ].join(" "),
  );
  return {
    modelId: input.modelId,
    label: input.label,
    sid: input.sid,
    wavPath: input.wavPath,
    result,
  };
}

/** 2-3 extra voices spread across the model's range; none for single voice. */
function varietySids(numSpeakers: number): number[] {
  return VARIETY_SIDS.filter((sid) => sid < numSpeakers);
}

function printAggregate(rows: UtteranceRow[], models: TtsModelId[]): void {
  console.log("aggregate (medians across all utterances per model):");
  const header = ["model", "first-audio", "total-synth", "rtf", "utterances"];
  const table = models.map((modelId) => {
    const results = rows
      .filter((row) => row.modelId === modelId)
      .map((row) => row.result);
    return [
      modelId,
      formatOptionalMs(
        medianOptional(results.map((result) => result.firstAudioMs)),
      ),
      formatMs(median(results.map((result) => result.totalSynthMs))),
      median(results.map((result) => result.rtf)).toFixed(3),
      String(results.length),
    ];
  });
  printColumns([header, ...table]);
}

/** Left-aligned fixed-width table printer shared by the sweep and qual sweep. */
export function printColumns(rows: string[][]): void {
  const widths = rows[0]!.map((_, column) =>
    Math.max(...rows.map((row) => row[column]!.length)),
  );
  for (const row of rows) {
    console.log(
      row.map((cell, column) => cell.padEnd(widths[column]!)).join("  "),
    );
  }
}
