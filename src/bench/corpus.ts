import { readCorpusEntries, type CorpusEntry } from "../corpus/store.ts";
import { formatMs } from "./format.ts";
import { createEngine, runPttOnce } from "./harness.ts";
import { median } from "./stats.ts";
import { wordErrorAlignment, type WordAlignment } from "./transcript.ts";
import type { CliOptions } from "./types.ts";
import { readWavFrames } from "./wav.ts";

/**
 * Corpus WER scorer: replays each labeled recording (wav + json sidecar
 * pair) through the push-to-talk path and scores the fresh hypothesis
 * against the hand-confirmed expected transcript. Feeds the model-quality
 * sweep: `--engine` selects the engine, the corpus stays fixed.
 *
 * Release point: the last frame of the capture. Demo recordings end at the
 * release keypress by construction, so end-of-file IS the strict release.
 * The bench's RMS speech-end reference is deliberately not used here: its
 * tail-noise estimate assumes a quiet trailing window, which release-bounded
 * captures lack, inflating the threshold and clipping weak final phonemes
 * (measured: voiced end 1980ms vs true 2140ms on a jfk capture, dropping
 * the final "S").
 */

type CorpusUtteranceResult = {
  entry: CorpusEntry;
  flushToFinalMs: number;
  hypothesis: string;
  alignment: WordAlignment;
};

export async function runCorpusBench(options: CliOptions): Promise<void> {
  const dir = options.corpus;
  if (!dir) {
    throw new Error("runCorpusBench requires --corpus <dir>");
  }
  const entries = await readCorpusEntries(dir);
  const labeled = entries.filter((entry) => entry.sidecar.expected !== null);

  console.log("speak-easy corpus WER scorer");
  console.log(
    `corpus=${dir} sidecars=${entries.length} labeled=${labeled.length} skipped-unlabeled=${entries.length - labeled.length}`,
  );
  if (labeled.length === 0) {
    console.log(
      "no labeled utterances: set \"expected\" in the sidecars by hand, or record with the demo's --save prompt",
    );
    return;
  }

  const engine = createEngine(options.engine, options.model);
  await engine.prepare?.();
  console.log(
    `engine=${engine.label ?? options.engine} endpoint=manual cadence=${options.frameMs}ms/frame release=end-of-capture (demo recordings end at the release keypress)`,
  );

  const results: CorpusUtteranceResult[] = [];
  for (const entry of labeled) {
    const result = await scoreUtterance(engine, entry, options.frameMs);
    printUtterance(result);
    results.push(result);
  }
  printAggregate(results);
}

async function scoreUtterance(
  engine: ReturnType<typeof createEngine>,
  entry: CorpusEntry,
  frameMs: number,
): Promise<CorpusUtteranceResult> {
  const expected = entry.sidecar.expected!;
  const audio = await readWavFrames(entry.wavPath, frameMs);
  const run = await runPttOnce({
    run: 1,
    engine,
    audio,
    releaseFrame: audio.frames.length - 1,
    frameMs,
    expected,
  });
  return {
    entry,
    flushToFinalMs: run.flushToFinalMs,
    hypothesis: run.finalText,
    alignment: wordErrorAlignment(run.finalText, expected),
  };
}

function printUtterance(result: CorpusUtteranceResult): void {
  console.log(
    [
      `utt=${result.entry.sidecar.audio}`,
      `WER=${werPercent(result.alignment).toFixed(1)}%`,
      `errors=${result.alignment.errors}/${result.alignment.referenceWords}`,
      `flush->final=${formatMs(result.flushToFinalMs)}`,
      formatEdits(result.alignment),
      `expected=${JSON.stringify(result.entry.sidecar.expected)}`,
      `hypothesis=${JSON.stringify(result.hypothesis)}`,
    ].join(" "),
  );
}

function printAggregate(results: CorpusUtteranceResult[]): void {
  const errors = results.reduce(
    (sum, result) => sum + result.alignment.errors,
    0,
  );
  const words = results.reduce(
    (sum, result) => sum + result.alignment.referenceWords,
    0,
  );
  const corpusWer = words > 0 ? (errors / words) * 100 : 0;
  const latencies = results
    .map((result) => result.flushToFinalMs)
    .filter((value) => Number.isFinite(value));
  const medianLatency =
    latencies.length > 0 ? formatMs(median(latencies)) : "n/a";
  const maxLatency =
    latencies.length > 0 ? formatMs(Math.max(...latencies)) : "n/a";

  console.log(
    `corpus WER=${corpusWer.toFixed(1)}% errors=${errors}/${words} utterances=${results.length} median flush->final=${medianLatency} max flush->final=${maxLatency}`,
  );
  const worst = tallySubstitutions(results);
  console.log(
    worst.length > 0
      ? `worst substitutions: ${worst
          .map(([pair, count]) => `${pair} x${count}`)
          .join(", ")}`
      : "worst substitutions: none",
  );
}

function tallySubstitutions(
  results: CorpusUtteranceResult[],
): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const result of results) {
    for (const substitution of result.alignment.substitutions) {
      const pair = `${substitution.expected}->${substitution.actual}`;
      counts.set(pair, (counts.get(pair) ?? 0) + 1);
    }
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1]);
}

function formatEdits(alignment: WordAlignment): string {
  const subs = alignment.substitutions
    .map((s) => `${s.expected}->${s.actual}`)
    .join(",");
  return `subs=[${subs}] ins=[${alignment.insertions.join(",")}] del=[${alignment.deletions.join(",")}]`;
}

function werPercent(alignment: WordAlignment): number {
  return alignment.referenceWords > 0
    ? (alignment.errors / alignment.referenceWords) * 100
    : 0;
}
