import type { NumbersMode, RewriteConfig } from "../contract.ts";
import { readCorpusEntries, type CorpusEntry } from "../corpus/store.ts";
import { rewriteText } from "../rewrite/pipeline.ts";
import { DEFAULT_RULES } from "../rewrite/rules.ts";
import { formatMs } from "./format.ts";
import { createEngine, runPttOnce } from "./harness.ts";
import { scoreIdentifiers } from "./identifiers.ts";
import type { SilenceGateMode } from "../engines/offline-session.ts";
import { isOfflineBenchModel, type BenchModelId } from "./models.ts";
import { median } from "./stats.ts";
import { wordErrorAlignment, type WordSubstitution } from "./transcript.ts";
import type { BenchEngine, CliOptions } from "./types.ts";
import { readWavFrames } from "./wav.ts";

/**
 * Corpus scorer: replays each labeled recording (wav + json sidecar pair)
 * through the push-to-talk path once per model and scores the fresh
 * hypothesis against the expected transcript. Every model decodes the same
 * takes, so rows pair up per utterance for model comparison.
 *
 * Scores: normalized WER (`wordErrorAlignment`) and the identifier error rate
 * on raw tokens (`scoreIdentifiers`), ignoring casing with a case sensitive
 * second column, overall and per sidecar stratum. With
 * `--rewrite on` each take also gets a rewrite column: the same decode passed
 * through `rewriteText`, the function the production `withRewrite` decorator
 * applies, so raw and rewritten scores never differ by decoder noise.
 *
 * Offline models score twice, with the silence gate off (column `<id>`) and
 * on (column `<id>+gate`), whatever their descriptor ships, so the report
 * shows what the gate costs on speech and saves on silence.
 *
 * Release point: the last frame of the capture. Demo recordings end at the
 * release keypress by construction, so end-of-file IS the strict release.
 * The bench's RMS speech-end reference is deliberately not used here: its
 * tail-noise estimate assumes a quiet trailing window, which release-bounded
 * captures lack, inflating the threshold and clipping weak final phonemes
 * (measured: voiced end 1980ms vs true 2140ms on a jfk capture, dropping
 * the final "S").
 */

export const NO_STRATUM = "unstratified";

export type CorpusBenchDeps = {
  /** Engine for one model column; undefined is the engine's default model. */
  createEngine: (
    model: BenchModelId | undefined,
    gate: SilenceGateMode,
  ) => BenchEngine | Promise<BenchEngine>;
  /** Human-readable lines; stderr under --json so stdout is only the report. */
  log: (line: string) => void;
  /** Receives the report document under --json. */
  writeJson: (json: string) => void;
};

export type TextScore = {
  hypothesis: string;
  wordErrors: number;
  referenceWords: number;
  substitutions: WordSubstitution[];
  insertions: string[];
  deletions: string[];
  identifierErrors: number;
  caseSensitiveIdentifierErrors: number;
  identifiers: number;
  missedIdentifiers: string[];
  missedCaseSensitiveIdentifiers: string[];
};

export type ModelTake = {
  /** Null when the engine never emitted a final after flush(). */
  flushToFinalMs: number | null;
  raw: TextScore;
  rewrite: TextScore | null;
};

export type CorpusRow = {
  audio: string;
  promptId: string | null;
  stratum: string | null;
  expected: string;
  /** Keyed by model column (registry id, or engine name for its default). */
  models: Record<string, ModelTake>;
};

export type Tally = { errors: number; total: number; rate: number | null };

export type CorpusAggregate = {
  utterances: number;
  wer: Tally;
  identifiers: Tally;
  caseSensitiveIdentifiers: Tally;
};

export type ScoreTable = {
  all: CorpusAggregate;
  /** Keyed by sidecar stratum, NO_STRATUM for unprompted takes. */
  strata: Record<string, CorpusAggregate>;
};

export type ModelSummary = {
  /** Column key: registry id, engine name for its default, `+gate` when gated. */
  model: string;
  engineLabel: string;
  /** Silence gate of an offline model's column; null for engines without one. */
  gate: "on" | "off" | null;
  /** The engine's thread setting, null when its runtime picks. */
  threads: number | null;
  raw: ScoreTable;
  rewrite: ScoreTable | null;
  flushToFinalMs: { median: number | null; max: number | null };
  worstSubstitutions: Array<[string, number]>;
};

export type CorpusReport = {
  corpus: string;
  sidecars: number;
  labeled: number;
  frameMs: number;
  rewrite: { rules: number; numbers: NumbersMode } | null;
  models: ModelSummary[];
  utterances: CorpusRow[];
};

export async function runCorpusBench(
  options: CliOptions,
  deps: Partial<CorpusBenchDeps> = {},
): Promise<CorpusReport> {
  const dir = options.corpus;
  if (!dir) {
    throw new Error("runCorpusBench requires --corpus <dir>");
  }
  const log =
    deps.log ?? ((line: string) => (options.json ? console.error : console.log)(line));
  const writeJson =
    deps.writeJson ?? ((json: string) => process.stdout.write(`${json}\n`));
  const openEngine =
    deps.createEngine ?? ((model, gate) => createEngine(options.engine, model, gate));
  const rewrite: RewriteConfig | null = options.rewrite
    ? { rules: DEFAULT_RULES, numbers: options.numbers }
    : null;

  const entries = await readCorpusEntries(dir);
  const labeled = entries.filter((entry) => entry.sidecar.expected !== null);
  const report: CorpusReport = {
    corpus: dir,
    sidecars: entries.length,
    labeled: labeled.length,
    frameMs: options.frameMs,
    rewrite: rewrite && { rules: rewrite.rules.length, numbers: rewrite.numbers },
    models: [],
    utterances: labeled.map((entry) => ({
      audio: entry.sidecar.audio,
      promptId: entry.sidecar.promptId,
      stratum: entry.sidecar.stratum,
      expected: entry.sidecar.expected!,
      models: {},
    })),
  };

  log("speak-easy corpus scorer (normalized WER + raw identifier rate)");
  log(
    `corpus=${dir} sidecars=${entries.length} labeled=${labeled.length} skipped-unlabeled=${entries.length - labeled.length}`,
  );
  if (labeled.length === 0) {
    log(
      "no labeled utterances: set \"expected\" in the sidecars by hand, record with the demo's --save prompt, or read a prompt set in the browser prompt reader",
    );
  }

  const models = labeled.length > 0 ? (options.models ?? [options.model]) : [];
  const columns = models.flatMap((model) => gateColumns(model, options.engine));
  for (const { model, gate, column } of columns) {
    const engine = await openEngine(model, gate);
    await engine.prepare?.();
    const engineLabel = engine.label ?? options.engine;
    const threads = engine.threads ?? null;
    const gateLabel = gate === "model" ? null : gate;
    log(
      `model=${column} engine=${engineLabel} threads=${threads ?? "runtime-default"} gate=${gateLabel ?? "n/a"} endpoint=manual cadence=${options.frameMs}ms/frame release=end-of-capture rewrite=${rewrite ? `column(rules=${rewrite.rules.length},numbers=${rewrite.numbers})` : "off"}`,
    );
    for (const [index, entry] of labeled.entries()) {
      const row = report.utterances[index]!;
      row.models[column] = await scoreTake(engine, entry, options.frameMs, rewrite);
      log(formatTake(column, row));
    }
    report.models.push(
      summarize({ column, engineLabel, gate: gateLabel, threads }, report.utterances),
    );
  }

  for (const summary of report.models) {
    printSummary(summary, log);
  }
  if (options.json) {
    writeJson(JSON.stringify(report, null, 2));
  }
  return report;
}

type Column = { model: BenchModelId | undefined; gate: SilenceGateMode; column: string };

/** An offline model's gate off and gate on columns; one column otherwise. */
function gateColumns(model: BenchModelId | undefined, engine: string): Column[] {
  if (model && isOfflineBenchModel(model)) {
    return [
      { model, gate: "off", column: model },
      { model, gate: "on", column: `${model}+gate` },
    ];
  }
  return [{ model, gate: "model", column: model ?? engine }];
}

async function scoreTake(
  engine: BenchEngine,
  entry: CorpusEntry,
  frameMs: number,
  rewrite: RewriteConfig | null,
): Promise<ModelTake> {
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
    flushToFinalMs: Number.isFinite(run.flushToFinalMs) ? run.flushToFinalMs : null,
    raw: scoreText(run.finalText, expected),
    rewrite: rewrite && scoreText(rewriteText(run.finalText, rewrite), expected),
  };
}

function scoreText(hypothesis: string, expected: string): TextScore {
  const alignment = wordErrorAlignment(hypothesis, expected);
  const identifiers = scoreIdentifiers(hypothesis, expected);
  const caseSensitive = scoreIdentifiers(hypothesis, expected, { caseSensitive: true });
  return {
    hypothesis,
    wordErrors: alignment.errors,
    referenceWords: alignment.referenceWords,
    substitutions: alignment.substitutions,
    insertions: alignment.insertions,
    deletions: alignment.deletions,
    identifierErrors: identifiers.errors,
    caseSensitiveIdentifierErrors: caseSensitive.errors,
    identifiers: identifiers.identifiers,
    missedIdentifiers: identifiers.missed,
    missedCaseSensitiveIdentifiers: caseSensitive.missed,
  };
}

function summarize(
  { column, engineLabel, gate, threads }: Pick<ModelSummary, "engineLabel" | "gate" | "threads"> & {
    column: string;
  },
  rows: CorpusRow[],
): ModelSummary {
  const takes = rows.map((row) => ({ row, take: row.models[column]! }));
  const latencies = takes
    .map(({ take }) => take.flushToFinalMs)
    .filter((value): value is number => value !== null);
  const counts = new Map<string, number>();
  for (const { take } of takes) {
    for (const substitution of take.raw.substitutions) {
      const pair = `${substitution.expected}->${substitution.actual}`;
      counts.set(pair, (counts.get(pair) ?? 0) + 1);
    }
  }
  const hasRewrite = takes.some(({ take }) => take.rewrite !== null);
  return {
    model: column,
    engineLabel,
    gate,
    threads,
    raw: tabulate(takes.map(({ row, take }) => ({ row, score: take.raw }))),
    rewrite: hasRewrite
      ? tabulate(takes.map(({ row, take }) => ({ row, score: take.rewrite! })))
      : null,
    flushToFinalMs: {
      median: latencies.length > 0 ? median(latencies) : null,
      max: latencies.length > 0 ? Math.max(...latencies) : null,
    },
    worstSubstitutions: [...counts.entries()].sort(
      (left, right) => right[1] - left[1],
    ),
  };
}

function tabulate(scores: Array<{ row: CorpusRow; score: TextScore }>): ScoreTable {
  const groups = new Map<string, TextScore[]>();
  for (const { row, score } of scores) {
    const key = row.stratum ?? NO_STRATUM;
    groups.set(key, [...(groups.get(key) ?? []), score]);
  }
  return {
    all: aggregate(scores.map(({ score }) => score)),
    strata: Object.fromEntries(
      [...groups].map(([stratum, group]) => [stratum, aggregate(group)]),
    ),
  };
}

function aggregate(scores: TextScore[]): CorpusAggregate {
  return {
    utterances: scores.length,
    wer: tally(scores, "wordErrors", "referenceWords"),
    identifiers: tally(scores, "identifierErrors", "identifiers"),
    caseSensitiveIdentifiers: tally(scores, "caseSensitiveIdentifierErrors", "identifiers"),
  };
}

function tally(
  scores: TextScore[],
  errorsKey: "wordErrors" | "identifierErrors" | "caseSensitiveIdentifierErrors",
  totalKey: "referenceWords" | "identifiers",
): Tally {
  const errors = scores.reduce((sum, score) => sum + score[errorsKey], 0);
  const total = scores.reduce((sum, score) => sum + score[totalKey], 0);
  return { errors, total, rate: total > 0 ? errors / total : null };
}

function formatTake(column: string, row: CorpusRow): string {
  const take = row.models[column]!;
  return [
    `model=${column}`,
    `utt=${row.audio}`,
    `stratum=${row.stratum ?? NO_STRATUM}`,
    `WER=${formatRate({ errors: take.raw.wordErrors, total: take.raw.referenceWords })}`,
    `ident=${formatRate({ errors: take.raw.identifierErrors, total: take.raw.identifiers })}`,
    `ident-case=${formatRate({ errors: take.raw.caseSensitiveIdentifierErrors, total: take.raw.identifiers })}`,
    ...(take.rewrite
      ? [`WER+rewrite=${formatRate({ errors: take.rewrite.wordErrors, total: take.rewrite.referenceWords })}`]
      : []),
    `flush->final=${take.flushToFinalMs === null ? "n/a" : formatMs(take.flushToFinalMs)}`,
    formatEdits(take.raw),
    `expected=${JSON.stringify(row.expected)}`,
    `hypothesis=${JSON.stringify(take.raw.hypothesis)}`,
  ].join(" ");
}

function printSummary(summary: ModelSummary, log: (line: string) => void): void {
  const lines = [
    { label: "all", raw: summary.raw.all, rewrite: summary.rewrite?.all },
    ...Object.entries(summary.raw.strata).map(([label, raw]) => ({
      label,
      raw,
      rewrite: summary.rewrite?.strata[label],
    })),
  ];
  for (const { label, raw, rewrite } of lines) {
    log(
      [
        `model=${summary.model}`,
        `stratum=${label}`,
        `utterances=${raw.utterances}`,
        `WER=${formatRate(raw.wer)}`,
        `ident=${formatRate(raw.identifiers)}`,
        `ident-case=${formatRate(raw.caseSensitiveIdentifiers)}`,
        ...(rewrite
          ? [`WER+rewrite=${formatRate(rewrite.wer)}`, `ident+rewrite=${formatRate(rewrite.identifiers)}`, `ident-case+rewrite=${formatRate(rewrite.caseSensitiveIdentifiers)}`]
          : []),
      ].join(" "),
    );
  }
  const latency = summary.flushToFinalMs;
  log(
    `model=${summary.model} median flush->final=${latency.median === null ? "n/a" : formatMs(latency.median)} max flush->final=${latency.max === null ? "n/a" : formatMs(latency.max)}`,
  );
  log(
    summary.worstSubstitutions.length > 0
      ? `model=${summary.model} worst substitutions: ${summary.worstSubstitutions
          .map(([pair, count]) => `${pair} x${count}`)
          .join(", ")}`
      : `model=${summary.model} worst substitutions: none`,
  );
}

function formatRate(tally: Pick<Tally, "errors" | "total">): string {
  const percent = tally.total > 0 ? ((tally.errors / tally.total) * 100).toFixed(1) : "n/a";
  return `${percent}${tally.total > 0 ? "%" : ""}(${tally.errors}/${tally.total})`;
}

function formatEdits(score: TextScore): string {
  const subs = score.substitutions
    .map((s) => `${s.expected}->${s.actual}`)
    .join(",");
  return `subs=[${subs}] ins=[${score.insertions.join(",")}] del=[${score.deletions.join(",")}] missed-ident=[${score.missedIdentifiers.join(",")}]`;
}
