import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { STTSession } from "../contract.ts";
import { saveCorpusPair, type CorpusSidecarMeta } from "../corpus/store.ts";
import { NO_STRATUM, runCorpusBench, type CorpusBenchDeps } from "./corpus.ts";
import type { BenchModelId } from "./models.ts";
import type { BenchEngine, CliOptions } from "./types.ts";

const FRAME_MS = 20;
const MODEL_A: BenchModelId = "en-2023-06-21";
const MODEL_B: BenchModelId = "en-kroko-2025-08-06";

/** Takes in filename order: the schema 1 free take sorts before prompted ones. */
const TAKES = [
  { promptId: null, stratum: null, expected: "and so my fellow americans" },
  { promptId: "p001", stratum: "short", expected: "spawn 10 agents" },
  { promptId: "p002", stratum: "long", expected: "Open .lilo/docs/index.md first." },
] as const;

/** Per model, the hypothesis each take decodes to, in take order. */
const HYPOTHESES: Record<string, string[]> = {
  [MODEL_A]: ["and so my fellow americans", "spawn ten agents", "OPEN .LILO/DOCS/INDEX.MD FIRST."],
  [MODEL_B]: ["and saw my fellow americans", "spawn 10 agents", "open lilo docs index md first"],
};

/** Commits the next scripted hypothesis on each flush, as a ptt engine does. */
function scriptedEngine(texts: string[]): BenchEngine {
  let next = 0;
  return {
    label: "scripted",
    async open(): Promise<STTSession> {
      const session = new EventEmitter() as STTSession;
      Object.assign(session, {
        pushAudio() {},
        reset() {},
        flush() {
          session.emit("endpoint", {});
          session.emit("final", { text: texts[next++] ?? "" });
        },
        async end() {},
      });
      return session;
    },
  };
}

async function writeCorpus(dir: string): Promise<void> {
  for (const [index, take] of TAKES.entries()) {
    const meta: CorpusSidecarMeta = {
      recordedAt: `2026-10-31T10:00:0${index}.000Z`,
      hypothesis: "",
      expected: take.expected,
      engineLabel: "none",
      endpoint: "hold",
      flushToFinalMs: null,
      device: "test",
      peakLevel: 0.1,
      ...(take.promptId ? { promptId: take.promptId, stratum: take.stratum } : {}),
    };
    const frames = [new Float32Array(320), new Float32Array(320)];
    const saved = await saveCorpusPair(dir, frames, 16_000, meta);
    if (take.promptId === null) {
      // Downgrade to a schema 1 sidecar, as the demo wrote before schema 2.
      const { promptId: _p, stratum: _s, ...rest } = JSON.parse(
        await readFile(saved.sidecarPath, "utf8"),
      );
      await writeFile(saved.sidecarPath, JSON.stringify({ ...rest, schema: 1, flushToFinalMs: 0 }));
    }
  }
  // Unlabeled sidecars are counted but never scored.
  await saveCorpusPair(dir, [new Float32Array(320)], 16_000, {
    recordedAt: "2026-10-31T11:00:00.000Z",
    hypothesis: "",
    expected: null,
    engineLabel: "none",
    endpoint: "hold",
    flushToFinalMs: null,
    device: "test",
    peakLevel: 0.1,
    promptId: "p003",
    stratum: "short",
  });
}

function cliOptions(corpus: string, overrides: Partial<CliOptions>): CliOptions {
  return {
    engine: "sherpa",
    runs: 5,
    frameMs: FRAME_MS,
    mode: "sweep",
    rewrite: false,
    numbers: "off",
    json: false,
    corpus,
    ...overrides,
  };
}

async function bench(overrides: Partial<CliOptions>) {
  const dir = await mkdtemp(join(tmpdir(), "speakeasy-corpus-"));
  const created: Array<BenchModelId | undefined> = [];
  const stdout: string[] = [];
  const log: string[] = [];
  try {
    await writeCorpus(dir);
    const deps: CorpusBenchDeps = {
      createEngine: (model) => {
        created.push(model);
        return scriptedEngine(HYPOTHESES[model ?? MODEL_A]!);
      },
      log: (line) => log.push(line),
      writeJson: (json) => stdout.push(json),
    };
    const report = await runCorpusBench(cliOptions(dir, overrides), deps);
    return { report, created, stdout, log };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test("every model decodes the same takes and rows pair up per utterance", async () => {
  const { report, created, stdout } = await bench({ models: [MODEL_A, MODEL_B] });

  assert.deepEqual(created, [MODEL_A, MODEL_B]);
  assert.equal(report.sidecars, 4);
  assert.equal(report.labeled, 3);
  assert.equal(report.rewrite, null);
  assert.deepEqual(
    report.utterances.map((row) => [row.promptId, row.stratum, Object.keys(row.models)]),
    [
      [null, null, [MODEL_A, MODEL_B]],
      ["p001", "short", [MODEL_A, MODEL_B]],
      ["p002", "long", [MODEL_A, MODEL_B]],
    ],
  );
  for (const [index, row] of report.utterances.entries()) {
    assert.equal(row.models[MODEL_A]?.raw.hypothesis, HYPOTHESES[MODEL_A]![index]);
    assert.equal(row.models[MODEL_B]?.raw.hypothesis, HYPOTHESES[MODEL_B]![index]);
    assert.equal(row.models[MODEL_A]?.rewrite, null);
    assert.equal(typeof row.models[MODEL_A]?.flushToFinalMs, "number");
  }
  assert.equal(stdout.length, 0, "no JSON without --json");
});

test("summaries split WER and the raw identifier rate by stratum", async () => {
  const { report } = await bench({ models: [MODEL_A, MODEL_B] });
  const [a, b] = report.models;

  // Model A: "ten" for "10" is clean normalized WER but a raw identifier miss.
  assert.equal(a?.model, MODEL_A);
  assert.deepEqual(a?.raw.all.wer, { errors: 0, total: 14, rate: 0 });
  assert.deepEqual(a?.raw.all.identifiers, { errors: 1, total: 2, rate: 0.5 });
  assert.deepEqual(a?.raw.all.caseSensitiveIdentifiers, { errors: 2, total: 2, rate: 1 });
  assert.deepEqual(a?.raw.strata.long?.caseSensitiveIdentifiers, { errors: 1, total: 1, rate: 1 });
  const path = report.utterances[2]!.models[MODEL_A]!.raw;
  assert.equal(path.identifierErrors, 0);
  assert.equal(path.caseSensitiveIdentifierErrors, 1);
  assert.deepEqual(path.missedCaseSensitiveIdentifiers, [".lilo/docs/index.md"]);
  assert.deepEqual(Object.keys(a?.raw.strata ?? {}), [NO_STRATUM, "short", "long"]);
  assert.deepEqual(a?.raw.strata.short?.identifiers, { errors: 1, total: 1, rate: 1 });
  assert.deepEqual(a?.raw.strata[NO_STRATUM]?.identifiers, { errors: 0, total: 0, rate: null });

  // Model B: one substitution; the spelled-out path keeps WER low but misses the identifier.
  assert.deepEqual(b?.raw.all.wer, { errors: 1, total: 14, rate: 1 / 14 });
  assert.deepEqual(b?.raw.strata.long?.identifiers, { errors: 1, total: 1, rate: 1 });
  assert.deepEqual(b?.worstSubstitutions, [["so->saw", 1]]);
  assert.equal(b?.rewrite, null);
});

test("rewrite on adds a column scored from the same decode", async () => {
  const { report } = await bench({ models: [MODEL_A], rewrite: true, numbers: "digits" });
  const take = report.utterances[1]!.models[MODEL_A]!;
  assert.equal(take.raw.hypothesis, "spawn ten agents");
  assert.equal(take.rewrite?.hypothesis, "spawn 10 agents");
  assert.deepEqual(report.rewrite, { rules: report.rewrite!.rules, numbers: "digits" });
  assert.deepEqual(report.models[0]?.raw.all.identifiers, { errors: 1, total: 2, rate: 0.5 });
  assert.deepEqual(report.models[0]?.rewrite?.all.identifiers, { errors: 0, total: 2, rate: 0 });
});

test("--json writes one parseable report; a single model column keys by engine default", async () => {
  const { report, created, stdout, log } = await bench({ json: true });
  assert.deepEqual(created, [undefined]);
  assert.deepEqual(Object.keys(report.utterances[0]!.models), ["sherpa"]);
  assert.equal(stdout.length, 1);
  assert.deepEqual(JSON.parse(stdout[0]!), report);
  assert.ok(log.some((line) => line.startsWith("model=sherpa stratum=all")));
});

test("the real --json CLI emits stable scoring data across repeated runs", async () => {
  const dir = await mkdtemp(join(tmpdir(), "speakeasy-corpus-"));
  try {
    await writeCorpus(dir);
    const run = () => {
      const result = spawnSync(process.execPath, [
        "packages/speech-io/src/bench/run.ts", "--corpus", dir, "--engine", "stub", "--json",
      ], { cwd: new URL("../../../../", import.meta.url), encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
      assert.match(result.stderr, /model=stub stratum=all/);
      const report = JSON.parse(result.stdout);
      assert.deepEqual(Object.keys(report), [
        "corpus", "sidecars", "labeled", "frameMs", "rewrite", "models", "utterances",
      ]);
      assert.deepEqual(report.utterances.map((row: { expected: string }) => row.expected), TAKES.map((take) => take.expected));
      assert.ok(report.models[0].raw.all.caseSensitiveIdentifiers);
      // Timings are measured anew; every other field and its order is stable.
      return JSON.stringify(JSON.parse(result.stdout, (key, value) => key === "flushToFinalMs" ? undefined : value));
    };
    assert.equal(run(), run());
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
