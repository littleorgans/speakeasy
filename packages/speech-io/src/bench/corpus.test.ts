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
import type { SilenceGateMode } from "../engines/offline-session.ts";
import type { BenchModelId } from "./models.ts";
import type { BenchEngine, CliOptions } from "./types.ts";

const FRAME_MS = 20;
const MODEL_A: BenchModelId = "en-2023-06-21";
const MODEL_B: BenchModelId = "en-kroko-2025-08-06";
const OFFLINE: BenchModelId = "whistle-onnx-fp32";

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
  [OFFLINE]: ["and so my fellow americans", "spawn 10 agents", "Open .lilo/docs/index.md first."],
};

/** Commits the next scripted hypothesis on each flush, as a ptt engine does. */
function scriptedEngine(texts: string[]): BenchEngine {
  let next = 0;
  return {
    label: "scripted",
    threads: 2,
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

async function bench(
  overrides: Partial<CliOptions>,
  extra: (dir: string) => Promise<void> = async () => {},
) {
  const dir = await mkdtemp(join(tmpdir(), "speakeasy-corpus-"));
  const created: Array<BenchModelId | undefined> = [];
  const gates: SilenceGateMode[] = [];
  const stdout: string[] = [];
  const log: string[] = [];
  try {
    await writeCorpus(dir);
    await extra(dir);
    const deps: CorpusBenchDeps = {
      createEngine: (model, gate) => {
        created.push(model);
        gates.push(gate);
        return scriptedEngine(HYPOTHESES[model ?? MODEL_A]!);
      },
      log: (line) => log.push(line),
      writeJson: (json) => stdout.push(json),
    };
    const report = await runCorpusBench(cliOptions(dir, overrides), deps);
    return { report, created, gates, stdout, log };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test("every model decodes the same takes and rows pair up per utterance", async () => {
  const { report, created, stdout } = await bench({ models: [MODEL_A, MODEL_B] });

  assert.deepEqual(created, [MODEL_A, MODEL_B]);
  assert.equal(report.sidecars, 4);
  assert.equal(report.labeled, 3);
  assert.equal(report.scored, 3);
  assert.equal(report.superseded, 0);
  assert.deepEqual(report.retakes, []);
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

/** Re-takes of p001: sidecar recordedAt orders them, the file name only when it is missing. */
async function writeRetakes(dir: string): Promise<{ newest: string; skipped: string[] }> {
  const retake = (recordedAt: string, expected: string) =>
    saveCorpusPair(dir, [new Float32Array(320)], 16_000, {
      recordedAt,
      hypothesis: "",
      expected,
      engineLabel: "none",
      endpoint: "hold",
      flushToFinalMs: null,
      device: "test",
      peakLevel: 0.1,
      promptId: "p001",
      stratum: "short",
    });
  const rewriteSidecar = async (path: string, edit: (sidecar: Record<string, unknown>) => void) => {
    const sidecar = JSON.parse(await readFile(path, "utf8"));
    edit(sidecar);
    await writeFile(path, JSON.stringify(sidecar));
  };
  // Named early, recorded last: the sidecar time wins over the file name.
  const newest = await retake("2026-10-31T09:00:00.000Z", "spawn ten agents now");
  await rewriteSidecar(newest.sidecarPath, (sidecar) => {
    sidecar.recordedAt = "2026-10-31T13:00:00.000Z";
  });
  // No sidecar time: falls back to the file name, 12:00.
  const undated = await retake("2026-10-31T12:00:00.000Z", "spawn agents");
  await rewriteSidecar(undated.sidecarPath, (sidecar) => {
    delete sidecar.recordedAt;
  });
  const audio = (path: string) => path.slice(dir.length + 1).replace(/\.json$/, ".wav");
  return {
    newest: audio(newest.sidecarPath),
    skipped: ["utt-p001-2026-10-31T10-00-01.000Z.wav", audio(undated.sidecarPath)],
  };
}

test("the bench scores the newest take per prompt and lists the superseded ones", async () => {
  let takes: { newest: string; skipped: string[] } | undefined;
  const { report, log } = await bench({ models: [MODEL_A] }, async (dir) => {
    takes = await writeRetakes(dir);
  });

  assert.equal(report.sidecars, 6);
  assert.equal(report.labeled, 5);
  assert.equal(report.scored, 3);
  assert.equal(report.superseded, 2);
  // Unprompted takes stay one row each; p001 keeps only its newest take.
  assert.deepEqual(
    report.utterances.map((row) => [row.promptId, row.audio, row.expected]),
    [
      [null, report.utterances[0]!.audio, TAKES[0].expected],
      ["p001", takes!.newest, "spawn ten agents now"],
      ["p002", report.utterances[2]!.audio, TAKES[2].expected],
    ],
  );
  assert.deepEqual(report.retakes, [
    { promptId: "p001", takes: 3, chosen: takes!.newest, skipped: [...takes!.skipped].sort() },
  ]);
  assert.ok(log.some((line) => / labeled=5 .*scored=3 superseded=2/.test(line)), log.join("\n"));
  assert.ok(
    log.some((line) =>
      line === `retake prompt=p001 takes=3 chosen=${takes!.newest} skipped=${[...takes!.skipped].sort().join(",")}`),
    log.join("\n"),
  );
});

test("an offline model scores a gate off and a gate on column from separate engines", async () => {
  const { report, created, gates } = await bench({ models: [MODEL_A, OFFLINE] });
  assert.deepEqual(created, [MODEL_A, OFFLINE, OFFLINE]);
  assert.deepEqual(gates, ["model", "off", "on"]);
  assert.deepEqual(
    report.models.map((summary) => [summary.model, summary.gate]),
    [[MODEL_A, null], [OFFLINE, "off"], [`${OFFLINE}+gate`, "on"]],
  );
  assert.deepEqual(Object.keys(report.utterances[0]!.models), [MODEL_A, OFFLINE, `${OFFLINE}+gate`]);
});

test("summaries split WER and the raw identifier rate by stratum", async () => {
  const { report } = await bench({ models: [MODEL_A, MODEL_B] });
  const [a, b] = report.models;

  // Model A: "ten" for "10" is clean normalized WER but a raw identifier miss.
  assert.equal(a?.model, MODEL_A);
  assert.deepEqual(report.models.map((summary) => summary.threads), [2, 2]);
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
        "corpus", "sidecars", "labeled", "scored", "superseded", "frameMs", "rewrite", "retakes", "models", "utterances",
      ]);
      assert.deepEqual(report.utterances.map((row: { expected: string }) => row.expected), TAKES.map((take) => take.expected));
      assert.ok(report.models[0].raw.all.caseSensitiveIdentifiers);
      assert.equal(report.models[0].threads, null, "the stub sets no threads");
      assert.equal(report.models[0].gate, null, "the stub has no silence gate");
      // Timings are measured anew; every other field and its order is stable.
      return JSON.stringify(JSON.parse(result.stdout, (key, value) => key === "flushToFinalMs" ? undefined : value));
    };
    assert.equal(run(), run());
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
