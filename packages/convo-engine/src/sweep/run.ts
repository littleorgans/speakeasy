import { basename, join } from "node:path";
import { parseArgs } from "node:util";
import {
  CAPTURE_FRAME_MS,
  DEFAULT_CORPUS_DIR,
  readCorpusEntries,
  readWavFrames,
} from "@speakeasy/speech-io";
import { createConversationRuntime } from "../runtime.ts";
import { expandMatrix } from "./matrix.ts";
import { writeSweepResults } from "./results.ts";
import { runSweep } from "./runner.ts";
import type { SweepResult, SweepUtterance } from "./types.ts";

export const DEFAULT_UTTERANCE_IDS = [
  "utt-2026-07-03T20-26-00.828Z",
  "utt-2026-07-03T20-32-04.686Z",
  "utt-2026-07-03T20-30-23.115Z",
] as const;

type CliOptions = {
  only: string | undefined;
  utteranceIds: string[];
  runs: number;
};

async function main(): Promise<void> {
  const options = parseCliOptions(process.argv.slice(2));
  const matrix = expandMatrix();
  const rows = options.only
    ? matrix.filter((row) => row.id === options.only)
    : matrix;
  if (rows.length === 0) {
    throw new Error(`Unknown config id "${options.only}"`);
  }

  const utterances = await loadUtterances(options.utteranceIds);
  const results = await runSweep(
    { rows, utterances, runs: options.runs },
    {
      createRuntime: createConversationRuntime,
      onResult: printResult,
    },
  );
  const generatedAt = new Date().toISOString();
  await writeSweepResults({
    textPath: join(process.cwd(), "results/cascade-sweep.txt"),
    jsonPath: join(process.cwd(), "results/cascade-sweep.json"),
    generatedAt,
    utteranceIds: options.utteranceIds,
    results,
  });
}

function parseCliOptions(args: string[]): CliOptions {
  const { values } = parseArgs({
    args,
    options: {
      only: { type: "string" },
      utterances: { type: "string" },
      runs: { type: "string" },
    },
    strict: true,
    allowPositionals: false,
  });
  const utteranceIds = (values.utterances ?? DEFAULT_UTTERANCE_IDS.join(","))
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (utteranceIds.length === 0) {
    throw new Error("--utterances must name at least one corpus id");
  }
  const runs = Number(values.runs ?? "1");
  if (!Number.isInteger(runs) || runs < 1) {
    throw new Error(`--runs must be a positive integer, received "${values.runs}"`);
  }
  return { only: values.only, utteranceIds, runs };
}

async function loadUtterances(ids: readonly string[]): Promise<SweepUtterance[]> {
  const entries = await readCorpusEntries(DEFAULT_CORPUS_DIR);
  const entriesById = new Map(
    entries.map((entry) => [basename(entry.sidecarPath, ".json"), entry]),
  );
  return Promise.all(
    ids.map(async (id) => {
      const entry = entriesById.get(id);
      if (!entry) {
        throw new Error(`Corpus utterance "${id}" was not found`);
      }
      return {
        id,
        audio: await readWavFrames(entry.wavPath, CAPTURE_FRAME_MS),
      };
    }),
  );
}

function printResult(result: SweepResult): void {
  if (result.status === "skipped") {
    console.log(`${result.id}: skipped (${result.reason})`);
    return;
  }
  console.log(`${result.id}\n${result.summary}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
