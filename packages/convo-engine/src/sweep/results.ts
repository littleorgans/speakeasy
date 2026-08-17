import { writeFile } from "node:fs/promises";
import { formatMs, median, upsertReportSection } from "@speakeasy/speech-io";
import type { SweepResult, SweepTurn } from "./types.ts";

const REPORT_HEADER = "# cascade sweep";
const TABLE_HEADER =
  "config | status | cold stt-final | warm stt-final | cold first-token | warm first-token | cold first-audio | warm first-audio | turns | reason";
const TABLE_RULE = "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---";

export type SweepOutput = {
  textPath: string;
  jsonPath: string;
  generatedAt: string;
  utteranceIds: readonly string[];
  results: readonly SweepResult[];
};

export async function writeSweepResults(output: SweepOutput): Promise<void> {
  await upsertReportSection(output.textPath, REPORT_HEADER, [
    `generated: ${output.generatedAt}`,
    `utterances: ${output.utteranceIds.join(", ")}`,
    "",
    TABLE_HEADER,
    TABLE_RULE,
    ...output.results.map(formatResultRow),
  ]);
  const turns = output.results.flatMap((result) => result.turns);
  await writeFile(output.jsonPath, `${JSON.stringify(turns, null, 2)}\n`);
}

export function formatResultRow(result: SweepResult): string {
  if (result.status === "skipped") {
    return [
      result.id,
      result.status,
      "n/a",
      "n/a",
      "n/a",
      "n/a",
      "n/a",
      "n/a",
      String(result.turns.length),
      cleanCell(result.reason),
    ].join(" | ");
  }

  return [
    result.id,
    result.status,
    ...formatColdAndWarm(
      result.turns,
      (turn) => turn.metrics.endpointToFinalMs,
    ),
    ...formatColdAndWarm(
      result.turns,
      (turn) => turn.metrics.endpointToFirstTokenMs,
    ),
    ...formatColdAndWarm(
      result.turns,
      (turn) => turn.metrics.endpointToFirstAudioMs,
    ),
    String(result.turns.length),
    "",
  ]
    .join(" | ")
    .trimEnd();
}

function formatColdAndWarm(
  turns: readonly SweepTurn[],
  pick: (turn: SweepTurn) => number,
): [cold: string, warm: string] {
  const cold = turns[0];
  const warm = turns.slice(1);
  return [
    cold ? formatMs(pick(cold)) : "n/a",
    warm.length > 0 ? formatMs(median(warm.map(pick))) : "n/a",
  ];
}

function cleanCell(value: string): string {
  return value.replaceAll("|", "/").replaceAll("\n", " ");
}
