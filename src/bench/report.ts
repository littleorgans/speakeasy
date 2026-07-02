import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  EXPECTED_JFK_TRANSCRIPT,
  PARTIAL_ROOTCAUSE,
  PASS_THRESHOLD_MS,
} from "./config.ts";
import { formatBoolean, formatEndpoint, formatMs } from "./format.ts";
import type { SpeechProfile } from "./speech.ts";
import { DEFAULT_MAX_WORD_ERRORS } from "./transcript.ts";
import type { PttRunResult, PttSummary, RunResult, Summary } from "./types.ts";

const SWEEP_SECTION_HEADER = "# sherpa endpoint sweep";
const PTT_SECTION_HEADER = "# ptt mode";

export async function writeSherpaSweep(
  path: string,
  summaries: Summary[],
  knee: Summary | undefined,
  selected: Summary | undefined,
  speechProfile: SpeechProfile,
): Promise<void> {
  const rows = summaries.map((summary) =>
    [
      formatEndpoint(summary.endpoint),
      formatMs(summary.speechEndToFinalMedian),
      formatBoolean(summary.textCorrect),
      JSON.stringify(mostCommonFinalText(summary.results)),
    ].join(" | "),
  );

  await upsertReportSection(path, SWEEP_SECTION_HEADER, [
    `engine: ${summaries[0]?.engineLabel ?? "unknown"}`,
    `speech-end-reference: rms-window ${speechProfile.windowMs}ms + ${speechProfile.hangoverMs}ms hangover, threshold=${speechProfile.threshold.toFixed(4)}, end=${speechProfile.endMs.toFixed(1)}ms`,
    `expected: ${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} (case/punct-insensitive exact)`,
    `knee: ${knee ? formatEndpoint(knee.endpoint) : "none"}`,
    `selected: ${selected ? formatEndpoint(selected.endpoint) : "none"}`,
    `partial-rootcause: ${PARTIAL_ROOTCAUSE}; decode is called whenever sherpa reports readiness per pushed frame, but this model emits no non-empty result until its chunk/right-context is satisfied`,
    "",
    "config | perceived endpoint->final median | text-correct | finalText",
    "--- | ---: | :---: | ---",
    ...rows,
  ]);
}

export async function writePttReport(
  path: string,
  summaries: PttSummary[],
  speechProfile: SpeechProfile,
): Promise<void> {
  const lines: string[] = [
    `engine: ${summaries[0]?.engineLabel ?? "unknown"}`,
    "scenario: push-to-talk; frames fed in real time up to the release point, then flush() immediately; engine endpointing=manual (disabled); flush->final measured on the final EVENT (no synchrony assumed)",
    `release points: strict=${speechProfile.voicedEndMs.toFixed(1)}ms (last voiced sample; honest headline, no real trailing audio after release); loose=${speechProfile.endMs.toFixed(1)}ms (rms reference incl. ${speechProfile.hangoverMs}ms hangover)`,
    "flush: synthetic silence pushed instantly at flush() to satisfy the model's chunk/right-context, then decode+commit; session stays open",
    `expected: ${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} gate=word-tolerant (word-error-count <= ${DEFAULT_MAX_WORD_ERRORS} after case/punct normalization)`,
    `pass-rule: warm median flush->final < ${PASS_THRESHOLD_MS}ms AND all runs text-correct`,
  ];

  for (const summary of summaries) {
    lines.push(
      "",
      `## ${summary.variant} (release at ${summary.releaseMs.toFixed(1)}ms)`,
      "",
      "run | flush->final | text-correct | word-errors | finalText",
      "--- | ---: | :---: | ---: | ---",
      ...summary.results.map(formatPttRow),
      "",
      `runs=${summary.results.length} cold=${formatMs(summary.coldFlushToFinalMs)} warm-median=${formatMs(summary.warmFlushToFinalMedianMs)} text-correct=${formatBoolean(summary.textCorrect)} result=${summary.passFail}`,
    );
  }

  await upsertReportSection(path, PTT_SECTION_HEADER, lines);
}

function formatPttRow(result: PttRunResult): string {
  return [
    result.run === 1 ? "1 (cold)" : String(result.run),
    formatMs(result.flushToFinalMs),
    formatBoolean(result.textCorrect),
    String(result.wordErrors),
    JSON.stringify(result.finalText),
  ].join(" | ");
}

function mostCommonFinalText(results: RunResult[]): string {
  const counts = new Map<string, number>();
  for (const result of results) {
    counts.set(result.finalText, (counts.get(result.finalText) ?? 0) + 1);
  }
  return (
    [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? ""
  );
}

/**
 * Section-scoped report writer. Each bench mode owns one "# <title>" section
 * in the shared results file; rewriting a section leaves the others intact,
 * so a sweep re-run never clobbers the ptt table and vice versa.
 */
export async function upsertReportSection(
  path: string,
  header: string,
  lines: string[],
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const existing = await readFile(path, "utf8").catch(() => "");
  const section = [header, "", ...lines, ""].join("\n");
  const remainder = removeSection(existing, header).trimEnd();
  const content = remainder ? `${remainder}\n\n${section}` : section;
  await writeFile(path, content);
}

function removeSection(content: string, header: string): string {
  const lines = content.split("\n");
  const start = lines.findIndex((line) => line.trim() === header);
  if (start === -1) {
    return content;
  }
  let end = start + 1;
  while (end < lines.length && !lines[end]!.startsWith("# ")) {
    end += 1;
  }
  return [...lines.slice(0, start), ...lines.slice(end)].join("\n");
}
