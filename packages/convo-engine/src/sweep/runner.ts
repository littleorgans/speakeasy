import type { VoiceToText } from "@speakeasy/speech-io";
import { createNullSink } from "../audio/null-sink.ts";
import {
  ScriptedWavSource,
  type WavSourceOptions,
} from "../audio/sources.ts";
import type { ConversationEvent } from "../events.ts";
import { ConversationLoop } from "../loop.ts";
import { formatSessionSummary } from "../metrics.ts";
import type { ConversationRuntime, RuntimeConfig } from "../runtime.ts";
import type {
  SweepMatrixRow,
  SweepResult,
  SweepTurn,
  SweepUtterance,
} from "./types.ts";

type RuntimeFactory = (
  config: RuntimeConfig,
  env: NodeJS.ProcessEnv,
  stt?: VoiceToText,
) => Promise<ConversationRuntime>;

export type SweepInput = {
  rows: readonly SweepMatrixRow[];
  utterances: readonly SweepUtterance[];
  runs: number;
  turnGapMs: number;
};

export type SweepDependencies = {
  createRuntime: RuntimeFactory;
  env?: NodeJS.ProcessEnv;
  sourceOptions?: WavSourceOptions;
  onResult?: (result: SweepResult) => void;
};

export async function runSweep(
  input: SweepInput,
  dependencies: SweepDependencies,
): Promise<SweepResult[]> {
  if (input.utterances.length === 0) {
    throw new Error("The sweep requires at least one utterance");
  }
  if (!Number.isInteger(input.runs) || input.runs < 1) {
    throw new Error(`runs must be a positive integer, received ${input.runs}`);
  }

  const results: SweepResult[] = [];
  let sharedStt: VoiceToText | undefined;
  for (const row of input.rows) {
    let result: SweepResult;
    try {
      const runtime = await dependencies.createRuntime(
        row.config,
        dependencies.env ?? process.env,
        sharedStt,
      );
      sharedStt ??= runtime.stt;
      result = await runRow(
        row,
        runtime,
        expandUtterances(input.utterances, input.runs),
        input.turnGapMs,
        dependencies.sourceOptions,
      );
    } catch (error) {
      result = {
        status: "skipped",
        id: row.id,
        reason: errorMessage(error),
        turns: [],
      };
    }
    results.push(result);
    dependencies.onResult?.(result);
  }
  return results;
}

async function runRow(
  row: SweepMatrixRow,
  runtime: ConversationRuntime,
  utterances: readonly SweepUtterance[],
  turnGapMs: number,
  sourceOptions?: WavSourceOptions,
): Promise<SweepResult> {
  let committedTurns = 0;
  const source = new ScriptedWavSource(
    utterances.map((utterance) => utterance.audio),
    {
      silenceTailMs: 0,
      ...sourceOptions,
      utteranceGapMs: turnGapMs,
      onUtteranceEnd: () => {
        committedTurns += 1;
        loop.commitInput();
      },
    },
  );
  const turns: SweepTurn[] = [];
  const errors: { reason: string; failedTurn: number }[] = [];
  const onEvent = (event: ConversationEvent): void => {
    source.onEvent(event);
    if (event.type === "metrics") {
      const utterance = utterances[event.metrics.turn - 1];
      if (utterance) {
        turns.push({
          configId: row.id,
          utteranceId: utterance.id,
          turnIndex: event.metrics.turn,
          metrics: event.metrics,
        });
      }
    } else if (event.type === "notice" && event.level === "error") {
      errors.push({
        reason: stripTurnPrefix(event.message),
        failedTurn: committedTurns,
      });
    }
  };
  const loop = new ConversationLoop(
    {
      stt: runtime.stt,
      responder: runtime.responder,
      mic: source,
      createSink: createNullSink,
    },
    {
      sttConfig: { endpoint: { mode: "manual" } },
      maxTurns: utterances.length,
      onEvent,
    },
  );
  try {
    await loop.start();
    await Promise.all([loop.done, source.done]);
  } finally {
    await loop.stop();
  }

  if (errors.length > 0 || turns.length !== utterances.length) {
    const firstError = errors[0];
    const reason =
      firstError?.reason ??
      `completed ${turns.length} of ${utterances.length} expected turns`;
    if (turns.length > 0) {
      return {
        status: "partial",
        id: row.id,
        label: runtime.label,
        reason,
        failedTurn: firstError?.failedTurn ?? firstMissingTurn(turns),
        turns,
        summary: formatSessionSummary(turns.map((turn) => turn.metrics)),
      };
    }
    return {
      status: "skipped",
      id: row.id,
      reason,
      turns: [],
    };
  }
  return {
    status: "completed",
    id: row.id,
    label: runtime.label,
    turns,
    summary: formatSessionSummary(turns.map((turn) => turn.metrics)),
  };
}

function firstMissingTurn(turns: readonly SweepTurn[]): number {
  const completed = new Set(turns.map((turn) => turn.turnIndex));
  let turn = 1;
  while (completed.has(turn)) {
    turn += 1;
  }
  return turn;
}

function stripTurnPrefix(message: string): string {
  return message.replace(/^turn \d+ \| /, "");
}

function expandUtterances(
  utterances: readonly SweepUtterance[],
  runs: number,
): SweepUtterance[] {
  return Array.from({ length: runs }, () => utterances).flat();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
