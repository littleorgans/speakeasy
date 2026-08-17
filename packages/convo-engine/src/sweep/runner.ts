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
  sourceOptions?: WavSourceOptions,
): Promise<SweepResult> {
  const source = new ScriptedWavSource(
    utterances.map((utterance) => utterance.audio),
    {
      silenceTailMs: 0,
      ...sourceOptions,
      onUtteranceEnd: () => loop.commitInput(),
    },
  );
  const turns: SweepTurn[] = [];
  const errors: string[] = [];
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
      errors.push(event.message);
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
    return {
      status: "skipped",
      id: row.id,
      reason:
        errors[0] ??
        `completed ${turns.length} of ${utterances.length} expected turns`,
      turns,
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

function expandUtterances(
  utterances: readonly SweepUtterance[],
  runs: number,
): SweepUtterance[] {
  return Array.from({ length: runs }, () => utterances).flat();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
