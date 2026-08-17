import type { WavAudio } from "@speakeasy/speech-io";
import type { TurnMetrics } from "../metrics.ts";
import type { RuntimeConfig } from "../runtime.ts";

export type SweepMatrixRow = {
  id: string;
  config: RuntimeConfig;
};

export type SweepUtterance = {
  id: string;
  audio: WavAudio;
};

export type SweepTurn = {
  configId: string;
  utteranceId: string;
  turnIndex: number;
  metrics: TurnMetrics;
};

export type SweepResult =
  | {
      status: "completed";
      id: string;
      label: string;
      turns: SweepTurn[];
      summary: string;
    }
  | {
      status: "partial";
      id: string;
      label: string;
      turns: SweepTurn[];
      summary: string;
      reason: string;
      failedTurn: number;
    }
  | {
      status: "skipped";
      id: string;
      reason: string;
      turns: [];
    };
