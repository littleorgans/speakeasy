import type { RuntimeConfig } from "../runtime.ts";
import type { SweepMatrixRow } from "./types.ts";

const CASCADE_MODELS = [
  { id: "cerebras", config: { llmProvider: "cerebras" } },
  {
    id: "mercury-instant",
    config: { llmProvider: "mercury", llmReasoningEffort: "instant" },
  },
  {
    id: "mercury-low",
    config: { llmProvider: "mercury", llmReasoningEffort: "low" },
  },
] satisfies readonly {
  id: string;
  config: Pick<RuntimeConfig, "llmProvider" | "llmReasoningEffort">;
}[];

const TTS_ENGINES = ["sherpa", "cartesia"] satisfies readonly NonNullable<
  RuntimeConfig["ttsEngine"]
>[];

export function expandMatrix(): SweepMatrixRow[] {
  const cascade = CASCADE_MODELS.flatMap((model) =>
    TTS_ENGINES.map((ttsEngine): SweepMatrixRow => ({
      id: `${model.id}-${ttsEngine}`,
      config: {
        responder: "cascade",
        ...model.config,
        ttsEngine,
      },
    })),
  );
  return [{ id: "realtime", config: { responder: "realtime" } }, ...cascade];
}
