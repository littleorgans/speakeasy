import { createHash } from "node:crypto";
import type { CatalogueEntry, OutputApiKind, SpeechOutputConfig, SpeechOutputRegistry } from "@speakeasy/speech-io/providers";
import type { getLocalOutputCatalogue } from "../../speech-io/src/tts/catalogue.ts";
import { MAX_SAMPLE_TEXT_LENGTH } from "./speech-output-protocol.ts";

type LocalCatalogue = Awaited<ReturnType<typeof getLocalOutputCatalogue>>;
export type SpeechOutputModel = {
  id: string;
  label: string;
  provider: SpeechOutputConfig["provider"];
  voices: { id: string | number; label: string; languages: readonly string[] }[];
  defaultVoice: string | number;
  defaultSpeed: number;
  speed: { min: number; max: number };
  sampleRate: number;
  streaming: "sentence" | "pcm";
  licence: string;
  readiness: { state: "ready" | "assets-missing" | "runtime-missing" | "adapter-deferred" | "failed"; reason: string | null };
  configured: boolean;
};

export type ApiOutputModel = Omit<SpeechOutputModel, "provider" | "configured"> & {
  provider: { kind: OutputApiKind };
};

export type SpeechOutputCatalogue = {
  schema: 1;
  revision: string;
  default: SpeechOutputConfig;
  maxTextLength: number;
  models: SpeechOutputModel[];
  output: CatalogueEntry<SpeechOutputConfig>[];
  registry: SpeechOutputRegistry;
};

/**
 * Project the real approved TTS catalogue, supplied by host composition after its
 * metadata readiness check. No factory, environment or provider is opened here.
 */
export function buildSpeechOutputCatalogue(options: {
  local: LocalCatalogue;
  localRegistry: SpeechOutputRegistry;
  apiModels?: readonly ApiOutputModel[];
  configured?: (kind: OutputApiKind) => boolean;
}): SpeechOutputCatalogue {
  const models: SpeechOutputModel[] = options.local.models.map((model) => ({
    id: model.id,
    label: model.label,
    provider: { kind: "local" },
    voices: model.voices.map(({ id, label, languages }) => ({ id, label, languages: [...languages] })),
    defaultVoice: model.defaultVoice,
    defaultSpeed: options.local.default.speed,
    speed: { ...options.localRegistry.find((entry) => entry.kind === "local" && entry.model === model.id)!.speed },
    sampleRate: model.sampleRate,
    streaming: model.streaming,
    licence: model.licence,
    readiness: { ...model.readiness },
    configured: true,
  }));
  for (const model of options.apiModels ?? []) {
    if (!options.configured?.(model.provider.kind)) continue;
    models.push({
      id: model.id, label: model.label, provider: { kind: model.provider.kind },
      voices: model.voices.map(({ id, label, languages }) => ({ id, label, languages: [...languages] })),
      defaultVoice: model.defaultVoice, defaultSpeed: model.defaultSpeed, speed: { ...model.speed },
      sampleRate: model.sampleRate, streaming: model.streaming, licence: model.licence,
      readiness: { ...model.readiness }, configured: true,
    });
  }
  const output = models.map((model): CatalogueEntry<SpeechOutputConfig> => ({
    id: `${model.provider.kind}/${model.id}`,
    label: model.label,
    config: { schema: 1, provider: { ...model.provider }, model: model.id,
      voice: model.defaultVoice, speed: model.defaultSpeed },
    selectable: model.readiness.state === "ready",
    ...(model.readiness.state === "ready" ? {} : { reason: unavailableReason(model.readiness.state) }),
    transfer: model.provider.kind === "local" ? "stays-on-mac" : "text-to-api-audio-returned",
    samples: [],
  }));
  const registry: SpeechOutputRegistry = models.map((model) => {
    const local = model.provider.kind === "local"
      ? options.localRegistry.find((entry) => entry.kind === "local" && entry.model === model.id) : undefined;
    return { kind: model.provider.kind, model: model.id,
      voices: [...(local?.voices ?? model.voices.map((voice) => voice.id))], speed: { ...model.speed } };
  });
  const defaults = { ...options.local.default, provider: { ...options.local.default.provider } };
  const revision = createHash("sha256").update(JSON.stringify({ defaults, models, registry })).digest("hex");
  return { schema: 1, revision, default: defaults, maxTextLength: MAX_SAMPLE_TEXT_LENGTH, models, output, registry };
}

function unavailableReason(state: SpeechOutputModel["readiness"]["state"]): NonNullable<CatalogueEntry<SpeechOutputConfig>["reason"]> {
  if (state === "assets-missing") return "assets-missing";
  if (state === "adapter-deferred") return "adapter-deferred";
  return "runtime-missing";
}
