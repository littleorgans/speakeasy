import type { TTSConfig } from "./tts/contract.ts";

export type InputApiKind = "cartesia" | "groq" | "mistral" | "openai" | "gemini";
export type OutputApiKind = "cartesia" | "openai" | "elevenlabs";
export type ApiProviderKind = InputApiKind | OutputApiKind;
export type SpeechDirection = "input" | "output";

export type SpeechInputConfig = {
  schema: 1;
  provider:
    | { kind: "local"; presetId: string }
    | { kind: InputApiKind; model: string };
};

export type SpeechOutputConfig = {
  schema: 1;
  provider: { kind: "local" | OutputApiKind };
} & Required<Pick<TTSConfig, "model" | "voice" | "speed">>;

export type SpeechPreferences = {
  schema: 1;
  input: SpeechInputConfig;
  output: SpeechOutputConfig;
};

/** Supply the actual preset manifest and adapter models, without loading them. */
export type SpeechInputRegistry = {
  localPresets: readonly { id: string }[];
  apiModels: readonly { kind: InputApiKind; model: string }[];
};

/** Project the existing output registry; numeric and named voices stay distinct. */
export type SpeechOutputRegistry = readonly {
  kind: SpeechOutputConfig["provider"]["kind"];
  model: string;
  voices: readonly SpeechOutputConfig["voice"][];
  speed: { min: number; max: number };
}[];

export type SpeechConfigRegistries = {
  input: SpeechInputRegistry;
  output: SpeechOutputRegistry;
};

export type CatalogueSample = {
  id: string;
  storage: "local-asset";
  production: "local-synthesis" | "api-synthesis" | "vendor-example" | "unknown";
  text: string | null;
  config: SpeechOutputConfig | null;
  evidenceRef: string | null;
};

export type CatalogueEntry<C> = {
  id: string;
  config: C;
  label: string;
  selectable: boolean;
  reason?: "assets-missing" | "runtime-missing" | "adapter-deferred";
  transfer: "stays-on-mac" | "audio-to-api" | "text-to-api-audio-returned";
  samples: CatalogueSample[];
};

export type SpeechCatalogue = {
  schema: 1;
  revision: string;
  input: CatalogueEntry<SpeechInputConfig>[];
  output: CatalogueEntry<SpeechOutputConfig>[];
};

export type TimingObservation =
  | { status: "measured"; ms: number; clock: "server" | "worker" | "browser";
      boundary: string }
  | { status: "provider-reported"; ms: number; metric: string;
      boundary: string; evidenceRef: string }
  | { status: "unavailable"; reason: string };

/** Durations have independent boundaries. No residual network time is inferred. */
export type SpeechTiming = {
  clientWallMs: TimingObservation;
  rowWallMs: TimingObservation;
  queueMs: TimingObservation;
  loadMs: TimingObservation;
  firstUseMs: TimingObservation;
  localProcessingMs: TimingObservation;
  providerRequestWallMs: TimingObservation;
  providerProcessingMs: TimingObservation;
  providerNetworkMs: TimingObservation;
  serverFirstAudioReadyMs: TimingObservation;
  browserFirstAudioMs: TimingObservation;
};

const ERROR_MESSAGES = {
  CONFIG_UNSUPPORTED: "Speech configuration is unsupported. Refresh the catalogue.",
  TIMING_INVALID: "Speech timing observation is invalid.",
  PROVIDER_UNAVAILABLE: "Speech provider is unavailable. Refresh the catalogue.",
  SELECTION_REQUIRED: "Explicit provider selection is required for this session.",
  SELECTION_STALE: "Provider selection has changed. Select again before running.",
  SAVED_REPLAY_NOT_AUTHORIZED: "Saved material is not authorized for this API.",
  SOURCE_UNVERIFIED: "This source is not eligible for API dispatch.",
  PROVIDER_FAILED: "Speech provider request failed.",
} as const;

export type ProviderErrorCode = keyof typeof ERROR_MESSAGES;

/** Never include caller values, credentials, transport bodies or raw causes. */
export class ProviderError extends Error {
  readonly code: ProviderErrorCode;

  constructor(code: ProviderErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = "ProviderError";
    this.code = code;
  }

  toJSON(): { code: ProviderErrorCode; message: string } {
    return { code: this.code, message: ERROR_MESSAGES[this.code] };
  }
}

export function safeProviderError(error: unknown): ProviderError {
  return new ProviderError(error instanceof ProviderError ? error.code : "PROVIDER_FAILED");
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function keys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === expected.length && expected.every((key) => Object.hasOwn(value, key));
}

function text(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function invalidConfig(): never {
  throw new ProviderError("CONFIG_UNSUPPORTED");
}

/** Parse JSON data against supplied identities. No preset or model list lives here. */
export function parseSpeechInputConfig(
  value: unknown,
  registry: SpeechInputRegistry,
): SpeechInputConfig {
  if (!record(value) || !keys(value, ["schema", "provider"]) || value.schema !== 1 ||
      !record(value.provider)) return invalidConfig();
  const provider = value.provider;
  if (provider.kind === "local") {
    if (!keys(provider, ["kind", "presetId"]) || !text(provider.presetId) ||
        !registry.localPresets.some((preset) => preset.id === provider.presetId)) {
      return invalidConfig();
    }
    return { schema: 1, provider: { kind: "local", presetId: provider.presetId } };
  }
  if (!keys(provider, ["kind", "model"]) || !text(provider.model) ||
      !isInputApiKind(provider.kind) ||
      !registry.apiModels.some((entry) => entry.kind === provider.kind && entry.model === provider.model)) {
    return invalidConfig();
  }
  return { schema: 1, provider: { kind: provider.kind, model: provider.model } };
}

export function parseSpeechOutputConfig(
  value: unknown,
  registry: SpeechOutputRegistry,
): SpeechOutputConfig {
  if (!record(value) || !keys(value, ["schema", "provider", "model", "voice", "speed"]) ||
      value.schema !== 1 || !record(value.provider) || !keys(value.provider, ["kind"]) ||
      !isOutputKind(value.provider.kind) || !text(value.model) ||
      !(text(value.voice) || (typeof value.voice === "number" && Number.isInteger(value.voice) && value.voice >= 0)) ||
      typeof value.speed !== "number" || !Number.isFinite(value.speed) || value.speed <= 0) {
    return invalidConfig();
  }
  const { kind } = value.provider;
  const { model, voice, speed } = value;
  if (!registry.some((entry) => entry.kind === kind && entry.model === model &&
      entry.voices.includes(voice) && speed >= entry.speed.min && speed <= entry.speed.max)) {
    return invalidConfig();
  }
  return { schema: 1, provider: { kind }, model, voice, speed };
}

export function parseSpeechPreferences(
  value: unknown,
  registries: SpeechConfigRegistries,
): SpeechPreferences {
  if (!record(value) || !keys(value, ["schema", "input", "output"]) || value.schema !== 1) {
    return invalidConfig();
  }
  return { schema: 1, input: parseSpeechInputConfig(value.input, registries.input),
    output: parseSpeechOutputConfig(value.output, registries.output) };
}

export function isInputApiKind(value: unknown): value is InputApiKind {
  return value === "cartesia" || value === "groq" || value === "mistral" ||
    value === "openai" || value === "gemini";
}

function isOutputKind(value: unknown): value is SpeechOutputConfig["provider"]["kind"] {
  return value === "local" || value === "cartesia" || value === "openai" || value === "elevenlabs";
}

export function sameSpeechInputConfig(a: SpeechInputConfig, b: SpeechInputConfig): boolean {
  return a.provider.kind === "local" && b.provider.kind === "local"
    ? a.provider.presetId === b.provider.presetId
    : a.provider.kind !== "local" && b.provider.kind !== "local" &&
      a.provider.kind === b.provider.kind && a.provider.model === b.provider.model;
}

export function sameSpeechOutputConfig(a: SpeechOutputConfig, b: SpeechOutputConfig): boolean {
  return a.provider.kind === b.provider.kind && a.model === b.model &&
    a.voice === b.voice && a.speed === b.speed;
}

export function parseTimingObservation(value: unknown): TimingObservation {
  const fail = (): never => { throw new ProviderError("TIMING_INVALID"); };
  if (!record(value)) return fail();
  if (value.status === "unavailable") {
    if (!keys(value, ["status", "reason"]) || !text(value.reason)) return fail();
    return { status: "unavailable", reason: value.reason };
  }
  if (typeof value.ms !== "number" || !Number.isFinite(value.ms) || value.ms < 0 ||
      !text(value.boundary)) return fail();
  if (value.status === "measured" && keys(value, ["status", "ms", "clock", "boundary"]) &&
      (value.clock === "server" || value.clock === "worker" || value.clock === "browser")) {
    return { status: "measured", ms: value.ms, clock: value.clock, boundary: value.boundary };
  }
  if (value.status === "provider-reported" &&
      keys(value, ["status", "ms", "metric", "boundary", "evidenceRef"]) &&
      text(value.metric) && text(value.evidenceRef)) {
    return { status: "provider-reported", ms: value.ms, metric: value.metric,
      boundary: value.boundary, evidenceRef: value.evidenceRef };
  }
  return fail();
}
