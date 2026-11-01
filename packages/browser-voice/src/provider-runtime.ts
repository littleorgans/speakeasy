import {
  CARTESIA_BROWSER_MODEL,
  CARTESIA_SAMPLE_RATE,
  CARTESIA_SPEED,
  DEFAULT_CARTESIA_VOICE,
  DEFAULT_OPENAI_OUTPUT_CONFIG,
  OPENAI_SAMPLE_RATE,
  OPENAI_SPEED,
  SpeechOutputFactory,
  type LocalSpeechOutputFactory,
  type OpenAISocketFactory,
} from "@speakeasy/speech-io";
import { LOCAL_OUTPUT_REGISTRY, type getLocalOutputCatalogue } from "../../speech-io/src/tts/catalogue.ts";
import type { BrowserVoiceServerOptions } from "./host.ts";
import type { ProviderCredentials } from "./provider-availability.ts";
import { buildSpeechOutputCatalogue, type ApiOutputModel } from "./speech-output-catalog.ts";
import { createFileInputAdapters, createHostProviderConfig,
  type ProviderRoutesConfig, type SetupReadiness } from "./setup-routes.ts";

/** Application choices only; reading these descriptors never opens a provider. */
export const HOSTED_OUTPUT_MODELS: readonly ApiOutputModel[] = [
  {
    id: CARTESIA_BROWSER_MODEL,
    label: "Cartesia Sonic · Katie",
    provider: { kind: "cartesia" },
    voices: [{ id: DEFAULT_CARTESIA_VOICE, label: "Katie", languages: ["en"] }],
    defaultVoice: DEFAULT_CARTESIA_VOICE,
    defaultSpeed: 1,
    speed: CARTESIA_SPEED,
    sampleRate: CARTESIA_SAMPLE_RATE,
    streaming: "sentence",
    licence: "Provider terms unverified",
    readiness: { state: "ready", reason: null },
  },
  {
    id: DEFAULT_OPENAI_OUTPUT_CONFIG.model,
    label: "OpenAI · Marin",
    provider: { kind: "openai" },
    voices: [{ id: DEFAULT_OPENAI_OUTPUT_CONFIG.voice, label: "Marin", languages: ["en"] }],
    defaultVoice: DEFAULT_OPENAI_OUTPUT_CONFIG.voice,
    defaultSpeed: DEFAULT_OPENAI_OUTPUT_CONFIG.speed,
    speed: OPENAI_SPEED,
    sampleRate: OPENAI_SAMPLE_RATE,
    streaming: "sentence",
    licence: "Provider terms unverified",
    readiness: { state: "ready", reason: null },
  },
];

export type ProviderRuntimeOptions = {
  credentials: ProviderCredentials;
  readiness: () => readonly SetupReadiness[];
  localOutput: Pick<LocalSpeechOutputFactory, "create" | "quiesce" | "close">;
  localCatalogue: Awaited<ReturnType<typeof getLocalOutputCatalogue>>;
  createSocket: OpenAISocketFactory;
  fetch?: typeof globalThis.fetch;
};

/** One inert assembly shared by input, independent trials and room output. */
export function createProviderRuntime(options: ProviderRuntimeOptions): {
  output: SpeechOutputFactory;
  providers: ProviderRoutesConfig;
  speechOutput: NonNullable<BrowserVoiceServerOptions["speechOutput"]>;
} {
  const output = new SpeechOutputFactory({ local: options.localOutput,
    fetch: options.fetch, createSocket: options.createSocket });
  const local = structuredClone(options.localCatalogue);
  const catalogue = () => buildSpeechOutputCatalogue({ local, localRegistry: LOCAL_OUTPUT_REGISTRY,
    apiModels: HOSTED_OUTPUT_MODELS, configured: options.credentials.configured });
  const providers = createHostProviderConfig({ credentials: options.credentials,
    readiness: options.readiness, inputAdapters: createFileInputAdapters({ fetch: options.fetch }),
    outputCatalogue: catalogue });
  return { output, providers, speechOutput: {
    catalogue,
    createOutput: (config, context) => {
      context.recheck();
      return output.create(config, { beforeRequest: context.recheck, getKey: context.apiKey,
        signal: context.signal, observe: context.observe });
    },
    quiesce: (config) => output.quiesce(config),
    close: () => output.close(),
  } };
}
