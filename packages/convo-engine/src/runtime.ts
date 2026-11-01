import {
  CartesiaTextToSpeech,
  createLocalSherpaEngine,
  DEFAULT_CARTESIA_MODEL,
  DEFAULT_CARTESIA_VOICE,
  DEFAULT_RULES,
  parseTtsModelId,
  SherpaTextToSpeech,
  SpeechOutputFactory,
  withRewrite,
  type TextToSpeech,
  type TTSConfig,
  type SpeechOutputConfig,
  type SpeechOutputContext,
  type VoiceToText,
} from "@speakeasy/speech-io";
import {
  CerebrasChatModel,
  DEFAULT_CEREBRAS_MODEL,
  DEFAULT_MERCURY_MODEL,
  MercuryChatModel,
  type ChatModel,
  type MercuryReasoningEffort,
} from "@speakeasy/llm";
import { CascadeResponder } from "./responder/cascade.ts";
import {
  DEFAULT_REALTIME_MODEL,
  DEFAULT_REALTIME_VOICE,
  OpenAIRealtimeResponder,
  parseRealtimeVoice,
} from "./responder/openai-realtime.ts";
import type { VoiceResponder } from "./responder/contract.ts";

export type ResponderKind = "cascade" | "realtime";
export type LlmProvider = "cerebras" | "mercury";
export type TtsEngine = "sherpa" | "cartesia";

export type RuntimeConfig = {
  responder?: ResponderKind;
  llmProvider?: LlmProvider;
  llmModel?: string;
  llmReasoningEffort?: MercuryReasoningEffort;
  ttsEngine?: TtsEngine;
  ttsModel?: string;
  voice?: string;
  /** Explicit selection for Mercury cascade. Omitted callers retain legacy TTS defaults. */
  speechOutput?: SpeechOutputConfig;
};

export type RuntimeOptions = {
  /** Reuse the host's output factory so room and voice trials share resident graphs. */
  speechOutputFactory?: Pick<SpeechOutputFactory, "create" | "quiesce">;
  speechOutputContext?: SpeechOutputContext;
};

export type ConversationRuntime = {
  stt: VoiceToText;
  responder: VoiceResponder;
  label: string;
  /** Await after responder close before releasing native work admission. */
  quiesce?: () => Promise<void>;
};

/**
 * Shared composition root for every host. Concrete engine selection stays here
 * so browser, terminal, and future desktop shells never grow parallel wiring.
 */
export async function createConversationRuntime(
  config: RuntimeConfig = {},
  env: NodeJS.ProcessEnv = process.env,
  stt?: VoiceToText,
  options: RuntimeOptions = {},
): Promise<ConversationRuntime> {
  const responder = buildResponder(config, env, options);
  if (stt) {
    return {
      stt,
      responder: responder.value,
      quiesce: responder.quiesce,
      label: `shared STT · ${responder.label}`,
    };
  }
  const engine = await createLocalSherpaEngine();
  await engine.prepare();
  return {
    stt: withRewrite(engine, { rules: DEFAULT_RULES, numbers: "off" }),
    responder: responder.value,
    quiesce: responder.quiesce,
    label: `${engine.label} · ${responder.label}`,
  };
}

function buildResponder(
  config: RuntimeConfig,
  env: NodeJS.ProcessEnv,
  options: RuntimeOptions,
): { value: VoiceResponder; label: string; quiesce?: () => Promise<void> } {
  if ((config.responder ?? "cascade") === "realtime") {
    requireKey(env.OPENAI_API_KEY, "OPENAI_API_KEY", "realtime responder");
    const voice = parseRealtimeVoice(config.voice ?? DEFAULT_REALTIME_VOICE);
    const model = config.llmModel ?? DEFAULT_REALTIME_MODEL;
    return {
      value: new OpenAIRealtimeResponder({
        model,
        voice,
        apiKey: () => env.OPENAI_API_KEY,
      }),
      label: `OpenAI Realtime ${model} · ${voice}`,
    };
  }

  const voice = buildTts(config, env, options);
  const llm = buildChatModel(config, env);
  return {
    value: new CascadeResponder({
      llm: llm.value,
      tts: voice.value,
      ttsConfig: voice.config,
    }),
    label: `${llm.label} · ${voice.label}`,
    quiesce: voice.quiesce,
  };
}

function buildChatModel(
  config: RuntimeConfig,
  env: NodeJS.ProcessEnv,
): { value: ChatModel; label: string } {
  if ((config.llmProvider ?? "cerebras") === "mercury") {
    requireKey(
      env.INCEPTIONLABS_API_KEY,
      "INCEPTIONLABS_API_KEY",
      "Mercury responder",
    );
    const model = config.llmModel ?? DEFAULT_MERCURY_MODEL;
    const reasoning = config.llmReasoningEffort ?? "instant";
    return {
      value: new MercuryChatModel({
        config: { model },
        reasoningEffort: reasoning,
        apiKey: () => env.INCEPTIONLABS_API_KEY,
      }),
      label: `Mercury ${model} · ${reasoning}`,
    };
  }

  requireKey(env.CEREBRAS_API_KEY, "CEREBRAS_API_KEY", "Cerebras responder");
  const model = config.llmModel ?? DEFAULT_CEREBRAS_MODEL;
  return {
    value: new CerebrasChatModel({
      config: { model },
      apiKey: () => env.CEREBRAS_API_KEY,
    }),
    label: `Cerebras ${model}`,
  };
}

function buildTts(
  config: RuntimeConfig,
  env: NodeJS.ProcessEnv,
  options: RuntimeOptions,
): { value: TextToSpeech; config?: TTSConfig; label: string; quiesce?: () => Promise<void> } {
  if (config.llmProvider === "mercury" && config.speechOutput) {
    const selected = { ...config.speechOutput, provider: { ...config.speechOutput.provider } };
    const factory = options.speechOutputFactory ?? new SpeechOutputFactory();
    return {
      value: factory.create(selected, options.speechOutputContext),
      label: `${selected.provider.kind} ${selected.model} · ${selected.voice}`,
      quiesce: () => factory.quiesce(selected),
    };
  }
  if ((config.ttsEngine ?? "sherpa") === "cartesia") {
    requireKey(env.CARTESIA_API_KEY, "CARTESIA_API_KEY", "Cartesia voice");
    const voice = config.voice ?? DEFAULT_CARTESIA_VOICE;
    return {
      value: new CartesiaTextToSpeech({ apiKey: () => env.CARTESIA_API_KEY }),
      config: { model: config.ttsModel, voice },
      label: `Cartesia ${config.ttsModel ?? DEFAULT_CARTESIA_MODEL} · ${voice}`,
    };
  }

  const model = parseTtsModelId(config.ttsModel ?? "kokoro-v0.19");
  const speaker = config.voice === undefined ? undefined : Number(config.voice);
  if (speaker !== undefined && !Number.isFinite(speaker)) {
    throw new Error(`Sherpa voice must be a numeric speaker id, got "${config.voice}"`);
  }
  return {
    value: new SherpaTextToSpeech(),
    config: { model, voice: speaker },
    label: `${model}${speaker === undefined ? "" : ` · voice ${speaker}`}`,
  };
}

function requireKey(
  value: string | undefined,
  name: string,
  purpose: string,
): asserts value is string {
  if (!value) {
    throw new Error(`${name} is required for the ${purpose}`);
  }
}
