import type { ChatConfig } from "./contract.ts";
import {
  OpenAICompatibleChatModel,
  type FetchLike,
} from "./openai-compatible.ts";

export const INCEPTION_BASE_URL = "https://api.inceptionlabs.ai/v1";
export const DEFAULT_MERCURY_MODEL = "mercury-2";
export const MERCURY_REASONING_EFFORTS = ["instant", "low"] as const;
export type MercuryReasoningEffort = (typeof MERCURY_REASONING_EFFORTS)[number];

export type MercuryOptions = {
  config?: Partial<ChatConfig>;
  reasoningEffort?: MercuryReasoningEffort;
  fetch?: FetchLike;
  baseUrl?: string;
  apiKey?: () => string | undefined;
};

/** Mercury 2 streaming chat, tuned for low latency voice responses. */
export class MercuryChatModel extends OpenAICompatibleChatModel {
  constructor(options: MercuryOptions = {}) {
    super({
      provider: "Mercury",
      baseUrl: options.baseUrl ?? INCEPTION_BASE_URL,
      defaultModel: DEFAULT_MERCURY_MODEL,
      config: options.config,
      fetch: options.fetch,
      apiKey: options.apiKey ?? (() => process.env.INCEPTIONLABS_API_KEY),
      apiKeyName: "INCEPTIONLABS_API_KEY",
      extraBody: {
        reasoning_effort: options.reasoningEffort ?? "instant",
      },
    });
  }
}
