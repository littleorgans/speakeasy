import type { ChatConfig } from "./contract.ts";
import {
  OpenAICompatibleChatModel,
  interpretChatCompletionData,
  type FetchLike,
} from "./openai-compatible.ts";
export type { FetchLike } from "./openai-compatible.ts";

export const CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1";
export const DEFAULT_CEREBRAS_MODEL = "gemma-4-31b";

export type CerebrasOptions = {
  config?: Partial<ChatConfig>;
  fetch?: FetchLike;
  baseUrl?: string;
  apiKey?: () => string | undefined;
};

/** Cerebras provider configuration over the shared chat completion adapter. */
export class CerebrasChatModel extends OpenAICompatibleChatModel {
  constructor(options: CerebrasOptions = {}) {
    super({
      provider: "Cerebras",
      baseUrl: options.baseUrl ?? CEREBRAS_BASE_URL,
      defaultModel: DEFAULT_CEREBRAS_MODEL,
      config: options.config,
      fetch: options.fetch,
      apiKey: options.apiKey ?? (() => process.env.CEREBRAS_API_KEY),
      apiKeyName: "CEREBRAS_API_KEY",
    });
  }
}

/** Compatibility alias for existing callers and tests. */
export const interpretCerebrasData = interpretChatCompletionData;
