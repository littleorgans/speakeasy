/**
 * @speakeasy/llm public surface: the provider-agnostic chat contract plus the
 * OpenAI compatible provider adapters. convo-engine builds on the ChatModel
 * type and selects a concrete provider at its composition root.
 */
export type { ChatModel, ChatMessage, ChatRole, ChatConfig } from "./contract.ts";
export {
  CerebrasChatModel,
  CEREBRAS_BASE_URL,
  DEFAULT_CEREBRAS_MODEL,
  type CerebrasOptions,
} from "./cerebras.ts";
export {
  MercuryChatModel,
  INCEPTION_BASE_URL,
  DEFAULT_MERCURY_MODEL,
  MERCURY_REASONING_EFFORTS,
  type MercuryOptions,
  type MercuryReasoningEffort,
} from "./mercury.ts";
export {
  OpenAICompatibleChatModel,
  interpretChatCompletionData,
  type OpenAICompatibleOptions,
  type ChatCompletionEvent,
  type FetchLike,
} from "./openai-compatible.ts";
