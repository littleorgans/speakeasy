/**
 * @speakeasy/llm public surface: the provider-agnostic chat contract plus the
 * Cerebras adapter. convo-engine builds on the ChatModel type; the demo wires
 * the concrete Cerebras model.
 */
export type { ChatModel, ChatMessage, ChatRole, ChatConfig } from "./contract.ts";
export {
  CerebrasChatModel,
  CEREBRAS_BASE_URL,
  DEFAULT_CEREBRAS_MODEL,
  type CerebrasOptions,
} from "./cerebras.ts";
