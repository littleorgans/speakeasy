/**
 * @speakeasy/llm contract: the provider-agnostic chat boundary.
 *
 * The middle stage of the cascade (STT -> LLM -> TTS). A ChatModel takes a
 * conversation and streams the assistant reply as text deltas. No vendor types
 * leak through it: behind the contract an adapter may talk to Cerebras, a local
 * runtime, or anything else. The convo-engine feeds these deltas straight into
 * TextToSpeech.speak, so the unit of streaming is a text fragment, not a token
 * object.
 */

export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ChatConfig = {
  /** Provider model id. */
  model: string;
  /** Sampling temperature; provider default when omitted. */
  temperature?: number;
  /** Cap on generated tokens; provider default when omitted. */
  maxTokens?: number;
};

export interface ChatModel {
  /** Stream the assistant reply as text deltas in arrival order. */
  stream(messages: ChatMessage[]): AsyncIterable<string>;
}
