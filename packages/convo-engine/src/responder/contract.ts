import type { AudioSegment } from "@speakeasy/speech-io";
import type { ChatMessage } from "@speakeasy/llm";

/**
 * VoiceResponder: the "respond and speak" boundary of the conversation loop.
 *
 * A responder takes the conversation so far and produces the assistant's spoken
 * reply as an interleaved stream of text deltas and audio segments. Two shapes
 * live behind it:
 *
 * - CascadeResponder: the classic pipeline. A ChatModel streams tokens into a
 *   TTSSession sentence pipeline (two engines, freely mixed).
 * - Fused speech models (OpenAI Realtime): one model receives the transcript
 *   and emits audio + reply text directly, collapsing LLM TTFT and TTS TTFA
 *   into a single hop.
 *
 * The loop never knows which shape it is driving, so engines stay swappable
 * behind one seam (and behind one host protocol for embedders).
 */

/** One event of a spoken reply. Token events carry their arrival timestamp so
 * latency metrics survive any internal buffering between text and audio. */
export type ResponderEvent =
  | { type: "token"; text: string; at: number }
  | { type: "audio"; segment: AudioSegment };

/** A per-conversation session; `respond` is called once per user turn. */
export interface ResponderSession {
  /**
   * Produce the spoken reply to the conversation. `messages` is the full
   * window (system prompt first); implementations that keep server-side state
   * may use only the newest user message.
   */
  respond(messages: ChatMessage[]): AsyncIterable<ResponderEvent>;
  close(): Promise<void>;
}

export interface VoiceResponder {
  open(): Promise<ResponderSession>;
}
