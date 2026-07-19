import type { ChatMessage } from "@speakeasy/llm";

/**
 * Running conversation history fed to the ChatModel each turn. The system prompt
 * is pinned at the head; user/assistant turns are capped to a constant window so
 * the prompt stays bounded over a long conversation.
 */

export const DEFAULT_SYSTEM_PROMPT =
  "You are a concise, friendly voice assistant. Answer in one or two short spoken sentences, no lists or markdown.";

/** Max user/assistant messages retained (excludes the system prompt). */
export const DEFAULT_HISTORY_LIMIT = 12;

export class ChatHistory {
  readonly #system: string;
  readonly #limit: number;
  #turns: ChatMessage[] = [];

  constructor(
    system: string = DEFAULT_SYSTEM_PROMPT,
    limit: number = DEFAULT_HISTORY_LIMIT,
  ) {
    this.#system = system;
    this.#limit = Math.max(1, limit);
  }

  addUser(content: string): void {
    this.#push({ role: "user", content });
  }

  addAssistant(content: string): void {
    this.#push({ role: "assistant", content });
  }

  /** The full prompt: system message plus the capped turn window. */
  messages(): ChatMessage[] {
    return [{ role: "system", content: this.#system }, ...this.#turns];
  }

  get size(): number {
    return this.#turns.length;
  }

  #push(message: ChatMessage): void {
    this.#turns.push(message);
    if (this.#turns.length > this.#limit) {
      this.#turns = this.#turns.slice(-this.#limit);
    }
  }
}
