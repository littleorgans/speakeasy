import type { ChatConfig, ChatMessage, ChatModel } from "./contract.ts";
import { parseSSEStream } from "./sse.ts";

/**
 * Cerebras adapter over the OpenAI-compatible streaming chat API.
 *
 * No new npm dependency: it uses the global fetch and hand-rolled SSE parsing.
 * The API key is read from process.env.CEREBRAS_API_KEY at call time only,
 * never stored on the instance, never logged, and never echoed in an error
 * (error bodies come from the server and carry no key). fetch and the key
 * getter are injectable so the adapter is unit-testable without a network.
 */

export const CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1";

/**
 * Verified against GET /v1/models on 2026-07-07 (served ids: zai-glm-4.7,
 * gemma-4-31b, gpt-oss-120b). gemma-4-31b is the served Gemma variant and the
 * default general chat model for the cascade.
 */
export const DEFAULT_CEREBRAS_MODEL = "gemma-4-31b";

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

export type CerebrasOptions = {
  /** Overrides for the default config (model, temperature, maxTokens). */
  config?: Partial<ChatConfig>;
  /** Injectable fetch; defaults to the global. */
  fetch?: FetchLike;
  baseUrl?: string;
  /** Key source, read on each stream() call; defaults to the environment. */
  apiKey?: () => string | undefined;
};

/** One decoded event from the Cerebras stream. */
export type CerebrasEvent =
  | { type: "delta"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

export class CerebrasChatModel implements ChatModel {
  readonly #fetch: FetchLike;
  readonly #baseUrl: string;
  readonly #apiKey: () => string | undefined;
  readonly #config: ChatConfig;

  constructor(options: CerebrasOptions = {}) {
    this.#fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.#baseUrl = options.baseUrl ?? CEREBRAS_BASE_URL;
    this.#apiKey = options.apiKey ?? (() => process.env.CEREBRAS_API_KEY);
    this.#config = { model: DEFAULT_CEREBRAS_MODEL, ...options.config };
  }

  async *stream(messages: ChatMessage[]): AsyncGenerator<string> {
    const key = this.#apiKey();
    if (!key) {
      throw new Error("CEREBRAS_API_KEY is not set");
    }
    const response = await this.#fetch(`${this.#baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: this.#config.model,
        stream: true,
        messages,
        ...(this.#config.temperature !== undefined
          ? { temperature: this.#config.temperature }
          : {}),
        ...(this.#config.maxTokens !== undefined
          ? { max_tokens: this.#config.maxTokens }
          : {}),
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(
        `Cerebras request failed: HTTP ${response.status}${await errorDetail(response)}`,
      );
    }

    for await (const payload of parseSSEStream(decodeBody(response.body))) {
      const event = interpretCerebrasData(payload);
      if (!event) {
        continue;
      }
      if (event.type === "done") {
        return;
      }
      if (event.type === "error") {
        throw new Error(`Cerebras stream error: ${event.message}`);
      }
      yield event.content;
    }
  }
}

/**
 * Interpret one SSE `data:` payload: the sentinel [DONE], a server error frame,
 * a content delta, or nothing (role-only / empty frames are skipped).
 */
export function interpretCerebrasData(payload: string): CerebrasEvent | undefined {
  if (payload === "[DONE]") {
    return { type: "done" };
  }
  let json: unknown;
  try {
    json = JSON.parse(payload);
  } catch {
    return undefined; // SSE comments or keep-alives are not JSON.
  }
  if (!isRecord(json)) {
    return undefined;
  }
  if (isRecord(json.error)) {
    return { type: "error", message: String(json.error.message ?? "unknown error") };
  }
  const content = deltaContent(json);
  return content ? { type: "delta", content } : undefined;
}

function deltaContent(json: Record<string, unknown>): string | undefined {
  const choices = json.choices;
  const first = Array.isArray(choices) ? choices[0] : undefined;
  const delta = isRecord(first) ? first.delta : undefined;
  const content = isRecord(delta) ? delta.content : undefined;
  return typeof content === "string" && content.length > 0 ? content : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Decode a response body stream into text chunks for the SSE parser. */
async function* decodeBody(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        yield decoder.decode(value, { stream: true });
      }
    }
    const tail = decoder.decode();
    if (tail) {
      yield tail;
    }
  } finally {
    reader.releaseLock();
  }
}

/** Best-effort server error text for the thrown message; never includes the key. */
async function errorDetail(response: Response): Promise<string> {
  try {
    const text = (await response.text()).trim();
    return text ? ` - ${text.slice(0, 300)}` : "";
  } catch {
    return "";
  }
}
