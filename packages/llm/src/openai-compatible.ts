import type { ChatConfig, ChatMessage, ChatModel } from "./contract.ts";
import { parseSSEStream } from "./sse.ts";

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

export type OpenAICompatibleOptions = {
  provider: string;
  baseUrl: string;
  defaultModel: string;
  config?: Partial<ChatConfig>;
  fetch?: FetchLike;
  apiKey: () => string | undefined;
  apiKeyName: string;
  extraBody?: Readonly<Record<string, unknown>>;
};

export type ChatCompletionEvent =
  | { type: "delta"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

/** Shared streaming adapter for OpenAI compatible chat completion providers. */
export class OpenAICompatibleChatModel implements ChatModel {
  readonly #options: OpenAICompatibleOptions;
  readonly #fetch: FetchLike;
  readonly #config: ChatConfig;

  constructor(options: OpenAICompatibleOptions) {
    this.#options = options;
    this.#fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.#config = { model: options.defaultModel, ...options.config };
  }

  async *stream(messages: ChatMessage[]): AsyncGenerator<string> {
    const key = this.#options.apiKey();
    if (!key) throw new Error(`${this.#options.apiKeyName} is not set`);

    const response = await this.#fetch(`${this.#options.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: this.#config.model,
        stream: true,
        messages,
        ...(this.#config.temperature === undefined
          ? {}
          : { temperature: this.#config.temperature }),
        ...(this.#config.maxTokens === undefined
          ? {}
          : { max_tokens: this.#config.maxTokens }),
        ...this.#options.extraBody,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(
        `${this.#options.provider} request failed: HTTP ${response.status}${await errorDetail(response)}`,
      );
    }

    for await (const payload of parseSSEStream(decodeBody(response.body))) {
      const event = interpretChatCompletionData(payload);
      if (!event) continue;
      if (event.type === "done") return;
      if (event.type === "error") {
        throw new Error(`${this.#options.provider} stream error: ${event.message}`);
      }
      yield event.content;
    }
  }
}

/** Decode one OpenAI compatible SSE data payload. */
export function interpretChatCompletionData(
  payload: string,
): ChatCompletionEvent | undefined {
  if (payload === "[DONE]") return { type: "done" };
  let json: unknown;
  try {
    json = JSON.parse(payload);
  } catch {
    return undefined;
  }
  if (!isRecord(json)) return undefined;
  if (isRecord(json.error)) {
    return { type: "error", message: String(json.error.message ?? "unknown error") };
  }
  const choices = json.choices;
  const first = Array.isArray(choices) ? choices[0] : undefined;
  const delta = isRecord(first) ? first.delta : undefined;
  const content = isRecord(delta) ? delta.content : undefined;
  return typeof content === "string" && content.length > 0
    ? { type: "delta", content }
    : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function* decodeBody(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) yield decoder.decode(value, { stream: true });
    }
    const tail = decoder.decode();
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const text = (await response.text()).trim();
    return text ? ` - ${text.slice(0, 300)}` : "";
  } catch {
    return "";
  }
}
