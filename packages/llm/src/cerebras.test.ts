import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { test } from "node:test";
import type { ChatMessage, ChatModel } from "./contract.ts";
import {
  CerebrasChatModel,
  DEFAULT_CEREBRAS_MODEL,
  type FetchLike,
} from "./cerebras.ts";

async function collect(stream: AsyncIterable<string>): Promise<string[]> {
  const out: string[] = [];
  for await (const item of stream) {
    out.push(item);
  }
  return out;
}

/** A fetch stub returning a streamed SSE body, capturing the request it saw. */
function stubFetch(
  sse: string,
  capture?: (url: string, init: RequestInit) => void,
): FetchLike {
  return async (url, init) => {
    capture?.(url, init);
    return new Response(sse, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  };
}

const SSE_HELLO = [
  'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":", world"}}]}\n\n',
  "data: [DONE]\n\n",
].join("");

test("stream() yields content deltas from the SSE body", async () => {
  const model = new CerebrasChatModel({
    apiKey: () => "test-key",
    fetch: stubFetch(SSE_HELLO),
  });
  const parts = await collect(model.stream([{ role: "user", content: "hi" }]));
  assert.deepEqual(parts, ["Hello", ", world"]);
});

test("stream() posts the configured model and auth to the completions endpoint", async () => {
  let seenUrl = "";
  let seenInit: RequestInit = {};
  const model = new CerebrasChatModel({
    apiKey: () => "test-key",
    config: { model: "gemma-4-31b", temperature: 0.2, maxTokens: 64 },
    fetch: stubFetch(SSE_HELLO, (url, init) => {
      seenUrl = url;
      seenInit = init;
    }),
  });
  await collect(model.stream([{ role: "user", content: "hi" }]));

  assert.match(seenUrl, /\/chat\/completions$/);
  const headers = seenInit.headers as Record<string, string>;
  assert.equal(headers.authorization, "Bearer test-key");
  const body = JSON.parse(String(seenInit.body));
  assert.equal(body.model, "gemma-4-31b");
  assert.equal(body.stream, true);
  assert.equal(body.temperature, 0.2);
  assert.equal(body.max_tokens, 64);
});

test("stream() throws a redacted error on HTTP failure", async () => {
  const model = new CerebrasChatModel({
    apiKey: () => "secret-key-value",
    fetch: async () =>
      new Response('{"error":{"message":"unauthorized"}}', { status: 401 }),
  });
  await assert.rejects(
    () => collect(model.stream([{ role: "user", content: "hi" }])),
    (error: Error) => {
      assert.match(error.message, /HTTP 401/);
      assert.doesNotMatch(error.message, /secret-key-value/);
      return true;
    },
  );
});

test("stream() surfaces an in-band error frame", async () => {
  const model = new CerebrasChatModel({
    apiKey: () => "test-key",
    fetch: stubFetch('data: {"error":{"message":"rate limited"}}\n\n'),
  });
  await assert.rejects(
    () => collect(model.stream([{ role: "user", content: "hi" }])),
    /rate limited/,
  );
});

test("stream() throws when no API key is available", async () => {
  const model = new CerebrasChatModel({
    apiKey: () => undefined,
    fetch: stubFetch(SSE_HELLO),
  });
  await assert.rejects(
    () => collect(model.stream([{ role: "user", content: "hi" }])),
    /CEREBRAS_API_KEY is not set/,
  );
});

test("a fake ChatModel satisfies the contract", async () => {
  const fake: ChatModel = {
    async *stream(messages: ChatMessage[]): AsyncGenerator<string> {
      yield `echo:${messages.at(-1)?.content ?? ""}`;
    },
  };
  const parts = await collect(fake.stream([{ role: "user", content: "ping" }]));
  assert.deepEqual(parts, ["echo:ping"]);
});

// Gated live smoke: only runs when a real key is present in the environment.
// It records first-token latency and prints timing only, never any content.
test(
  "live: streams a short completion and records first-token latency",
  { skip: !process.env.CEREBRAS_API_KEY },
  async () => {
    const model = new CerebrasChatModel({ config: { maxTokens: 16 } });
    const start = performance.now();
    let firstTokenMs: number | undefined;
    let tokenCount = 0;
    for await (const token of model.stream([
      { role: "user", content: "Say hi in three words." },
    ])) {
      firstTokenMs ??= performance.now() - start;
      tokenCount += 1;
      void token; // never assert on or print model content
    }
    assert.ok(firstTokenMs !== undefined && firstTokenMs > 0);
    assert.ok(tokenCount > 0);
    console.log(
      `live cerebras model=${DEFAULT_CEREBRAS_MODEL} first-token=${firstTokenMs.toFixed(1)}ms tokens=${tokenCount}`,
    );
  },
);
