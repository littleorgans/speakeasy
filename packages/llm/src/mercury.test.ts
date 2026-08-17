import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { test } from "node:test";
import {
  DEFAULT_MERCURY_MODEL,
  MERCURY_REASONING_EFFORTS,
  MercuryChatModel,
} from "./mercury.ts";

test("Mercury streams through the shared adapter with instant reasoning", async () => {
  let request: RequestInit | undefined;
  const model = new MercuryChatModel({
    apiKey: () => "test-key",
    fetch: async (_url, init) => {
      request = init;
      return sseResponse([
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
        "data: [DONE]\n\n",
      ]);
    },
  });

  const chunks: string[] = [];
  for await (const chunk of model.stream([{ role: "user", content: "hello" }])) {
    chunks.push(chunk);
  }

  assert.deepEqual(chunks, ["Hello", " there"]);
  const body = JSON.parse(String(request?.body)) as Record<string, unknown>;
  assert.equal(body.model, "mercury-2");
  assert.equal(body.reasoning_effort, "instant");
  assert.equal(body.stream, true);
});

test("Mercury accepts the low reasoning comparison mode", async () => {
  let body: Record<string, unknown> | undefined;
  const model = new MercuryChatModel({
    reasoningEffort: "low",
    apiKey: () => "test-key",
    fetch: async (_url, init) => {
      body = JSON.parse(String(init.body)) as Record<string, unknown>;
      return sseResponse(["data: [DONE]\n\n"]);
    },
  });
  for await (const _chunk of model.stream([{ role: "user", content: "hello" }])) {
    // drain
  }
  assert.equal(body?.reasoning_effort, "low");
});

test("Mercury fails before network access when its key is missing", async () => {
  const model = new MercuryChatModel({ apiKey: () => undefined });
  await assert.rejects(
    async () => {
      for await (const _chunk of model.stream([{ role: "user", content: "hello" }])) {
        // drain
      }
    },
    /INCEPTIONLABS_API_KEY/,
  );
});

test(
  "live: Mercury records first-token latency without exposing content",
  {
    skip:
      process.env.SPEAKEASY_LIVE_MERCURY !== "1" ||
      !process.env.INCEPTIONLABS_API_KEY,
  },
  async () => {
    for (const reasoningEffort of MERCURY_REASONING_EFFORTS) {
      const model = new MercuryChatModel({
        config: { maxTokens: 64 },
        reasoningEffort,
      });
      const start = performance.now();
      let firstTokenMs: number | undefined;
      let tokenCount = 0;
      for await (const token of model.stream([
        { role: "user", content: "Say hi in three words." },
      ])) {
        firstTokenMs ??= performance.now() - start;
        tokenCount += 1;
        void token;
      }
      assert.ok(firstTokenMs !== undefined && firstTokenMs > 0);
      assert.ok(tokenCount > 0);
      console.log(
        `live mercury model=${DEFAULT_MERCURY_MODEL} reasoning=${reasoningEffort} first-token=${firstTokenMs.toFixed(1)}ms chunks=${tokenCount}`,
      );
    }
  },
);

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { status: 200 },
  );
}
