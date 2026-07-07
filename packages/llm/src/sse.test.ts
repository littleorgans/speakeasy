import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSSEStream } from "./sse.ts";
import { interpretCerebrasData } from "./cerebras.ts";

async function* chunks(...parts: string[]): AsyncGenerator<string> {
  for (const part of parts) {
    yield part;
  }
}

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of stream) {
    out.push(item);
  }
  return out;
}

test("parseSSEStream extracts data payloads across well-formed frames", async () => {
  const payloads = await collect(
    parseSSEStream(
      chunks(
        'data: {"a":1}\n\n',
        'data: {"b":2}\n\n',
        "data: [DONE]\n\n",
      ),
    ),
  );
  assert.deepEqual(payloads, ['{"a":1}', '{"b":2}', "[DONE]"]);
});

test("parseSSEStream reassembles frames split across chunk boundaries", async () => {
  const payloads = await collect(
    parseSSEStream(
      chunks('data: {"choi', 'ces":[{"delta":{"content":"Hi"}}]}\n', "\n"),
    ),
  );
  assert.deepEqual(payloads, ['{"choices":[{"delta":{"content":"Hi"}}]}']);
});

test("parseSSEStream flushes a final line with no trailing newline", async () => {
  const payloads = await collect(parseSSEStream(chunks("data: [DONE]")));
  assert.deepEqual(payloads, ["[DONE]"]);
});

test("parseSSEStream ignores blank lines and non-data lines", async () => {
  const payloads = await collect(
    parseSSEStream(chunks(": keep-alive\n", "event: message\n", 'data: {"x":1}\n\n')),
  );
  assert.deepEqual(payloads, ['{"x":1}']);
});

test("interpretCerebrasData decodes deltas, done, errors, and skips empties", () => {
  assert.deepEqual(
    interpretCerebrasData('{"choices":[{"delta":{"content":"Hello"}}]}'),
    { type: "delta", content: "Hello" },
  );
  assert.deepEqual(interpretCerebrasData("[DONE]"), { type: "done" });
  assert.deepEqual(interpretCerebrasData('{"error":{"message":"bad request"}}'), {
    type: "error",
    message: "bad request",
  });
  // role-only opening frame carries no content
  assert.equal(
    interpretCerebrasData('{"choices":[{"delta":{"role":"assistant"}}]}'),
    undefined,
  );
  // non-JSON keep-alive
  assert.equal(interpretCerebrasData(": ping"), undefined);
});

test("parseSSEStream + interpret reconstruct the full assistant message", async () => {
  const frames = [
    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    "data: [DONE]\n\n",
  ];
  // Deliberately re-chunk mid-frame to stress the buffer.
  const joined = frames.join("");
  const mid = Math.floor(joined.length / 2);
  let text = "";
  for await (const payload of parseSSEStream(chunks(joined.slice(0, mid), joined.slice(mid)))) {
    const event = interpretCerebrasData(payload);
    if (event?.type === "delta") {
      text += event.content;
    }
  }
  assert.equal(text, "Hello");
});
