import assert from "node:assert/strict";
import { test } from "node:test";
import type { ChatMessage } from "@speakeasy/llm";
import type { ResponderEvent } from "./contract.ts";
import {
  OpenAIRealtimeResponder,
  decodePcm16,
  type SocketLike,
} from "./openai-realtime.ts";

/**
 * Fake ws client: records sent payloads, lets tests emit server events, and
 * auto-assigns a response id when it sees response.create (mirroring the
 * server's response.created acknowledgement).
 */
class FakeSocket implements SocketLike {
  readonly sent: Record<string, unknown>[] = [];
  #listeners = new Map<string, ((...args: never[]) => void)[]>();
  closed = 0;

  on(event: string, listener: (...args: never[]) => void): void {
    const list = this.#listeners.get(event) ?? [];
    list.push(listener);
    this.#listeners.set(event, list);
  }

  send(data: string): void {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }

  close(): void {
    this.closed += 1;
    this.#emit("close");
  }

  open(): void {
    this.#emit("open");
  }

  emitEvent(event: Record<string, unknown>): void {
    this.#emit("message", { toString: () => JSON.stringify(event) });
  }

  #emit(event: string, ...args: unknown[]): void {
    for (const listener of this.#listeners.get(event) ?? []) {
      (listener as (...a: unknown[]) => void)(...args);
    }
  }
}

function makeResponder(socket: FakeSocket): OpenAIRealtimeResponder {
  return new OpenAIRealtimeResponder({
    apiKey: () => "test-key",
    createSocket: () => {
      queueMicrotask(() => socket.open());
      return socket;
    },
  });
}

function pcm16Base64(...values: number[]): string {
  const buffer = Buffer.alloc(values.length * 2);
  values.forEach((value, i) => buffer.writeInt16LE(value, i * 2));
  return buffer.toString("base64");
}

const MESSAGES: ChatMessage[] = [
  { role: "system", content: "Be brief." },
  { role: "user", content: "hello there" },
];

/** Drive one respond() turn while feeding server events after the request lands. */
async function collectTurn(
  socket: FakeSocket,
  events: Record<string, unknown>[],
  options: { stopAfter?: number } = {},
): Promise<{ collected: ResponderEvent[]; session: Awaited<ReturnType<OpenAIRealtimeResponder["open"]>> }> {
  const responder = makeResponder(socket);
  const session = await responder.open();
  queueMicrotask(() => {
    for (const event of events) {
      socket.emitEvent(event);
    }
  });
  const collected: ResponderEvent[] = [];
  for await (const event of session.respond(MESSAGES)) {
    collected.push(event);
    if (options.stopAfter !== undefined && collected.length >= options.stopAfter) {
      break;
    }
  }
  return { collected, session };
}

test("open() configures the session: pcm output, voice, then instructions on first turn", async () => {
  const socket = new FakeSocket();
  const { session } = await collectTurn(socket, [
    { type: "response.created", response: { id: "r1" } },
    { type: "response.done", response_id: "r1" },
  ]);

  const types = socket.sent.map((p) => p.type);
  assert.deepEqual(types, [
    "session.update", // audio format + voice
    "session.update", // instructions from the system message
    "conversation.item.create",
    "response.create",
  ]);
  const item = socket.sent[2] as { item: { role: string; content: { text: string }[] } };
  assert.equal(item.item.role, "user");
  assert.equal(item.item.content[0]?.text, "hello there");
  await session.close();
});

test("audio and transcript deltas stream out as responder events", async () => {
  const socket = new FakeSocket();
  const audio = pcm16Base64(0, 16384, -16384, 32767);
  const { collected } = await collectTurn(socket, [
    { type: "response.created", response: { id: "r1" } },
    { type: "response.output_audio_transcript.delta", response_id: "r1", delta: "Hi" },
    { type: "response.output_audio.delta", response_id: "r1", delta: audio },
    { type: "response.output_audio_transcript.delta", response_id: "r1", delta: " there" },
    { type: "response.done", response_id: "r1" },
  ]);

  assert.deepEqual(
    collected.map((e) => e.type),
    ["token", "audio", "token"],
  );
  const segment = collected[1]!.type === "audio" ? collected[1].segment : undefined;
  assert.ok(segment);
  assert.equal(segment.sampleRate, 24000);
  assert.equal(segment.samples.length, 4);
  assert.ok(Math.abs(segment.samples[1]! - 0.5) < 0.001);
  assert.ok(segment.audioDurationMs > 0);
  const reply = collected
    .filter((e) => e.type === "token")
    .map((e) => (e.type === "token" ? e.text : ""))
    .join("");
  assert.equal(reply, "Hi there");
});

test("events tagged with a stale response id are dropped", async () => {
  const socket = new FakeSocket();
  const { collected } = await collectTurn(socket, [
    { type: "response.output_audio_transcript.delta", response_id: "r0", delta: "stale" },
    { type: "response.created", response: { id: "r1" } },
    { type: "response.output_audio_transcript.delta", response_id: "r0", delta: "stale2" },
    { type: "response.output_audio_transcript.delta", response_id: "r1", delta: "fresh" },
    { type: "response.done", response_id: "r1" },
  ]);

  assert.deepEqual(
    collected.map((e) => (e.type === "token" ? e.text : "?")),
    ["fresh"],
  );
});

test("breaking out early cancels the in-flight response", async () => {
  const socket = new FakeSocket();
  const audio = pcm16Base64(1000, 2000);
  const { collected } = await collectTurn(
    socket,
    [
      { type: "response.created", response: { id: "r1" } },
      { type: "response.output_audio.delta", response_id: "r1", delta: audio },
      { type: "response.output_audio.delta", response_id: "r1", delta: audio },
    ],
    { stopAfter: 1 },
  );

  assert.equal(collected.length, 1);
  assert.ok(socket.sent.some((p) => p.type === "response.cancel"));
});

test("a completed turn does not send response.cancel", async () => {
  const socket = new FakeSocket();
  await collectTurn(socket, [
    { type: "response.created", response: { id: "r1" } },
    { type: "response.done", response_id: "r1" },
  ]);
  assert.ok(!socket.sent.some((p) => p.type === "response.cancel"));
});

test("a server error event rejects the turn", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  queueMicrotask(() => {
    socket.emitEvent({ type: "response.created", response: { id: "r1" } });
    socket.emitEvent({ type: "error", error: { message: "rate limited" } });
  });
  await assert.rejects(
    async () => {
      for await (const _event of session.respond(MESSAGES)) {
        // drain
      }
    },
    /rate limited/,
  );
});

test("open() fails fast without an API key", async () => {
  const responder = new OpenAIRealtimeResponder({ apiKey: () => undefined });
  await assert.rejects(() => responder.open(), /OPENAI_API_KEY/);
});

test("decodePcm16 maps int16 to [-1, 1] floats", () => {
  const samples = decodePcm16(pcm16Base64(0, 32767, -32768));
  assert.equal(samples.length, 3);
  assert.equal(samples[0], 0);
  assert.ok(Math.abs(samples[1]! - 0.99997) < 0.001);
  assert.equal(samples[2], -1);
});
