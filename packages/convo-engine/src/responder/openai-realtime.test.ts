import assert from "node:assert/strict";
import { test } from "node:test";
import type { ChatMessage } from "@speakeasy/llm";
import type { ResponderEvent } from "./contract.ts";
import {
  OpenAIRealtimeResponder,
  decodePcm16,
  parseRealtimeVoice,
  type SocketLike,
} from "./openai-realtime.ts";

test("Realtime voice validation accepts supported voices and rejects unknown values", () => {
  assert.equal(parseRealtimeVoice("marin"), "marin");
  assert.equal(parseRealtimeVoice("cedar"), "cedar");
  assert.throws(() => parseRealtimeVoice("mystery"), /Unsupported Realtime voice/);
});

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

function pcm16Silence(durationMs: number): string {
  const samples = Math.floor((24_000 * durationMs) / 1_000);
  return Buffer.alloc(samples * 2).toString("base64");
}

const MESSAGES: ChatMessage[] = [
  { role: "system", content: "Be brief." },
  { role: "user", content: "hello there" },
];

/** Drive one respond() turn while feeding server events after the request lands. */
async function collectTurn(
  socket: FakeSocket,
  events: Record<string, unknown>[],
  options: { stopAfter?: number; playedAudioMs?: number } = {},
): Promise<{ collected: ResponderEvent[]; session: Awaited<ReturnType<OpenAIRealtimeResponder["open"]>> }> {
  const responder = makeResponder(socket);
  const session = await responder.open();
  const collected: ResponderEvent[] = [];
  const collecting = (async () => {
    for await (const event of session.respond(MESSAGES)) {
      collected.push(event);
      if (options.stopAfter !== undefined && collected.length >= options.stopAfter) {
        session.interrupt(Promise.resolve(options.playedAudioMs));
        break;
      }
    }
  })();
  await waitFor(() => socket.sent.some((event) => event.type === "response.create"));
  for (const event of events) socket.emitEvent(event);
  await collecting;
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

test("interruption cancels generation and truncates remote context to played audio", async () => {
  const socket = new FakeSocket();
  const audio = pcm16Silence(500);
  const { collected, session } = await collectTurn(
    socket,
    [
      { type: "response.created", response: { id: "r1" } },
      {
        type: "response.output_audio.delta",
        response_id: "r1",
        item_id: "item-1",
        content_index: 0,
        delta: audio,
      },
      {
        type: "response.output_audio.delta",
        response_id: "r1",
        item_id: "item-1",
        content_index: 0,
        delta: audio,
      },
    ],
    { stopAfter: 1, playedAudioMs: 375.9 },
  );

  assert.equal(collected.length, 1);
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );
  assert.deepEqual(socket.sent.slice(-2), [
    { event_id: "cancel_1", type: "response.cancel", response_id: "r1" },
    {
      event_id: "truncate_2",
      type: "conversation.item.truncate",
      item_id: "item-1",
      content_index: 0,
      audio_end_ms: 375,
    },
  ]);
  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 375,
  });
  await session.close();
});

test("interruption before the first audio delta truncates the assistant item at zero", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  const first = session.respond(MESSAGES)[Symbol.asyncIterator]();
  const firstResult = first.next();
  await waitFor(() =>
    socket.sent.some((event) => event.type === "response.create"),
  );
  socket.emitEvent({ type: "response.created", response: { id: "r1" } });
  socket.emitEvent({
    type: "response.output_item.added",
    response_id: "r1",
    item: { id: "item-1", type: "message" },
  });

  session.interrupt(Promise.resolve(0));
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );
  assert.deepEqual(socket.sent.slice(-2), [
    { event_id: "cancel_1", type: "response.cancel", response_id: "r1" },
    {
      event_id: "truncate_2",
      type: "conversation.item.truncate",
      item_id: "item-1",
      content_index: 0,
      audio_end_ms: 0,
    },
  ]);

  socket.emitEvent({ type: "response.done", response_id: "r1" });
  await firstResult;
  const secondMessages: ChatMessage[] = [
    ...MESSAGES,
    { role: "user", content: "new question" },
  ];
  const second = session.respond(secondMessages)[Symbol.asyncIterator]();
  const secondResult = second.next();
  await Promise.resolve();
  assert.equal(
    socket.sent.filter((event) => event.type === "conversation.item.create").length,
    1,
  );

  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 0,
  });
  await waitFor(() =>
    socket.sent.filter((event) => event.type === "conversation.item.create").length === 2,
  );
  socket.emitEvent({ type: "response.created", response: { id: "r2" } });
  socket.emitEvent({ type: "response.done", response_id: "r2" });
  await secondResult;
  await session.close();
});

test("the next response waits until interrupted context is truncated", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  const first = session.respond(MESSAGES)[Symbol.asyncIterator]();
  const audio = pcm16Silence(300);
  const firstResult = first.next();
  await waitFor(() => socket.sent.some((event) => event.type === "response.create"));
  socket.emitEvent({ type: "response.created", response: { id: "r1" } });
  socket.emitEvent({
    type: "response.output_audio.delta",
    response_id: "r1",
    item_id: "item-1",
    content_index: 0,
    delta: audio,
  });
  assert.equal((await firstResult).value?.type, "audio");

  let resolvePosition: (value: number) => void = () => {};
  const position = new Promise<number>((resolve) => {
    resolvePosition = resolve;
  });
  session.interrupt(position);
  await first.return?.();

  const secondMessages: ChatMessage[] = [
    ...MESSAGES,
    { role: "user", content: "new question" },
  ];
  const second = session.respond(secondMessages)[Symbol.asyncIterator]();
  const secondResult = second.next();
  await Promise.resolve();
  assert.equal(socket.sent.filter((event) => event.type === "response.create").length, 1);

  resolvePosition(240);
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );
  assert.equal(socket.sent.filter((event) => event.type === "response.create").length, 1);
  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 240,
  });
  await waitFor(() =>
    socket.sent.filter((event) => event.type === "response.create").length === 2,
  );
  queueMicrotask(() => {
    socket.emitEvent({ type: "response.created", response: { id: "r2" } });
    socket.emitEvent({ type: "response.done", response_id: "r2" });
  });
  await secondResult;

  const orderedTypes = socket.sent.map((event) => event.type);
  assert.ok(
    orderedTypes.indexOf("conversation.item.truncate") <
      orderedTypes.lastIndexOf("conversation.item.create"),
  );
  await session.close();
});

test("completed generation remains truncatable while its audio is buffered", async () => {
  const socket = new FakeSocket();
  const audio = pcm16Silence(250);
  const { session } = await collectTurn(socket, [
    { type: "response.created", response: { id: "r1" } },
    {
      type: "response.output_audio.delta",
      response_id: "r1",
      item_id: "item-1",
      content_index: 0,
      delta: audio,
    },
    { type: "response.done", response_id: "r1" },
  ]);

  session.interrupt(Promise.resolve(180));
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );

  assert.deepEqual(socket.sent.at(-1), {
    event_id: "truncate_1",
    type: "conversation.item.truncate",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 180,
  });
  assert.equal(
    socket.sent.filter((event) => event.type === "response.cancel").length,
    0,
  );
  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 180,
  });
  await session.close();
});

test("truncation cannot exceed audio delivered to browser playback", async () => {
  const socket = new FakeSocket();
  const audio = pcm16Silence(100);
  const { session } = await collectTurn(
    socket,
    [
      { type: "response.created", response: { id: "r1" } },
      {
        type: "response.output_audio.delta",
        response_id: "r1",
        item_id: "item-1",
        content_index: 0,
        delta: audio,
      },
    ],
    { stopAfter: 1, playedAudioMs: 105.8 },
  );
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );

  assert.equal(socket.sent.at(-1)?.audio_end_ms, 100);
  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 100,
  });
  await session.close();
});

test("a completed turn does not send response.cancel", async () => {
  const socket = new FakeSocket();
  await collectTurn(socket, [
    { type: "response.created", response: { id: "r1" } },
    { type: "response.done", response_id: "r1" },
  ]);
  assert.ok(!socket.sent.some((p) => p.type === "response.cancel"));
});

test("a completed response cancellation race remains recoverable", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  const first = session.respond(MESSAGES)[Symbol.asyncIterator]();
  const firstAudio = first.next();
  await waitFor(() => socket.sent.some((event) => event.type === "response.create"));
  socket.emitEvent({ type: "response.created", response: { id: "r1" } });
  socket.emitEvent({
    type: "response.output_audio.delta",
    response_id: "r1",
    item_id: "item-1",
    content_index: 0,
    delta: pcm16Silence(300),
  });
  assert.equal((await firstAudio).value?.type, "audio");

  session.interrupt(Promise.resolve(100));
  const cancel = socket.sent.find((event) => event.type === "response.cancel");
  assert.deepEqual(cancel, {
    event_id: "cancel_1",
    type: "response.cancel",
    response_id: "r1",
  });

  const firstDone = first.next();
  socket.emitEvent({
    type: "error",
    error: {
      type: "invalid_request_error",
      code: "response_cancel_not_active",
      message: "Cancellation failed: no active response found",
      event_id: "cancel_1",
    },
  });
  assert.equal((await firstDone).done, true);

  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );
  const secondMessages: ChatMessage[] = [
    ...MESSAGES,
    { role: "user", content: "new question" },
  ];
  const second = session.respond(secondMessages)[Symbol.asyncIterator]();
  const secondDone = second.next();
  await Promise.resolve();
  assert.equal(
    socket.sent.filter((event) => event.type === "conversation.item.create").length,
    1,
  );

  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 100,
  });
  await waitFor(() =>
    socket.sent.filter((event) => event.type === "conversation.item.create").length === 2,
  );
  socket.emitEvent({ type: "response.created", response: { id: "r2" } });
  socket.emitEvent({ type: "response.done", response_id: "r2" });
  assert.equal((await secondDone).done, true);
  await session.close();
});

test("a late correlated cancellation error does not poison the next turn", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  const first = session.respond(MESSAGES)[Symbol.asyncIterator]();
  const firstAudio = first.next();
  await waitFor(() => socket.sent.some((event) => event.type === "response.create"));
  socket.emitEvent({ type: "response.created", response: { id: "r1" } });
  socket.emitEvent({
    type: "response.output_audio.delta",
    response_id: "r1",
    item_id: "item-1",
    content_index: 0,
    delta: pcm16Silence(300),
  });
  await firstAudio;
  session.interrupt(Promise.resolve(100));
  await first.return?.();
  await waitFor(() =>
    socket.sent.some((event) => event.type === "conversation.item.truncate"),
  );
  socket.emitEvent({
    type: "conversation.item.truncated",
    item_id: "item-1",
    content_index: 0,
    audio_end_ms: 100,
  });

  const second = session.respond([
    ...MESSAGES,
    { role: "user", content: "new question" },
  ])[Symbol.asyncIterator]();
  const secondDone = second.next();
  await waitFor(() =>
    socket.sent.filter((event) => event.type === "response.create").length === 2,
  );
  socket.emitEvent({
    type: "error",
    error: {
      message: "Cancellation failed: no active response found",
      event_id: "cancel_1",
    },
  });
  socket.emitEvent({ type: "response.created", response: { id: "r2" } });
  socket.emitEvent({ type: "response.done", response_id: "r2" });
  assert.equal((await secondDone).done, true);
  await session.close();
});

test("an uncorrelated completed response cancellation error remains fatal", async () => {
  const socket = new FakeSocket();
  const responder = makeResponder(socket);
  const session = await responder.open();
  queueMicrotask(() => {
    socket.emitEvent({ type: "response.created", response: { id: "r1" } });
    socket.emitEvent({
      type: "error",
      error: {
        message: "Cancellation failed: no active response found",
        event_id: "unknown_cancel",
      },
    });
  });
  await assert.rejects(
    async () => {
      for await (const _event of session.respond(MESSAGES)) {
        // drain
      }
    },
    /Cancellation failed: no active response found/,
  );
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

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 1_000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for Realtime event");
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}
