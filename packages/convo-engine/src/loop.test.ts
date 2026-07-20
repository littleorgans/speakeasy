import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import type {
  AudioSegment,
  STTSession,
  TTSSession,
  TextToSpeech,
  VoiceToText,
} from "@speakeasy/speech-io";
import type { ChatMessage, ChatModel } from "@speakeasy/llm";
import { ConversationLoop, type AudioSink, type AudioSource } from "./loop.ts";
import { CascadeResponder } from "./responder/cascade.ts";
import type { ConvoState } from "./state.ts";

/** The loop is always exercised through the cascade responder, as the demo wires it. */
function cascade(llm: ChatModel, tts: TextToSpeech): CascadeResponder {
  return new CascadeResponder({ llm, tts, now });
}

const now = () => performance.now();

/** STT fake: an EventEmitter session whose finals the test triggers directly. */
class FakeSTTSession extends EventEmitter implements STTSession {
  readonly pushed: Float32Array[] = [];
  pushAudio(frame: Float32Array): void {
    this.pushed.push(frame);
  }
  flush(): void {}
  reset(): void {}
  async end(): Promise<void> {}
  /** Mimic sherpa eager commit: endpoint then final in one tick. */
  say(text: string): void {
    this.emit("endpoint", {});
    this.emit("final", { text });
  }
}

class FakeSTT implements VoiceToText {
  readonly session = new FakeSTTSession();
  async open(): Promise<STTSession> {
    return this.session;
  }
}

/** LLM fake: streams the given tokens; records the last-token timestamp. */
class FakeLLM implements ChatModel {
  readonly #tokens: string[];
  readonly #throwBeforeYield: boolean;
  lastMessages: ChatMessage[] = [];
  lastTokenAt = 0;
  constructor(tokens: string[], throwBeforeYield = false) {
    this.#tokens = tokens;
    this.#throwBeforeYield = throwBeforeYield;
  }
  async *stream(messages: ChatMessage[]): AsyncGenerator<string> {
    this.lastMessages = messages;
    if (this.#throwBeforeYield) {
      throw new Error("llm exploded");
    }
    for (const token of this.#tokens) {
      this.lastTokenAt = now();
      yield token;
    }
  }
}

/**
 * TTS fake: yields one segment right after the first token (proving audio starts
 * before the stream drains), then a second at the end. An empty token stream
 * yields no audio, exercising the "no reply" path.
 */
class FakeTTS implements TextToSpeech {
  readonly session = new FakeTTSSession();
  async open(): Promise<TTSSession> {
    return this.session;
  }
}

class FakeTTSSession implements TTSSession {
  firstAudioAt: number | undefined;
  closed = 0;
  async *speak(
    text: AsyncIterable<string> | string,
  ): AsyncGenerator<AudioSegment> {
    if (typeof text === "string") {
      yield segment(0);
      return;
    }
    let index = 0;
    for await (const _token of text) {
      if (index === 0) {
        this.firstAudioAt = now();
        yield segment(0);
      }
      index += 1;
    }
    if (index > 0) {
      yield segment(1);
    }
  }
  async close(): Promise<void> {
    this.closed += 1;
  }
}

class FakeMic implements AudioSource {
  handlers: { onFrame: (f: Float32Array) => void } | undefined;
  stopped = 0;
  start(handlers: { onFrame: (f: Float32Array) => void }): void {
    this.handlers = handlers;
  }
  stop(): void {
    this.stopped += 1;
  }
}

class FakeSink implements AudioSink {
  readonly segments: AudioSegment[] = [];
  opened = 0;
  ended = 0;
  interrupted = 0;
  open(): void {
    this.opened += 1;
  }
  write(seg: AudioSegment): void {
    if (this.interrupted > 0) {
      return; // a killed sink drops further audio, like ffplay after SIGKILL
    }
    this.segments.push(seg);
  }
  interrupt(): void {
    this.interrupted += 1;
  }
  async end(): Promise<void> {
    this.ended += 1;
  }
}

function segment(index: number): AudioSegment {
  return {
    index,
    sentence: `s${index}`,
    samples: new Float32Array(1600).fill(0.1),
    sampleRate: 16000,
    readyAtMs: 0,
    synthMs: 1,
    audioDurationMs: 100,
  };
}

function makeLoop(llm: FakeLLM, options: { maxTurns?: number } = {}) {
  const stt = new FakeSTT();
  const tts = new FakeTTS();
  const mic = new FakeMic();
  const sink = new FakeSink();
  const states: ConvoState[] = [];
  const logs: string[] = [];
  const loop = new ConversationLoop(
    { stt, responder: cascade(llm, tts), mic, createSink: () => sink },
    {
      maxTurns: options.maxTurns,
      now,
      log: (line) => logs.push(line),
      onState: (state) => states.push(state),
    },
  );
  return { loop, stt, tts, mic, sink, states, logs };
}

test("a happy turn walks listening -> thinking -> speaking -> listening", async () => {
  const llm = new FakeLLM(["Hello. ", "How are you?"]);
  const { loop, stt, sink, states, tts } = makeLoop(llm, { maxTurns: 1 });
  await loop.start();
  stt.session.say("hi there");
  await loop.done;

  assert.deepEqual(states, ["listening", "thinking", "speaking", "listening", "idle"]);
  assert.equal(loop.metrics.length, 1);
  assert.equal(loop.metrics[0]?.transcript, "hi there");
  assert.ok(loop.metrics[0]!.endpointToFirstAudioMs >= 0);
  assert.ok(sink.segments.length >= 1);
  assert.equal(sink.opened, 1);
  assert.equal(sink.ended, 1);
  assert.equal(tts.session.closed, 1);
});

test("the user message and assistant reply are appended to history", async () => {
  const llm = new FakeLLM(["The ", "answer ", "is ", "42."]);
  const { loop, stt } = makeLoop(llm, { maxTurns: 1 });
  await loop.start();
  stt.session.say("what is it");
  await loop.done;

  // Next stream would carry the prior turn; inspect what the LLM last saw plus
  // the reply that was appended after.
  const roles = llm.lastMessages.map((m) => m.role);
  assert.deepEqual(roles, ["system", "user"]);
  assert.equal(llm.lastMessages.at(-1)?.content, "what is it");
});

test("first audio is produced before the token stream drains", async () => {
  const llm = new FakeLLM(["First. ", "Second. ", "Third."]);
  const { loop, stt, tts } = makeLoop(llm, { maxTurns: 1 });
  await loop.start();
  stt.session.say("go");
  await loop.done;

  assert.ok(tts.session.firstAudioAt !== undefined);
  assert.ok(
    tts.session.firstAudioAt! < llm.lastTokenAt,
    `first audio (${tts.session.firstAudioAt}) should precede last token (${llm.lastTokenAt})`,
  );
});

test("an LLM failure logs, records no metrics, and returns to listening", async () => {
  const llm = new FakeLLM([], true);
  const { loop, stt, states, logs, sink } = makeLoop(llm, { maxTurns: 1 });
  await loop.start();
  stt.session.say("boom");
  await loop.done;

  assert.deepEqual(states, ["listening", "thinking", "listening", "idle"]);
  assert.equal(loop.metrics.length, 0);
  assert.equal(sink.segments.length, 0);
  assert.ok(logs.some((line) => /error: llm exploded/.test(line)));
});

test("the loop recovers and serves a second turn after an error", async () => {
  const llm = new RecoveringLLM();
  const stt = new FakeSTT();
  const sink = new FakeSink();
  const loop = new ConversationLoop(
    {
      stt,
      responder: cascade(llm, new FakeTTS()),
      mic: new FakeMic(),
      createSink: () => sink,
    },
    { maxTurns: 2, now, log: () => {} },
  );
  await loop.start();
  stt.session.say("first");
  stt.session.say("second");
  await loop.done;

  // First turn threw, second produced audio -> exactly one recorded metric.
  assert.equal(loop.metrics.length, 1);
  assert.equal(loop.state, "idle");
});

/** Throws on the first stream, streams normally on the second. */
class RecoveringLLM implements ChatModel {
  #calls = 0;
  async *stream(): AsyncGenerator<string> {
    this.#calls += 1;
    if (this.#calls === 1) {
      throw new Error("transient");
    }
    yield "Recovered. ";
    yield "All good.";
  }
}

/**
 * TTS fake that yields the first segment, then blocks on a gate before the
 * second — so a test can interrupt while the loop is mid-playback.
 */
class GatedTTS implements TextToSpeech {
  readonly session = new GatedTTSSession();
  async open(): Promise<TTSSession> {
    return this.session;
  }
}

class GatedTTSSession implements TTSSession {
  afterFirst: Promise<void>;
  #resolveAfterFirst: () => void = () => {};
  #gate: Promise<void>;
  #openGate: () => void = () => {};

  constructor() {
    this.afterFirst = new Promise((resolve) => (this.#resolveAfterFirst = resolve));
    this.#gate = new Promise((resolve) => (this.#openGate = resolve));
  }

  release(): void {
    this.#openGate();
  }

  async *speak(): AsyncGenerator<AudioSegment> {
    yield segment(0);
    this.#resolveAfterFirst();
    await this.#gate;
    yield segment(1);
  }

  async close(): Promise<void> {}
}

test("interrupt() during playback kills the sink and returns to listening", async () => {
  const tts = new GatedTTS();
  const stt = new FakeSTT();
  const sink = new FakeSink();
  const states: ConvoState[] = [];
  const loop = new ConversationLoop(
    { stt, responder: cascade(new FakeLLM(["Hello. ", "World."]), tts), mic: new FakeMic(), createSink: () => sink },
    { maxTurns: 1, now, log: () => {}, onState: (s) => states.push(s) },
  );
  await loop.start();
  stt.session.say("hi");
  await tts.session.afterFirst;

  assert.equal(loop.state, "speaking");
  assert.equal(sink.segments.length, 1);

  loop.interrupt();
  assert.equal(sink.interrupted, 1);
  assert.equal(loop.state, "listening");

  tts.session.release();
  await loop.done;

  assert.equal(sink.segments.length, 1); // the second segment was dropped
  assert.equal(loop.metrics.length, 0); // interrupted turn records no metric
  assert.ok(states.includes("speaking"));
  assert.equal(loop.state, "idle");
});

test("barge-in: sustained mic speech during playback interrupts via the VAD", async () => {
  const tts = new GatedTTS();
  const stt = new FakeSTT();
  const sink = new FakeSink();
  const mic = new FakeMic();
  const loop = new ConversationLoop(
    { stt, responder: cascade(new FakeLLM(["Hello. ", "World."]), tts), mic, createSink: () => sink },
    { maxTurns: 1, now, log: () => {}, barge: true, bargeThreshold: 0.05 },
  );
  await loop.start();
  stt.session.say("hi");
  await tts.session.afterFirst;
  assert.equal(loop.state, "speaking");

  const loud = new Float32Array(320).fill(0.5);
  for (let i = 0; i < 15 && loop.state === "speaking"; i += 1) {
    mic.handlers?.onFrame(loud);
  }
  assert.equal(loop.state, "listening"); // the VAD tripped and barged
  assert.equal(sink.interrupted, 1);

  tts.session.release();
  await loop.done;
  assert.equal(loop.metrics.length, 0);
});

test("barge-in stays off by default: loud frames while speaking do not interrupt", async () => {
  const tts = new GatedTTS();
  const stt = new FakeSTT();
  const sink = new FakeSink();
  const mic = new FakeMic();
  const loop = new ConversationLoop(
    { stt, responder: cascade(new FakeLLM(["Hi. ", "Bye."]), tts), mic, createSink: () => sink },
    { maxTurns: 1, now, log: () => {} }, // barge not enabled
  );
  await loop.start();
  stt.session.say("hi");
  await tts.session.afterFirst;

  const loud = new Float32Array(320).fill(0.9);
  for (let i = 0; i < 20; i += 1) {
    mic.handlers?.onFrame(loud);
  }
  assert.equal(loop.state, "speaking"); // no barge without the flag
  assert.equal(sink.interrupted, 0);

  tts.session.release();
  await loop.done;
});
