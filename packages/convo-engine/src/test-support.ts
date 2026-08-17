import { EventEmitter } from "node:events";
import type { ChatMessage, ChatModel } from "@speakeasy/llm";
import type {
  AudioSegment,
  STTSession,
  TTSSession,
  TextToSpeech,
  VoiceToText,
} from "@speakeasy/speech-io";

export class FakeSTTSession extends EventEmitter implements STTSession {
  readonly pushed: Float32Array[] = [];
  flushes = 0;
  onPushAudio: ((frame: Float32Array) => void) | undefined;
  onFlush: (() => void) | undefined;

  pushAudio(frame: Float32Array): void {
    this.pushed.push(frame);
    this.onPushAudio?.(frame);
  }

  flush(): void {
    this.flushes += 1;
    this.onFlush?.();
  }

  reset(): void {}

  async end(): Promise<void> {}

  /** Mimic sherpa eager commit: endpoint then final in one tick. */
  say(text: string): void {
    this.emit("endpoint", {});
    this.emit("final", { text });
  }
}

export class FakeSTT implements VoiceToText {
  readonly session = new FakeSTTSession();

  async open(): Promise<STTSession> {
    return this.session;
  }
}

export class FakeLLM implements ChatModel {
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
      this.lastTokenAt = performance.now();
      yield token;
    }
  }
}

export class FakeTTS implements TextToSpeech {
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
      yield fakeAudioSegment(0);
      return;
    }
    let index = 0;
    for await (const _token of text) {
      if (index === 0) {
        this.firstAudioAt = performance.now();
        yield fakeAudioSegment(0);
      }
      index += 1;
    }
    if (index > 0) {
      yield fakeAudioSegment(1);
    }
  }

  async close(): Promise<void> {
    this.closed += 1;
  }
}

export function fakeAudioSegment(index: number): AudioSegment {
  return {
    index,
    sentence: `s${index}`,
    samples: new Float32Array(1_600).fill(0.1),
    sampleRate: 16_000,
    readyAtMs: 0,
    synthMs: 1,
    audioDurationMs: 100,
  };
}
