import type { TTSConfig, TTSSession, TextToSpeech } from "@speakeasy/speech-io";
import type { ChatMessage, ChatModel } from "@speakeasy/llm";
import type { ResponderEvent, ResponderSession, VoiceResponder } from "./contract.ts";

/**
 * The cascaded implementation of VoiceResponder: ChatModel tokens piped
 * straight into a TTSSession sentence pipeline. This is the exact wiring the
 * loop used to own inline; it moved behind the seam so fused speech models can
 * be swapped in without touching the loop.
 *
 * Token events are queued as the TTS pipeline pulls the stream and flushed
 * ahead of each audio segment, stamped with their true arrival time (sentence
 * synthesis would otherwise skew first-token latency).
 */

export type CascadeDeps = {
  llm: ChatModel;
  tts: TextToSpeech;
  ttsConfig?: TTSConfig;
  now?: () => number;
};

export class CascadeResponder implements VoiceResponder {
  readonly #deps: CascadeDeps;

  constructor(deps: CascadeDeps) {
    this.#deps = deps;
  }

  async open(): Promise<ResponderSession> {
    const session = await this.#deps.tts.open(this.#deps.ttsConfig);
    return new CascadeSession(
      this.#deps.llm,
      session,
      this.#deps.now ?? (() => performance.now()),
    );
  }
}

class CascadeSession implements ResponderSession {
  readonly #llm: ChatModel;
  readonly #tts: TTSSession;
  readonly #now: () => number;

  constructor(llm: ChatModel, tts: TTSSession, now: () => number) {
    this.#llm = llm;
    this.#tts = tts;
    this.#now = now;
  }

  async *respond(messages: ChatMessage[]): AsyncGenerator<ResponderEvent> {
    const pending: ResponderEvent[] = [];
    const tokens = tap(this.#llm.stream(messages), (text) =>
      pending.push({ type: "token", text, at: this.#now() }),
    );
    for await (const segment of this.#tts.speak(tokens)) {
      yield* pending.splice(0);
      yield { type: "audio", segment };
    }
    yield* pending.splice(0); // trailing tokens after the last audio segment
  }

  async close(): Promise<void> {
    await this.#tts.close();
  }
}

/** Observe a token stream without buffering it. */
async function* tap(
  source: AsyncIterable<string>,
  onToken: (token: string) => void,
): AsyncGenerator<string> {
  for await (const token of source) {
    onToken(token);
    yield token;
  }
}
