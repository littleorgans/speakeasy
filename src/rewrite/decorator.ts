import { EventEmitter } from "node:events";
import type {
  FinalEvent,
  RewriteConfig,
  STTConfig,
  STTSession,
  VoiceToText,
} from "../contract.ts";
import { rewriteText } from "./pipeline.ts";

/**
 * withRewrite: wrap any VoiceToText so committed final text is post-processed
 * by the rewrite pipeline before consumers see it. The wrapped engine stays
 * pure; this is the single production rewrite path.
 *
 * FINAL events only. "partial" hypotheses are revised as more audio arrives,
 * so rewriting them would flicker; they pass through untouched. Rewriting
 * partials is a possible future option, deliberately not built.
 */
export function withRewrite(
  engine: VoiceToText,
  config: RewriteConfig,
): VoiceToText {
  return {
    async open(sttConfig?: STTConfig): Promise<STTSession> {
      const inner = await engine.open(sttConfig);
      return new RewriteSession(inner, config);
    },
  };
}

class RewriteSession extends EventEmitter implements STTSession {
  #inner: STTSession;

  constructor(inner: STTSession, config: RewriteConfig) {
    super();
    this.#inner = inner;
    inner.on("partial", (event) => this.emit("partial", event));
    inner.on("endpoint", (event) => this.emit("endpoint", event));
    inner.on("error", (event) => this.emit("error", event));
    inner.on("final", (event: FinalEvent) => {
      this.emit("final", { text: rewriteText(event.text, config) });
    });
  }

  pushAudio(frame: Float32Array): void {
    this.#inner.pushAudio(frame);
  }

  flush(): void {
    this.#inner.flush();
  }

  reset(): void {
    this.#inner.reset();
  }

  end(): Promise<void> {
    return this.#inner.end();
  }
}
