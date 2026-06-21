import { EventEmitter } from "node:events";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";

/**
 * StubEngine: a zero-cost engine used to validate the benchmark harness and the
 * contract end-to-end before a real model is wired in. It emits a partial part
 * way through the audio, then on end() simulates a tiny finalization compute,
 * fires "endpoint", and commits a placeholder "final".
 *
 * Replace with a real engine (Moonshine via onnxruntime-node, whisper.cpp +
 * Metal sidecar, ...) behind the same VoiceToText interface.
 */

const PARTIAL_AFTER_FRAMES = 10;
const SIMULATED_FINALIZE_MS = 5;

class StubSession extends EventEmitter implements STTSession {
  #frames = 0;
  #emittedPartial = false;
  #closed = false;

  pushAudio(_frame: Float32Array): void {
    if (this.#closed) {
      throw new Error("Cannot push audio after end()");
    }
    this.#frames += 1;
    if (!this.#emittedPartial && this.#frames >= PARTIAL_AFTER_FRAMES) {
      this.#emittedPartial = true;
      this.emit("partial", { text: "..." });
    }
  }

  flush(): void {
    this.emit("endpoint", {});
    this.emit("final", { text: "[stub transcript]" });
  }

  reset(): void {
    this.#frames = 0;
    this.#emittedPartial = false;
  }

  async end(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, SIMULATED_FINALIZE_MS));
    this.flush();
    this.#closed = true;
  }
}

export class StubEngine implements VoiceToText {
  async open(_config?: STTConfig): Promise<STTSession> {
    return new StubSession();
  }
}
