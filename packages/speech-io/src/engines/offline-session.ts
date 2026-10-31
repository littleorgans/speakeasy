import { EventEmitter } from "node:events";
import { performance } from "node:perf_hooks";
import type { EmptyEvent, FinalEvent, STTSession } from "../contract.ts";
import type { ResultDetail } from "./result.ts";
import { samplesToMs } from "./sherpa.ts";

/**
 * Push to talk session for whole utterance engines, independent of the
 * runtime behind `decode` (sherpa's OfflineRecognizer or onnxruntime graphs):
 * frames are buffered while held, up to `maxSamples` (later audio is
 * dropped), and decoded once on flush(); decodes run in flush order, and
 * flush() always answers with `final` or `empty`. An empty hold never
 * reaches `decode`, so no native decoder is handed an empty waveform.
 */

export type DecodeAudio = (audio: Float32Array) => Promise<ResultDetail>;

export class BufferedOfflineSession extends EventEmitter implements STTSession {
  readonly #maxSamples: number;
  readonly #decodeAudio: DecodeAudio;
  #frames: Float32Array[] = [];
  #samples = 0;
  #closed = false;
  #decoding: Promise<void> = Promise.resolve();

  constructor(maxSamples: number, decodeAudio: DecodeAudio) {
    super();
    this.#maxSamples = maxSamples;
    this.#decodeAudio = decodeAudio;
  }

  pushAudio(frame: Float32Array): void {
    if (this.#closed) {
      throw new Error("Cannot push audio after end()");
    }
    const room = this.#maxSamples - this.#samples;
    if (room <= 0) {
      return;
    }
    // Copy: hosts may reuse their frame buffers after the call returns.
    const kept = frame.slice(0, Math.min(room, frame.length));
    this.#frames.push(kept);
    this.#samples += kept.length;
  }

  flush(): void {
    if (this.#closed) {
      return;
    }
    this.#commit(true);
  }

  reset(): void {
    this.#frames = [];
    this.#samples = 0;
  }

  async end(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    this.emit("endpoint", {});
    if (this.#samples > 0) {
      this.#commit(false);
    }
    await this.#decoding;
  }

  #commit(explicit: boolean): void {
    const startedAt = performance.now();
    const audio = this.#take();
    this.#decoding = this.#decoding.then(() =>
      this.#decode(audio, startedAt, explicit),
    );
  }

  #take(): Float32Array {
    const audio = new Float32Array(this.#samples);
    let offset = 0;
    for (const frame of this.#frames) {
      audio.set(frame, offset);
      offset += frame.length;
    }
    this.reset();
    return audio;
  }

  async #decode(
    audio: Float32Array,
    startedAt: number,
    explicit: boolean,
  ): Promise<void> {
    try {
      const detail =
        audio.length === 0 ? { text: "", tokens: [] } : await this.#decodeAudio(audio);
      const timing = {
        audioMs: samplesToMs(audio.length),
        decodeMs: performance.now() - startedAt,
      };
      if (detail.text) {
        this.emit("final", { ...detail, ...timing } satisfies FinalEvent);
      } else if (explicit) {
        this.emit("empty", timing satisfies EmptyEvent);
      }
    } catch (err) {
      this.emit("error", { err });
    }
  }
}
