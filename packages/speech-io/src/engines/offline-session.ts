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
 *
 * An optional silence gate checks the hold for speech before decode, so a
 * model that answers silence with words ("Thank you.") emits `empty`
 * instead; a log probability floor then empties a final the model itself
 * scores as unlikely. Models opt in on their descriptor (`silenceGate`); the
 * host supplies the detector, which keeps this module free of native code.
 */

export type DecodeAudio = (audio: Float32Array) => Promise<ResultDetail>;

/** Whether a held utterance contains speech. */
export type SpeechDetector = (audio: Float32Array) => boolean;

/**
 * A descriptor's opt in to the silence gate. `logProbFloor`, for models that
 * score tokens, empties a final whose mean token log probability is below it.
 */
export type SilenceGatePolicy = { logProbFloor: number | null };

/** The gate a session runs: the host's detector plus the model's floor. */
export type SilenceGate = SilenceGatePolicy & { hasSpeech: SpeechDetector };

/** "model" follows the descriptor; "on" and "off" override it, as the bench does. */
export type SilenceGateMode = "model" | "on" | "off";

/** Whether a model with `policy` gates under `mode`. */
export function gatesSilence(
  policy: SilenceGatePolicy | undefined,
  mode: SilenceGateMode,
): boolean {
  return mode === "on" || (mode === "model" && policy !== undefined);
}

const EMPTY: ResultDetail = { text: "", tokens: [] };

export class BufferedOfflineSession extends EventEmitter implements STTSession {
  readonly #maxSamples: number;
  readonly #decodeAudio: DecodeAudio;
  readonly #gate: SilenceGate | undefined;
  #frames: Float32Array[] = [];
  #samples = 0;
  #closed = false;
  #decoding: Promise<void> = Promise.resolve();

  constructor(maxSamples: number, decodeAudio: DecodeAudio, gate?: SilenceGate) {
    super();
    this.#maxSamples = maxSamples;
    this.#decodeAudio = decodeAudio;
    this.#gate = gate;
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
      const detail = await this.#gatedDecode(audio);
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

  async #gatedDecode(audio: Float32Array): Promise<ResultDetail> {
    const gate = this.#gate;
    if (audio.length === 0 || (gate && !gate.hasSpeech(audio))) {
      return EMPTY;
    }
    const detail = await this.#decodeAudio(audio);
    if (gate && gate.logProbFloor !== null && belowFloor(detail, gate.logProbFloor)) {
      return EMPTY;
    }
    return detail;
  }
}

/** Mean log probability of the scored tokens under `floor`; unscored never is. */
function belowFloor({ tokens }: ResultDetail, floor: number): boolean {
  const scored = tokens.flatMap((token) => (token.logProb === null ? [] : [token.logProb]));
  return scored.length > 0 && scored.reduce((sum, value) => sum + value, 0) / scored.length < floor;
}
