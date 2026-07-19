/**
 * Dead-simple energy VAD for barge-in: trip when mic frames stay above an
 * amplitude threshold for a sustained span. It only has to answer "did the user
 * start talking while the assistant was speaking", so peak amplitude over a few
 * consecutive frames is enough — no model, no spectral analysis.
 *
 * On open speakers this also hears the assistant's own voice (no echo
 * cancellation), so voice barge-in is meant for headphones; the threshold is a
 * coarse guard, not a substitute for AEC.
 */

export type EnergyVadOptions = {
  /** Peak amplitude (0..1) a frame must exceed to count as speech. */
  threshold?: number;
  /** Sustained speech duration before tripping. */
  minSpeechMs?: number;
  /** Frame duration, to convert minSpeechMs into a frame count. */
  frameMs?: number;
};

export const DEFAULT_VAD_THRESHOLD = 0.08;
export const DEFAULT_VAD_MIN_SPEECH_MS = 160;

export class EnergyVad {
  readonly #threshold: number;
  readonly #minFrames: number;
  #run = 0;

  constructor(options: EnergyVadOptions = {}) {
    this.#threshold = options.threshold ?? DEFAULT_VAD_THRESHOLD;
    const minSpeechMs = options.minSpeechMs ?? DEFAULT_VAD_MIN_SPEECH_MS;
    this.#minFrames = Math.max(1, Math.ceil(minSpeechMs / (options.frameMs ?? 20)));
  }

  /** Feed one frame; returns true the moment sustained speech is detected. */
  accept(frame: Float32Array): boolean {
    if (peak(frame) < this.#threshold) {
      this.#run = 0;
      return false;
    }
    this.#run += 1;
    if (this.#run >= this.#minFrames) {
      this.#run = 0;
      return true;
    }
    return false;
  }

  reset(): void {
    this.#run = 0;
  }
}

function peak(frame: Float32Array): number {
  let max = 0;
  for (const sample of frame) {
    const abs = Math.abs(sample);
    if (abs > max) {
      max = abs;
    }
  }
  return max;
}
