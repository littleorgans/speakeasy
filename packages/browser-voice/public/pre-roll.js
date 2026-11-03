const SAMPLE_RATE = 16_000;
export const PRE_ROLL_MS = 300;
export const PRE_ROLL_SAMPLES = (SAMPLE_RATE * PRE_ROLL_MS) / 1_000;

/**
 * Idle capture frames, the last PRE_ROLL_MS of them, that lead the next
 * hold take so a word spoken at the press is kept. Shared by the room and
 * the prompt reader.
 */
export class PreRollRing {
  /** rate: the capture's sample rate, 16 kHz unless the caller records at another. */
  constructor(rate = SAMPLE_RATE) {
    this.limit = (rate * PRE_ROLL_MS) / 1_000;
    this.clear();
  }

  clear() {
    this.frames = [];
    this.samples = 0;
  }

  /** Keeps the last PRE_ROLL_MS of idle audio, in whole frames. */
  push(samples) {
    this.frames.push(samples);
    this.samples += samples.length;
    while (this.samples - this.frames[0].length >= this.limit) {
      this.samples -= this.frames.shift().length;
    }
  }

  /** Returns the last PRE_ROLL_MS of samples and empties the ring. */
  take() {
    const preRoll = concat(this.frames).slice(-this.limit);
    this.clear();
    return preRoll;
  }
}

export function concat(frames) {
  const samples = new Float32Array(frames.reduce((sum, frame) => sum + frame.length, 0));
  let offset = 0;
  for (const frame of frames) {
    samples.set(frame, offset);
    offset += frame.length;
  }
  return samples;
}
