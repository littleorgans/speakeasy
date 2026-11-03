export const FRAME_MS = 20;
/** Consecutive loud frames that count as the owner's voice. */
export const ONSET_FRAMES = 3;
/** Quiet time that ends a cut in. */
export const END_SILENCE_MS = 700;
/** RMS floor below which nothing counts as voice, about -36 dBFS. */
const MIN_LEVEL = 0.016;
const NOISE_RATIO = 3;
/** The quiet threshold sits below the loud one, so a word's tail does not end a cut in. */
const HYSTERESIS = 0.6;

/**
 * Voice detection over capture frame levels: an RMS window with hangover
 * against a tracked noise floor, never a single raw amplitude. Times are
 * capture context seconds at each frame's last sample.
 */
export class TalkOverDetector {
  constructor() {
    this.reset();
  }

  reset() {
    this.floor = null;
    this.loud = [];
    this.speaking = false;
    this.quietSince = null;
  }

  threshold() {
    return Math.max(MIN_LEVEL, this.floor * NOISE_RATIO);
  }

  /**
   * Feeds one frame. Returns { type: "onset", onsetTime } when the window
   * fills, { type: "end", endTime } after END_SILENCE_MS of quiet, else null.
   * onsetTime is the start of the first loud frame.
   */
  push(level, audioTime) {
    if (!Number.isFinite(level) || !Number.isFinite(audioTime)) return null;
    if (this.floor === null) {
      // The first frame sets the floor, so steady room noise never starts a cut in.
      this.floor = level;
      return null;
    }
    if (!this.speaking) {
      if (level > this.threshold()) {
        this.loud.push(audioTime);
        if (this.loud.length < ONSET_FRAMES) return null;
        this.speaking = true;
        this.quietSince = null;
        const onsetTime = this.loud[0] - FRAME_MS / 1_000;
        this.loud = [];
        return { type: "onset", onsetTime };
      }
      this.loud = [];
      // The floor follows quiet frames only, slowly, so speech cannot raise it.
      this.floor = this.floor * 0.95 + level * 0.05;
      return null;
    }
    if (level > this.threshold() * HYSTERESIS) {
      this.quietSince = null;
      return null;
    }
    this.quietSince ??= audioTime - FRAME_MS / 1_000;
    if ((audioTime - this.quietSince) * 1_000 < END_SILENCE_MS) return null;
    this.speaking = false;
    const endTime = this.quietSince;
    this.quietSince = null;
    return { type: "end", endTime };
  }
}
