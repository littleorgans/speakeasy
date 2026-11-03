/**
 * Browser safe mono Float32 PCM helpers shared by the engines, the corpus
 * store and (later) the browser pages: frame concatenation and the 16 kHz
 * speech sample rate rules. Plain JS with `pcm.d.ts` types so a browser can
 * load it unbundled.
 */

/** The rate every speech engine takes. */
export const SPEECH_SAMPLE_RATE = 16_000;

/**
 * Concatenates frames into one new buffer. `samples` presizes it when the
 * caller already tracks the total.
 */
export function concatFrames(
  frames,
  samples = frames.reduce((sum, frame) => sum + frame.length, 0),
) {
  const audio = new Float32Array(samples);
  let offset = 0;
  for (const frame of frames) {
    audio.set(frame, offset);
    offset += frame.length;
  }
  return audio;
}

export function assertSampleRate(engine, config) {
  if (config?.sampleRate && config.sampleRate !== SPEECH_SAMPLE_RATE) {
    throw new Error(
      `${engine} expects ${SPEECH_SAMPLE_RATE} Hz frames, received ${config.sampleRate} Hz`,
    );
  }
}

export function samplesToMs(samples) {
  return (samples * 1_000) / SPEECH_SAMPLE_RATE;
}
