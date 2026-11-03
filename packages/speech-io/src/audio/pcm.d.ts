export const SPEECH_SAMPLE_RATE: 16_000;

export function concatFrames(frames: readonly Float32Array[], samples?: number): Float32Array;

/** Throws unless `config.sampleRate` is unset or SPEECH_SAMPLE_RATE. */
export function assertSampleRate(engine: string, config?: { sampleRate?: number }): void;

/** Duration in milliseconds of `samples` at SPEECH_SAMPLE_RATE. */
export function samplesToMs(samples: number): number;
