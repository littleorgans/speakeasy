export const PRE_ROLL_MS: number;
export const PRE_ROLL_SAMPLES: number;

export class PreRollRing {
  clear(): void;
  push(samples: Float32Array): void;
  take(): Float32Array;
}

export function concat(frames: readonly Float32Array[]): Float32Array;
