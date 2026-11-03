export const FRAME_MS: number;
export const ONSET_FRAMES: number;
export const END_SILENCE_MS: number;

export type TalkOverEvent = { type: "onset"; onsetTime: number } | { type: "end"; endTime: number };

export class TalkOverDetector {
  floor: number | null;
  reset(): void;
  threshold(): number;
  push(level: number, audioTime: number): TalkOverEvent | null;
}
