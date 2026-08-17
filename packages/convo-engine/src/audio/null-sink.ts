import type { AudioSegment } from "@speakeasy/speech-io";
import type { AudioSink } from "../loop.ts";

/** Consume audio immediately while retaining its played duration. */
export function createNullSink(): AudioSink {
  let playedAudioMs = 0;

  return {
    open(): void {},
    write(segment: AudioSegment): void {
      playedAudioMs += segment.audioDurationMs;
    },
    async interrupt(): Promise<number> {
      return playedAudioMs;
    },
    async end(): Promise<void> {},
  };
}
