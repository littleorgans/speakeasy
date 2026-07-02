import { median } from "./stats.ts";
import type { WavAudio } from "./wav.ts";

/**
 * RMS-window speech-end detection: the bench-side reference for when speech
 * truly ends in the WAV, independent of any engine endpointing. Used as the
 * "endpoint" reference in sweep mode and as the push-to-talk release point
 * in ptt mode.
 */

const RMS_WINDOW_MS = 20;
const RMS_HANGOVER_MS = 650;
const RMS_PEAK_RATIO = 0.04;
const RMS_NOISE_MULTIPLIER = 8;
const RMS_MIN_THRESHOLD = 0.016;
const RMS_TAIL_NOISE_MS = 500;

export type SpeechProfile = {
  /** Loose reference: last voiced sample plus hangover (clamped to file end). */
  endMs: number;
  frameIndex: number;
  offsetWithinFrameMs: number;
  /** Strict reference: end of the last RMS window at or above threshold. */
  voicedEndMs: number;
  voicedFrameIndex: number;
  threshold: number;
  windowMs: number;
  hangoverMs: number;
};

export function detectSpeechProfile(
  audio: WavAudio,
  frameMs: number,
): SpeechProfile {
  const windowSamples = Math.round((audio.sampleRate * RMS_WINDOW_MS) / 1_000);
  const windows = buildRmsWindows(audio.samples, audio.sampleRate, windowSamples);
  const peakRms = Math.max(...windows.map((window) => window.rms));
  const tailNoise = median(
    windows
      .filter((window) => window.endMs >= audio.durationMs - RMS_TAIL_NOISE_MS)
      .map((window) => window.rms),
  );
  const threshold = Math.max(
    RMS_MIN_THRESHOLD,
    peakRms * RMS_PEAK_RATIO,
    tailNoise * RMS_NOISE_MULTIPLIER,
  );
  const lastSpeechWindow = windows.findLast((window) => window.rms >= threshold);
  if (!lastSpeechWindow) {
    throw new Error("WAV does not contain detectable speech");
  }

  const voicedEndMs = lastSpeechWindow.endMs;
  const endMs = Math.min(audio.durationMs, voicedEndMs + RMS_HANGOVER_MS);
  const loose = locateFrame(endMs, audio.sampleRate, frameMs);
  const strict = locateFrame(voicedEndMs, audio.sampleRate, frameMs);

  return {
    endMs,
    frameIndex: loose.frameIndex,
    offsetWithinFrameMs: loose.offsetWithinFrameMs,
    voicedEndMs,
    voicedFrameIndex: strict.frameIndex,
    threshold,
    windowMs: RMS_WINDOW_MS,
    hangoverMs: RMS_HANGOVER_MS,
  };
}

function locateFrame(
  ms: number,
  sampleRate: number,
  frameMs: number,
): { frameIndex: number; offsetWithinFrameMs: number } {
  const samplesPerFrame = Math.round((sampleRate * frameMs) / 1_000);
  const endSample = Math.round((ms / 1_000) * sampleRate);
  return {
    frameIndex: Math.floor(endSample / samplesPerFrame),
    offsetWithinFrameMs: ((endSample % samplesPerFrame) / sampleRate) * 1_000,
  };
}

function buildRmsWindows(
  samples: Float32Array,
  sampleRate: number,
  windowSamples: number,
): Array<{ endMs: number; rms: number }> {
  const windows: Array<{ endMs: number; rms: number }> = [];
  for (let start = 0; start < samples.length; start += windowSamples) {
    let sumSquares = 0;
    const end = Math.min(samples.length, start + windowSamples);
    for (let index = start; index < end; index += 1) {
      const sample = samples[index]!;
      sumSquares += sample * sample;
    }
    windows.push({
      endMs: (end / sampleRate) * 1_000,
      rms: Math.sqrt(sumSquares / Math.max(1, end - start)),
    });
  }
  return windows;
}
