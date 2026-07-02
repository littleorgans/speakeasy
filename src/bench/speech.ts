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
  endMs: number;
  frameIndex: number;
  offsetWithinFrameMs: number;
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

  const endMs = Math.min(
    audio.durationMs,
    lastSpeechWindow.endMs + RMS_HANGOVER_MS,
  );
  const samplesPerFrame = Math.round((audio.sampleRate * frameMs) / 1_000);
  const endSample = Math.round((endMs / 1_000) * audio.sampleRate);
  const frameIndex = Math.floor(endSample / samplesPerFrame);
  const sampleOffset = endSample % samplesPerFrame;
  const offsetWithinFrameMs = (sampleOffset / audio.sampleRate) * 1_000;

  return {
    endMs,
    frameIndex,
    offsetWithinFrameMs,
    threshold,
    windowMs: RMS_WINDOW_MS,
    hangoverMs: RMS_HANGOVER_MS,
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
