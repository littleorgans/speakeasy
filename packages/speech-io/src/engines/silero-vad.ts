import { join } from "node:path";
import { ensureAsset, type FetchArchive, type FileSetSpec } from "./assets.ts";
import type { SpeechDetector } from "./offline-session.ts";
import { SHERPA_SAMPLE_RATE } from "./sherpa.ts";
import type { ModelFile, ModelLicence } from "./sherpa-models.ts";

/**
 * The silence gate's speech detector: Silero VAD run through sherpa-onnx's
 * `Vad`, from the pinned single file sherpa-onnx publishes. It loads in tens
 * of milliseconds and scans a hold in a few milliseconds per second of audio.
 *
 * A hold holds speech when the VAD finds at least one segment of 100 ms or
 * more. sherpa's 250 ms default drops single short words ("Go.", "No.");
 * zeros and uniform noise up to RMS 0.03 yield no segment.
 */

/** sherpa's `Vad` and the config fields this detector sets. */
export type VadBinding = {
  Vad: new (config: VadConfig, bufferSizeInSeconds: number) => Vad;
};

export type VadConfig = {
  sileroVad: {
    model: string;
    threshold: number;
    minSilenceDuration: number;
    minSpeechDuration: number;
    windowSize: number;
    maxSpeechDuration: number;
  };
  sampleRate: number;
  numThreads: number;
  provider: string;
  debug: number;
};

export type Vad = {
  acceptWaveform(samples: Float32Array): void;
  flush(): void;
  isEmpty(): boolean;
  pop(): void;
  reset(): void;
};

export const SILERO_VAD = {
  name: "silero-vad",
  file: {
    name: "silero_vad.onnx",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6", // GitHub digest
    bytes: 643_854,
  } satisfies ModelFile,
  licence: {
    id: "MIT",
    terms:
      "MIT License: use, modify and redistribute, including commercially; keep the copyright and permission notice.",
    commercialUse: true,
    redistribution: true,
    attribution: "Silero VAD, Copyright (c) 2020-present Silero Team, MIT License.",
    source:
      "https://github.com/snakers4/silero-vad/blob/caddb3b7ce1dee88a14d5621a0e9a8fdeb2c2c48/LICENSE",
  } satisfies ModelLicence,
} as const;

/** Silero's window at 16 kHz. */
const WINDOW_SAMPLES = 512;
/** Covers the longest offline hold any registry model accepts. */
const BUFFER_SECONDS = 120;

export function sileroVadFiles(root: string): FileSetSpec {
  return { dir: join(root, SILERO_VAD.name), files: [SILERO_VAD.file] };
}

export async function loadSileroDetector(
  binding: VadBinding,
  root: string,
  fetchFile?: FetchArchive,
): Promise<SpeechDetector> {
  const files = sileroVadFiles(root);
  await ensureAsset(files, fetchFile);
  const vad = new binding.Vad(
    {
      sileroVad: {
        model: join(files.dir, SILERO_VAD.file.name),
        threshold: 0.5,
        minSilenceDuration: 0.25,
        minSpeechDuration: 0.1,
        windowSize: WINDOW_SAMPLES,
        maxSpeechDuration: BUFFER_SECONDS,
      },
      sampleRate: SHERPA_SAMPLE_RATE,
      numThreads: 1,
      provider: "cpu",
      debug: 0,
    },
    BUFFER_SECONDS,
  );
  // Synchronous end to end, so sessions may share the one detector.
  return (audio) => {
    vad.reset();
    for (let start = 0; start + WINDOW_SAMPLES <= audio.length; start += WINDOW_SAMPLES) {
      vad.acceptWaveform(audio.subarray(start, start + WINDOW_SAMPLES));
    }
    vad.flush();
    const speech = !vad.isEmpty();
    while (!vad.isEmpty()) {
      vad.pop();
    }
    return speech;
  };
}
