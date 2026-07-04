import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import type { OfflineTtsConfig } from "sherpa-onnx-node";
import {
  ensureTtsModel,
  resolveTtsModelPaths,
  TTS_MODELS,
  type TtsModelId,
  type TtsModelPaths,
} from "./models.ts";

/**
 * Thin OfflineTts wrapper for the TTS sweep.
 *
 * TTS is a bounded context separate from the STT VoiceToText contract: this
 * module talks to sherpa-onnx-node directly and shares nothing with
 * src/contract.ts.
 *
 * Streaming verdict for the 1.13.3 binding: generateAsync exposes an
 * onProgress callback (one call per maxNumSentences batch, default one per
 * sentence), but it is unusable. Root cause (addon source,
 * non-streaming-tts.cc, identical on master): TtsGenerateWorker's destructor
 * deletes every queued TtsCallbackData as soon as the AsyncWorker completes,
 * racing the TSFN drain on the main loop; a late delivery reads the freed
 * struct, gets a garbage samples.size(), and napi_create_arraybuffer aborts
 * the process (v8::ArrayBuffer::New OOM). Measured crash rate with any
 * callback: 2/5 to 5/5 per run shape; copying samples, enableExternalBuffer,
 * and forced GC change nothing (see TTS-STREAMING.md). So no onProgress here;
 * first-audio comes from sentence pipelining in stream.ts instead.
 */

const require = createRequire(import.meta.url);
const sherpa = require("sherpa-onnx-node") as typeof import("sherpa-onnx-node");

const DEFAULT_NUM_THREADS = 2;

export type SynthRequest = {
  text: string;
  /** Speaker id; 0 for single-voice models. */
  sid?: number;
  /** Speaking rate multiplier; 1 is the model's native pace. */
  speed?: number;
};

export type SynthResult = {
  samples: Float32Array;
  sampleRate: number;
  /**
   * Synthesis start to first audible chunk. Always undefined on the 1.13.3
   * binding: the onProgress path crashes the process (see module doc), so
   * there is no safe first-audio signal to observe.
   */
  firstAudioMs: number | undefined;
  totalSynthMs: number;
  audioDurationMs: number;
  /** Real-time factor: synth time / audio time; below 1 is faster than realtime. */
  rtf: number;
};

export class TtsSynth {
  #tts: import("sherpa-onnx-node").OfflineTts;
  readonly modelId: TtsModelId;
  readonly numThreads: number;
  readonly loadMs: number;

  private constructor(
    tts: import("sherpa-onnx-node").OfflineTts,
    modelId: TtsModelId,
    numThreads: number,
    loadMs: number,
  ) {
    this.#tts = tts;
    this.modelId = modelId;
    this.numThreads = numThreads;
    this.loadMs = loadMs;
  }

  /** Download the model if needed, then load it into an OfflineTts. */
  static async create(
    modelId: TtsModelId,
    numThreads: number = DEFAULT_NUM_THREADS,
  ): Promise<TtsSynth> {
    const paths = resolveTtsModelPaths(TTS_MODELS[modelId]);
    await ensureTtsModel(paths);
    const start = performance.now();
    const tts = new sherpa.OfflineTts(
      createTtsConfig(TTS_MODELS[modelId].family, paths, numThreads),
    );
    return new TtsSynth(tts, modelId, numThreads, performance.now() - start);
  }

  get numSpeakers(): number {
    return this.#tts.numSpeakers;
  }

  get sampleRate(): number {
    return this.#tts.sampleRate;
  }

  /**
   * Prime the onnxruntime session with one throwaway synth so the first real
   * request runs warm. Cheap and idempotent; call once at startup, off the
   * request path (in production the resident model is warmed during app boot).
   * Returns its own synth time so callers can attribute the startup cost.
   */
  async warmup(text = "Warm up."): Promise<number> {
    const { totalSynthMs } = await this.synth({ text });
    return totalSynthMs;
  }

  async synth(request: SynthRequest): Promise<SynthResult> {
    const start = performance.now();
    // No onProgress: the 1.13.3 callback path aborts the process (module doc).
    const audio = await this.#tts.generateAsync({
      text: request.text,
      sid: request.sid ?? 0,
      speed: request.speed ?? 1,
    });
    const totalSynthMs = performance.now() - start;
    const audioDurationMs = (audio.samples.length / audio.sampleRate) * 1_000;
    return {
      samples: audio.samples,
      sampleRate: audio.sampleRate,
      firstAudioMs: undefined,
      totalSynthMs,
      audioDurationMs,
      rtf: totalSynthMs / audioDurationMs,
    };
  }
}

/** Write synthesized samples to a 16-bit PCM wav file. */
export async function writeWav(
  path: string,
  audio: { samples: Float32Array; sampleRate: number },
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  sherpa.writeWave(path, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
}

function createTtsConfig(
  family: "vits" | "kokoro",
  paths: TtsModelPaths,
  numThreads: number = DEFAULT_NUM_THREADS,
): OfflineTtsConfig {
  const shared = {
    numThreads,
    provider: "cpu",
    debug: 0,
  };
  if (family === "kokoro") {
    return {
      model: {
        kokoro: {
          model: paths.model,
          voices: paths.voices,
          tokens: paths.tokens,
          dataDir: paths.dataDir,
        },
        ...shared,
      },
    };
  }
  return {
    model: {
      vits: {
        model: paths.model,
        tokens: paths.tokens,
        dataDir: paths.dataDir,
      },
      ...shared,
    },
  };
}
