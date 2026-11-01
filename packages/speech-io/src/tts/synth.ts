import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import type { OfflineTtsConfig } from "sherpa-onnx-node";
import { createTtsFrontend, type TtsFrontend } from "./frontend.ts";
import {
  ensureTtsModel,
  resolveTtsModelPaths,
  getTtsModel,
  type SynthModelId,
  type TtsModelPaths,
  type TtsModel,
  TTS_ROOT,
} from "./models.ts";

/**
 * Thin OfflineTts wrapper for the TTS sweep.
 *
 * TTS is a bounded context separate from the STT VoiceToText contract: this
 * module talks to sherpa-onnx-node directly and shares nothing with
 * src/contract.ts.
 *
 * Text reaches phonemes through frontend.ts: sherpa-onnx 2.x has no built in
 * espeak-ng.
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
 * first-audio comes from sentence pipelining in stream.ts instead. The 2.x
 * callback path is untested here, so the rule stands.
 */

const require = createRequire(import.meta.url);
const loadSherpa = () => require("sherpa-onnx-node") as typeof import("sherpa-onnx-node");

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
   * Synthesis start to first audible chunk. Always undefined: the onProgress
   * path crashed the 1.13.3 binding (see module doc) and stays unused, so
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
  #frontend: TtsFrontend;
  readonly modelId: SynthModelId;
  readonly numThreads: number;
  readonly loadMs: number;

  private constructor(
    tts: import("sherpa-onnx-node").OfflineTts,
    frontend: TtsFrontend,
    modelId: SynthModelId,
    numThreads: number,
    loadMs: number,
  ) {
    this.#tts = tts;
    this.#frontend = frontend;
    this.modelId = modelId;
    this.numThreads = numThreads;
    this.loadMs = loadMs;
  }

  /** Verify browser payloads or install legacy assets, then load without warmup. */
  static async create(
    modelId: SynthModelId,
    numThreads: number = DEFAULT_NUM_THREADS,
    options: { root?: string; allowDownload?: boolean } = {},
  ): Promise<TtsSynth> {
    const model: TtsModel = getTtsModel(modelId);
    const root = options.root ?? TTS_ROOT;
    if (options.allowDownload === false && !model.files) throw new Error("Model requires explicit provisioning");
    const paths = resolveTtsModelPaths(model, root);
    await ensureTtsModel(paths);
    const start = performance.now();
    const tts = new (loadSherpa().OfflineTts)(
      createTtsConfig(model.family, paths, numThreads),
    );
    const loadMs = performance.now() - start;
    if (model.numSpeakers !== undefined && tts.numSpeakers !== model.numSpeakers) {
      throw new Error(`TTS speaker count mismatch for ${modelId}`);
    }
    if (model.sampleRate !== undefined && tts.sampleRate !== model.sampleRate) {
      throw new Error(`TTS sample rate mismatch for ${modelId}`);
    }
    const frontend = await createTtsFrontend({
      tokens: paths.tokens, lang: model.lang ?? tts.lang, family: model.family,
    }, root, options.allowDownload ?? true);
    return new TtsSynth(tts, frontend, modelId, numThreads, loadMs);
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
    validateSpeaker(request.sid ?? 0, this.numSpeakers);
    validateSpeed(request.speed ?? 1);
    const start = performance.now();
    // No onProgress: the 1.13.3 callback path aborts the process (module doc).
    const generate = this.#frontend.request(request.text, request.sid ?? 0, request.speed ?? 1);
    const audio = generate
      ? await this.#tts.generateAsync(generate)
      : { samples: new Float32Array(0), sampleRate: this.#tts.sampleRate };
    const totalSynthMs = performance.now() - start;
    const audioDurationMs = (audio.samples.length / audio.sampleRate) * 1_000;
    return {
      samples: audio.samples,
      sampleRate: audio.sampleRate,
      firstAudioMs: undefined,
      totalSynthMs,
      audioDurationMs,
      rtf: audioDurationMs > 0 ? totalSynthMs / audioDurationMs : 0,
    };
  }
}

/** Write synthesized samples to a 16-bit PCM wav file. */
export async function writeWav(
  path: string,
  audio: { samples: Float32Array; sampleRate: number },
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  loadSherpa().writeWave(path, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
}

export function createTtsConfig(
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
      },
      ...shared,
    },
  };
}

export function validateSpeaker(sid: number, numSpeakers: number): void {
  if (!Number.isInteger(sid) || sid < 0 || sid >= numSpeakers) {
    throw new Error(`Invalid TTS speaker ${sid}; model has ${numSpeakers} speakers`);
  }
}

export function validateSpeed(speed: number): void {
  if (!Number.isFinite(speed) || speed <= 0) throw new Error("TTS speed must be finite and positive");
}
