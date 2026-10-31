import { EventEmitter } from "node:events";
import { performance } from "node:perf_hooks";
import type {
  EmptyEvent,
  FinalEvent,
  STTConfig,
  STTSession,
  VoiceToText,
} from "../contract.ts";
import { resultDetail } from "./result.ts";
import {
  assertSampleRate,
  samplesToMs,
  SHERPA_FEATURE_DIM,
  SHERPA_SAMPLE_RATE,
  SherpaModelAsset,
  type SherpaEngineOptions,
} from "./sherpa.ts";
import type {
  OfflineRecognizer,
  OfflineRecognizerConfig,
  SherpaBinding,
} from "./sherpa-api.ts";
import { resolveSherpaModel, type OfflineSherpaModel } from "./sherpa-models.ts";

/**
 * SherpaOfflineEngine: push to talk STT on sherpa's OfflineRecognizer for
 * whole utterance models (Parakeet TDT, Moonshine).
 *
 * Frames are buffered while the key is held, up to the model's `maxAudioMs`
 * (later audio is dropped), and decoded once on flush() with `decodeAsync`,
 * so the host's event loop stays free during decode. There are no partials
 * and no endpoint detection: flush() always answers with `final` or `empty`,
 * in flush order. Hotwords do not apply.
 */

export class SherpaOfflineEngine implements VoiceToText {
  readonly #asset: SherpaModelAsset<OfflineSherpaModel>;
  readonly #binding: SherpaBinding;
  #recognizer: OfflineRecognizer | undefined;

  constructor(options: SherpaEngineOptions) {
    const model = resolveSherpaModel(options.model);
    if (model.kind !== "offline") {
      throw new Error(
        `SherpaOfflineEngine needs an offline model; ${options.model} is streaming (use SherpaEngine)`,
      );
    }
    this.#asset = new SherpaModelAsset(model, options);
    this.#binding = options.binding;
  }

  get label(): string {
    return this.#asset.label;
  }

  get model(): OfflineSherpaModel {
    return this.#asset.model;
  }

  async prepare(): Promise<void> {
    await this.#asset.ensure();
  }

  async open(config?: STTConfig): Promise<STTSession> {
    assertSampleRate("SherpaOfflineEngine", config);
    await this.#asset.ensure();
    // One recognizer serves every session; offline streams are per decode.
    this.#recognizer ??= new this.#binding.OfflineRecognizer(
      createOfflineConfig(this.#asset),
    );
    const maxSamples = (this.model.maxAudioMs * SHERPA_SAMPLE_RATE) / 1_000;
    return new SherpaOfflineSession(this.#recognizer, maxSamples);
  }
}

class SherpaOfflineSession extends EventEmitter implements STTSession {
  readonly #recognizer: OfflineRecognizer;
  readonly #maxSamples: number;
  #frames: Float32Array[] = [];
  #samples = 0;
  #closed = false;
  #decoding: Promise<void> = Promise.resolve();

  constructor(recognizer: OfflineRecognizer, maxSamples: number) {
    super();
    this.#recognizer = recognizer;
    this.#maxSamples = maxSamples;
  }

  pushAudio(frame: Float32Array): void {
    if (this.#closed) {
      throw new Error("Cannot push audio after end()");
    }
    const room = this.#maxSamples - this.#samples;
    if (room <= 0) {
      return;
    }
    // Copy: hosts may reuse their frame buffers after the call returns.
    const kept = frame.slice(0, Math.min(room, frame.length));
    this.#frames.push(kept);
    this.#samples += kept.length;
  }

  flush(): void {
    if (this.#closed) {
      return;
    }
    this.#commit(true);
  }

  reset(): void {
    this.#frames = [];
    this.#samples = 0;
  }

  async end(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    this.emit("endpoint", {});
    if (this.#samples > 0) {
      this.#commit(false);
    }
    await this.#decoding;
  }

  #commit(explicit: boolean): void {
    const startedAt = performance.now();
    const audio = this.#take();
    this.#decoding = this.#decoding.then(() =>
      this.#decode(audio, startedAt, explicit),
    );
  }

  #take(): Float32Array {
    const audio = new Float32Array(this.#samples);
    let offset = 0;
    for (const frame of this.#frames) {
      audio.set(frame, offset);
      offset += frame.length;
    }
    this.reset();
    return audio;
  }

  async #decode(
    audio: Float32Array,
    startedAt: number,
    explicit: boolean,
  ): Promise<void> {
    try {
      // Never hand the native decoder an empty waveform: a hold with no audio
      // is an empty utterance by definition.
      const detail =
        audio.length === 0
          ? { text: "", tokens: [] }
          : resultDetail(await this.#decodeAudio(audio));
      const timing = {
        audioMs: samplesToMs(audio.length),
        decodeMs: performance.now() - startedAt,
      };
      if (detail.text) {
        this.emit("final", { ...detail, ...timing } satisfies FinalEvent);
      } else if (explicit) {
        this.emit("empty", timing satisfies EmptyEvent);
      }
    } catch (err) {
      this.emit("error", { err });
    }
  }

  async #decodeAudio(audio: Float32Array) {
    const stream = this.#recognizer.createStream();
    stream.acceptWaveform({ samples: audio, sampleRate: SHERPA_SAMPLE_RATE });
    return this.#recognizer.decodeAsync(stream);
  }
}

function createOfflineConfig(
  asset: SherpaModelAsset<OfflineSherpaModel>,
): OfflineRecognizerConfig {
  const { model } = asset;
  const shared = {
    tokens: asset.file(model.tokens),
    numThreads: 2,
    provider: "cpu",
    debug: 0,
  };
  const modelConfig =
    model.family === "nemo-transducer"
      ? {
          ...shared,
          modelType: "nemo_transducer",
          transducer: {
            encoder: asset.file(model.encoder),
            decoder: asset.file(model.decoder),
            joiner: asset.file(model.joiner),
          },
        }
      : {
          ...shared,
          moonshine: {
            preprocessor: asset.file(model.preprocessor),
            encoder: asset.file(model.encoder),
            uncachedDecoder: asset.file(model.uncachedDecoder),
            cachedDecoder: asset.file(model.cachedDecoder),
          },
        };
  return {
    featConfig: {
      sampleRate: SHERPA_SAMPLE_RATE,
      featureDim: SHERPA_FEATURE_DIM,
    },
    modelConfig,
    decodingMethod: "greedy_search",
  };
}
