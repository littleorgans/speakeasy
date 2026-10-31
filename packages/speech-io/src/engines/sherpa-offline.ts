import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { BufferedOfflineSession } from "./offline-session.ts";
import { resultDetail } from "./result.ts";
import {
  assertSampleRate,
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
 * whole utterance models (Parakeet TDT, Moonshine, Canary, Qwen3 ASR).
 *
 * Sessions are `BufferedOfflineSession`s: frames buffer while the key is
 * held, up to the model's `maxAudioMs`, and decode once on flush() with
 * `decodeAsync`, so the host's event loop stays free during decode. There are
 * no partials and no endpoint detection. Hotwords do not apply.
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
    const recognizer = this.#recognizer;
    const maxSamples = (this.model.maxAudioMs * SHERPA_SAMPLE_RATE) / 1_000;
    return new BufferedOfflineSession(maxSamples, async (audio) => {
      const stream = recognizer.createStream();
      stream.acceptWaveform({ samples: audio, sampleRate: SHERPA_SAMPLE_RATE });
      return resultDetail(await recognizer.decodeAsync(stream));
    });
  }
}

function createOfflineConfig(
  asset: SherpaModelAsset<OfflineSherpaModel>,
): OfflineRecognizerConfig {
  return {
    featConfig: {
      sampleRate: SHERPA_SAMPLE_RATE,
      featureDim: SHERPA_FEATURE_DIM,
    },
    modelConfig: {
      tokens: "tokens" in asset.model ? asset.file(asset.model.tokens) : "",
      numThreads: 2,
      provider: "cpu",
      debug: 0,
      ...familyConfig(asset),
    },
    decodingMethod: "greedy_search",
  };
}

function familyConfig(
  asset: SherpaModelAsset<OfflineSherpaModel>,
): Partial<OfflineRecognizerConfig["modelConfig"]> {
  const { model } = asset;
  switch (model.family) {
    case "nemo-transducer":
      return {
        modelType: "nemo_transducer",
        transducer: {
          encoder: asset.file(model.encoder),
          decoder: asset.file(model.decoder),
          joiner: asset.file(model.joiner),
        },
      };
    case "moonshine":
      return {
        moonshine: {
          preprocessor: asset.file(model.preprocessor),
          encoder: asset.file(model.encoder),
          uncachedDecoder: asset.file(model.uncachedDecoder),
          cachedDecoder: asset.file(model.cachedDecoder),
        },
      };
    case "canary":
      // Transcription, not translation: the same language in and out.
      return {
        canary: {
          encoder: asset.file(model.encoder),
          decoder: asset.file(model.decoder),
          srcLang: model.language,
          tgtLang: model.language,
          usePnc: 1,
        },
      };
    case "qwen3-asr":
      return {
        qwen3Asr: {
          convFrontend: asset.file(model.convFrontend),
          encoder: asset.file(model.encoder),
          decoder: asset.file(model.decoder),
          tokenizer: asset.file(model.tokenizer),
          hotwords: "",
          maxTotalLen: model.maxTotalLen,
          maxNewTokens: model.maxNewTokens,
        },
      };
  }
}
