import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { Lazy, OfflineLifecycle } from "./offline-engine.ts";
import { prepareOfflineHotwords, snapshotHotwords, type OfflineHotwords } from "./offline-hotwords.ts";
import { resultDetail } from "./result.ts";
import {
  SHERPA_FEATURE_DIM,
  SHERPA_NUM_THREADS,
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
 * no partials and no endpoint detection. Parakeet Ultra lists are validated
 * against its export vocabulary and select modified beam search.
 */

export class SherpaOfflineEngine implements VoiceToText {
  readonly #asset: SherpaModelAsset<OfflineSherpaModel>;
  readonly #binding: SherpaBinding;
  readonly #hotwords: OfflineHotwords | undefined;
  /** Assets and hotword vocabulary, kept for the engine's life once they succeed. */
  readonly #preparation = new Lazy(() => this.#preflight());
  readonly #lifecycle: OfflineLifecycle<string | undefined, OfflineRecognizer>;

  constructor(options: SherpaEngineOptions) {
    const model = resolveSherpaModel(options.model);
    if (model.kind !== "offline") {
      throw new Error(
        `SherpaOfflineEngine needs an offline model; ${options.model} is streaming (use SherpaEngine)`,
      );
    }
    this.#asset = new SherpaModelAsset(model, options);
    this.#binding = options.binding;
    this.#hotwords = snapshotHotwords(options.hotwords);
    if (this.#hotwords && (model.family !== "nemo-transducer" || !model.hotwordTokenizer)) {
      throw new Error(`${model.name}: unsupported offline hotword tokenizer`);
    }
    // One recognizer serves every session; offline streams are per decode.
    this.#lifecycle = new OfflineLifecycle({
      name: "SherpaOfflineEngine",
      maxAudioMs: model.maxAudioMs,
      silenceGate: options.silenceGate,
      // Construction waits for open(): browser-voice times it apart from processing.
      prepare: "preflight",
      preflight: () => this.#preparation.get(),
      load: (vocabulary) => this.#createRecognizer(vocabulary),
      decoder: (recognizer) => async (audio) => {
        const stream = recognizer.createStream();
        stream.acceptWaveform({ samples: audio, sampleRate: SHERPA_SAMPLE_RATE });
        return resultDetail(await recognizer.decodeAsync(stream));
      },
    });
  }

  get label(): string {
    return this.#asset.label;
  }

  get model(): OfflineSherpaModel {
    return this.#asset.model;
  }

  get threads(): number {
    return SHERPA_NUM_THREADS;
  }

  prepare(): Promise<void> {
    return this.#lifecycle.prepare();
  }

  open(config?: STTConfig): Promise<STTSession> {
    return this.#lifecycle.open(config);
  }

  /**
   * Drop the recognizer once its decodes finish; a later open() constructs
   * another from the kept preparation. sherpa exposes no native free.
   */
  release(): Promise<void> {
    return this.#lifecycle.release();
  }

  async #preflight(): Promise<string | undefined> {
    await this.#asset.ensure();
    if (!this.#hotwords) return undefined;
    const model = this.model;
    if (model.family !== "nemo-transducer" || !model.hotwordTokenizer) {
      throw new Error(`${model.name}: unsupported offline hotword tokenizer`);
    }
    let table: string;
    try {
      table = new TextDecoder("utf-8", { fatal: true }).decode(
        await readFile(this.#asset.file(model.tokens)),
      );
    } catch {
      throw new Error(`${model.name}: cannot read UTF8 tokens table (${model.tokens})`);
    }
    return prepareOfflineHotwords(model, table, this.#hotwords).vocabulary;
  }

  #createRecognizer(vocabulary?: string): OfflineRecognizer {
    const config = createOfflineConfig(this.#asset);
    if (!this.#hotwords || vocabulary === undefined) {
      return new this.#binding.OfflineRecognizer(config);
    }
    const parent = tmpdir();
    mkdirSync(parent, { recursive: true });
    const dir = mkdtempSync(join(parent, "speak-easy-offline-hotwords-"));
    try {
      config.modelConfig.modelingUnit = "bpe";
      config.modelConfig.bpeVocab = join(dir, "bpe.vocab");
      config.hotwordsFile = join(dir, "hotwords.txt");
      config.hotwordsScore = this.#hotwords.score;
      config.maxActivePaths = 4;
      config.decodingMethod = "modified_beam_search";
      writeFileSync(config.modelConfig.bpeVocab, vocabulary, { mode: 0o600 });
      writeFileSync(config.hotwordsFile, `${this.#hotwords.phrases.join("\n")}\n`, { mode: 0o600 });
      // The native constructor reads both files synchronously into memory.
      return new this.#binding.OfflineRecognizer(config);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
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
      numThreads: SHERPA_NUM_THREADS,
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
