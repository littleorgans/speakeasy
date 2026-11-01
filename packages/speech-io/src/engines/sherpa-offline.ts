import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { prepareOfflineHotwords, snapshotHotwords, type OfflineHotwords } from "./offline-hotwords.ts";
import { BufferedOfflineSession, type SilenceGate } from "./offline-session.ts";
import { resultDetail } from "./result.ts";
import {
  assertSampleRate,
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
  readonly #silenceGate: SilenceGate | undefined;
  readonly #hotwords: OfflineHotwords | undefined;
  #preparation: Promise<string | undefined> | undefined;
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
    this.#silenceGate = options.silenceGate;
    this.#hotwords = snapshotHotwords(options.hotwords);
    if (this.#hotwords && (model.family !== "nemo-transducer" || !model.hotwordTokenizer)) {
      throw new Error(`${model.name}: unsupported offline hotword tokenizer`);
    }
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

  async prepare(): Promise<void> {
    await this.#prepare();
  }

  async open(config?: STTConfig): Promise<STTSession> {
    assertSampleRate("SherpaOfflineEngine", config);
    const vocabulary = await this.#prepare();
    // One recognizer serves every session; offline streams are per decode.
    this.#recognizer ??= this.#createRecognizer(vocabulary);
    const recognizer = this.#recognizer;
    const maxSamples = (this.model.maxAudioMs * SHERPA_SAMPLE_RATE) / 1_000;
    return new BufferedOfflineSession(
      maxSamples,
      async (audio) => {
        const stream = recognizer.createStream();
        stream.acceptWaveform({ samples: audio, sampleRate: SHERPA_SAMPLE_RATE });
        return resultDetail(await recognizer.decodeAsync(stream));
      },
      this.#silenceGate,
    );
  }

  #prepare(): Promise<string | undefined> {
    this.#preparation ??= this.#preflight().catch((error: unknown) => {
      this.#preparation = undefined;
      throw error;
    });
    return this.#preparation;
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
