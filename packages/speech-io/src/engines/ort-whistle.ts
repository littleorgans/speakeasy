import { join } from "node:path";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { ensureFileAsset, type FileAssetSpec } from "./file-assets.ts";
import { BufferedOfflineSession } from "./offline-session.ts";
import { resultText, type ResultDetail } from "./result.ts";
import { assertSampleRate, SHERPA_SAMPLE_RATE } from "./sherpa.ts";
import {
  resolveWhistleFiles,
  resolveWhistleModel,
  type WhistleModel,
  type WhistleModelId,
} from "./ort-whistle-models.ts";
import type { TranscribeResult } from "./vendor/whistle/whistle.js";

/**
 * OrtWhistleEngine: push to talk STT on Cactus Whistle, an encoder decoder
 * model run on onnxruntime-node through the vendored reference pipeline
 * (`vendor/whistle/whistle.js::transcribe`: log mel, engram, greedy decode,
 * DTW word timestamps).
 *
 * Sessions buffer while held and decode on flush (`BufferedOfflineSession`).
 * The runtime is injected, as sherpa's binding is, so this module loads no
 * native code; `ort-whistle-runtime.ts` supplies the real one. One recognizer
 * serves every session and holds decoder state, so decodes are serialized.
 */

/** Paths of the model's files on disk. */
export type WhistleFiles = {
  encoder: string;
  decoder: string;
  pack: string;
};

export type WhistleRecognizer = {
  transcribe(pcm: Float32Array, language: string | undefined): Promise<TranscribeResult>;
};

export type WhistleRuntime = {
  version: string;
  createRecognizer(files: WhistleFiles): Promise<WhistleRecognizer>;
};

export type OrtWhistleEngineOptions = {
  model: WhistleModelId;
  /** Folder holding downloaded models. */
  modelRoot: string;
  runtime: WhistleRuntime;
};

export class OrtWhistleEngine implements VoiceToText {
  readonly model: WhistleModel;
  readonly paths: FileAssetSpec;
  readonly #runtime: WhistleRuntime;
  #recognizer: Promise<WhistleRecognizer> | undefined;
  #decoding: Promise<unknown> = Promise.resolve();

  constructor(options: OrtWhistleEngineOptions) {
    this.model = resolveWhistleModel(options.model);
    this.paths = resolveWhistleFiles(this.model, options.modelRoot);
    this.#runtime = options.runtime;
  }

  get label(): string {
    return `onnxruntime-node-${this.#runtime.version}:${this.model.name}`;
  }

  async prepare(): Promise<void> {
    await ensureFileAsset(this.paths);
  }

  async open(config?: STTConfig): Promise<STTSession> {
    assertSampleRate("OrtWhistleEngine", config);
    const recognizer = await this.#load();
    const language = supportedLanguage(this.model, config?.language);
    const maxSamples = (this.model.maxAudioMs * SHERPA_SAMPLE_RATE) / 1_000;
    return new BufferedOfflineSession(maxSamples, (audio) =>
      this.#serialized(async () =>
        whistleDetail(await recognizer.transcribe(audio, language)),
      ),
    );
  }

  #load(): Promise<WhistleRecognizer> {
    this.#recognizer ??= this.prepare().then(() =>
      this.#runtime.createRecognizer({
        encoder: this.#file(this.model.encoder),
        decoder: this.#file(this.model.decoder),
        pack: this.#file(this.model.pack),
      }),
    );
    // A failed load is retried on the next open.
    this.#recognizer.catch(() => {
      this.#recognizer = undefined;
    });
    return this.#recognizer;
  }

  #file(name: string): string {
    return join(this.paths.dir, name);
  }

  #serialized<T>(work: () => Promise<T>): Promise<T> {
    const result = this.#decoding.then(work, work);
    this.#decoding = result.catch(() => undefined);
    return result;
  }
}

/** A BCP 47 hint's primary subtag when the model speaks it, else language ID. */
function supportedLanguage(
  model: WhistleModel,
  hint: string | undefined,
): string | undefined {
  const primary = hint?.split("-")[0]?.toLowerCase();
  return primary && model.languages.includes(primary) ? primary : undefined;
}

/**
 * Whistle reports words, not tokens: each word becomes one token with a
 * leading space (a word start), its DTW start time, and the log of the
 * word's mean token probability (null when it rounds to 0).
 */
export function whistleDetail(result: TranscribeResult): ResultDetail {
  const text = resultText({ text: result.text });
  if (!text) {
    return { text, tokens: [] };
  }
  const timed = result.words.every((word) => word.start !== undefined);
  return {
    text,
    tokens: timed
      ? result.words.map((word) => ({
          text: ` ${word.word}`,
          startMs: Math.round((word.start ?? 0) * 1_000),
          logProb: word.probability > 0 ? Math.log(word.probability) : null,
        }))
      : [],
  };
}
