import { join } from "node:path";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { ensureAsset, type FileSetSpec } from "./assets.ts";
import { OfflineLifecycle } from "./offline-engine.ts";
import type { SilenceGate } from "./offline-session.ts";
import { resultText, type ResultDetail } from "./result.ts";
import { resolveModelPaths } from "./model.ts";
import {
  resolveWhistleModel,
  type WhistleModel,
  type WhistleModelId,
} from "./ort-whistle-models.ts";
import { ortThreads } from "./ort-models.ts";
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
  /** Free the native sessions. */
  release(): Promise<void>;
};

export type WhistleRuntime = {
  version: string;
  createRecognizer(
    files: WhistleFiles,
    options: { threads: number },
  ): Promise<WhistleRecognizer>;
};

export type OrtWhistleEngineOptions = {
  model: WhistleModelId;
  /** Folder holding downloaded models. */
  modelRoot: string;
  runtime: WhistleRuntime;
  /** The silence gate every session runs before decode. */
  silenceGate?: SilenceGate;
};

export class OrtWhistleEngine implements VoiceToText {
  readonly model: WhistleModel;
  readonly paths: FileSetSpec;
  readonly #runtime: WhistleRuntime;
  readonly #lifecycle: OfflineLifecycle<void, WhistleRecognizer>;

  constructor(options: OrtWhistleEngineOptions) {
    this.model = resolveWhistleModel(options.model);
    this.paths = resolveModelPaths(this.model, options.modelRoot);
    this.#runtime = options.runtime;
    // A failed load is retried on the next open.
    this.#lifecycle = new OfflineLifecycle({
      name: "OrtWhistleEngine",
      maxAudioMs: this.model.maxAudioMs,
      silenceGate: options.silenceGate,
      prepare: "preflight",
      serial: true,
      preflight: () => ensureAsset(this.paths),
      load: () =>
        this.#runtime.createRecognizer(
          {
            encoder: this.#file(this.model.encoder),
            decoder: this.#file(this.model.decoder),
            pack: this.#file(this.model.pack),
          },
          { threads: this.threads },
        ),
      release: (recognizer) => recognizer.release(),
      decoder: (recognizer, config) => {
        const language = supportedLanguage(this.model, config?.language);
        return async (audio) => whistleDetail(await recognizer.transcribe(audio, language));
      },
    });
  }

  get label(): string {
    return `onnxruntime-node-${this.#runtime.version}:${this.model.name}`;
  }

  get threads(): number {
    return ortThreads(this.model);
  }

  prepare(): Promise<void> {
    return this.#lifecycle.prepare();
  }

  open(config?: STTConfig): Promise<STTSession> {
    return this.#lifecycle.open(config);
  }

  /** Free the recognizer's native sessions; a later open() loads them again. */
  release(): Promise<void> {
    return this.#lifecycle.release();
  }

  #file(name: string): string {
    return join(this.paths.dir, name);
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
