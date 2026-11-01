import type { SpeechOutputConfig } from "../provider-config.ts";
import type { TextToSpeech, TTSSession } from "./contract.ts";
import { DEFAULT_LOCAL_OUTPUT, resolveLocalOutput, type LocalOutputModelId } from "./catalogue.ts";
import { LOCAL_TTS_MODELS } from "./models.ts";
import type { LoadedTtsSynth } from "./sherpa-tts.ts";

export type LocalSpeechOutputOptions = {
  /** Server supplied location. Never part of portable configuration. */
  root?: string;
  numThreads?: number;
  /** Test seam. Production uses TtsSynth with downloads disabled. */
  createSynth?: (model: LocalOutputModelId) => Promise<LoadedTtsSynth>;
};

/** At most the two approved models. Voice changes share a graph and serialize native calls. */
export class LocalSpeechOutputFactory {
  readonly #models = new Map<LocalOutputModelId, Promise<LoadedTtsSynth>>();
  readonly #options: LocalSpeechOutputOptions;
  #closed = false;

  constructor(options: LocalSpeechOutputOptions = {}) {
    this.#options = options;
  }

  create(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT): TextToSpeech {
    resolveLocalOutput(config);
    const selected = { ...config, provider: { ...config.provider } };
    return {
      open: async (override = {}) => {
        if (Object.keys(override).length) throw new Error("Selected output configuration is fixed");
        return this.open(selected);
      },
    };
  }

  async open(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT): Promise<TTSSession> {
    if (this.#closed) throw new Error("Local speech output factory is closed");
    const { model, voice, speed } = resolveLocalOutput(config);
    const synth = await this.#load(model.id);
    if (this.#closed) throw new Error("Local speech output factory is closed");
    const { SherpaTextToSpeech } = await import("./sherpa-tts.ts");
    return new SherpaTextToSpeech(async () => synth).open({ model: model.id, voice: voice.speakerId, speed });
  }

  /** Drops references; the native binding offers no explicit free or abort. */
  async close(): Promise<void> {
    this.#closed = true;
    this.#models.clear();
  }

  #load(modelId: LocalOutputModelId): Promise<LoadedTtsSynth> {
    let pending = this.#models.get(modelId);
    if (!pending) {
      pending = (async () => {
        const synth = this.#options.createSynth
          ? await this.#options.createSynth(modelId)
          : await (await import("./synth.ts")).TtsSynth.create(modelId, this.#options.numThreads, {
            root: this.#options.root, allowDownload: false,
          });
        const model = LOCAL_TTS_MODELS[modelId];
        if (synth.numSpeakers !== model.numSpeakers || synth.sampleRate !== model.sampleRate) {
          throw new Error("Loaded local model does not match its pinned bundle");
        }
        let tail = Promise.resolve();
        const serialized: LoadedTtsSynth = {
          numSpeakers: synth.numSpeakers,
          sampleRate: synth.sampleRate,
          synth: (request) => {
            const result = tail.then(() => synth.synth(request));
            tail = result.then(() => {}, () => {});
            return result;
          },
        };
        return serialized;
      })().catch((error: unknown) => {
        this.#models.delete(modelId);
        throw error;
      });
      this.#models.set(modelId, pending);
    }
    return pending;
  }
}
