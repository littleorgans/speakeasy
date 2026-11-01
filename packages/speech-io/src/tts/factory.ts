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

type SharedTtsSynth = LoadedTtsSynth & { quiesce(): Promise<void> };

/** At most the two approved models. Voice changes share a graph and serialize native calls. */
export class LocalSpeechOutputFactory {
  readonly #models = new Map<LocalOutputModelId, Promise<SharedTtsSynth>>();
  readonly #options: LocalSpeechOutputOptions;
  #closed = false;
  #closing?: Promise<void>;

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

  /**
   * After the operation and session close settle, await the selected graph's
   * pending load and queued/running native calls before releasing a work lease.
   * Does not load an unused graph. A failed load is already quiescent.
   */
  async quiesce(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT): Promise<void> {
    const { model } = resolveLocalOutput(config);
    const synth = await this.#models.get(model.id)?.catch(() => undefined);
    await synth?.quiesce();
  }

  /** Prevent new native work, then drain loads/calls before dropping references. */
  async close(): Promise<void> {
    if (!this.#closing) {
      this.#closed = true;
      this.#closing = (async () => {
        const synths = await Promise.all([...this.#models.values()].map((pending) => pending.catch(() => undefined)));
        await Promise.all(synths.map((synth) => synth?.quiesce()));
        // The native binding offers no explicit free or abort.
        this.#models.clear();
      })();
    }
    await this.#closing;
  }

  #load(modelId: LocalOutputModelId): Promise<SharedTtsSynth> {
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
        const serialized: SharedTtsSynth = {
          numSpeakers: synth.numSpeakers,
          sampleRate: synth.sampleRate,
          synth: (request) => {
            if (this.#closed) return Promise.reject(new Error("Local speech output factory is closed"));
            const result = tail.then(() => synth.synth(request));
            tail = result.then(() => {}, () => {});
            return result;
          },
          quiesce: () => tail,
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
