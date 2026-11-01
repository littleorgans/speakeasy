import { ProviderError, type OutputApiKind, type SpeechOutputConfig } from "../provider-config.ts";
import { CartesiaTextToSpeech, type CartesiaOptions } from "./cartesia.ts";
import { DEFAULT_LOCAL_OUTPUT } from "./catalogue.ts";
import type { TextToSpeech, TTSSession } from "./contract.ts";
import { LocalSpeechOutputFactory } from "./factory.ts";
import { OpenAITextToSpeech, type OpenAISocketFactory } from "./openai.ts";

/** Trusted host execution handles, kept outside portable preferences. */
export type SpeechOutputContext = {
  getKey?: () => string | undefined;
  beforeRequest?: () => void | Promise<void>;
  signal?: AbortSignal;
  observe?: CartesiaOptions["onRequestComplete"];
};

export type SpeechOutputFactoryOptions = {
  local?: Pick<LocalSpeechOutputFactory, "create" | "quiesce" | "close">;
  getKey?: (kind: OutputApiKind) => string | undefined;
  fetch?: CartesiaOptions["fetch"];
  createSocket?: OpenAISocketFactory;
  now?: () => number;
};

/** One provider composition path for voice trials and selected cascade output. */
export class SpeechOutputFactory {
  readonly #local: NonNullable<SpeechOutputFactoryOptions["local"]>;
  readonly #options: SpeechOutputFactoryOptions;
  #closed = false;

  constructor(options: SpeechOutputFactoryOptions = {}) {
    this.#options = { ...options };
    this.#local = options.local ?? new LocalSpeechOutputFactory();
  }

  /** Selection is inert. Local validation and model support belong to the local factory. */
  create(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT, context: SpeechOutputContext = {}): TextToSpeech {
    if (this.#closed) throw new Error("Speech output factory is closed");
    const selected = { ...config, provider: { ...config.provider } };
    const execution = { ...context };
    let open: () => Promise<TTSSession>;
    switch (selected.provider.kind) {
      case "local": {
        const output = this.#local.create(selected);
        open = () => output.open();
        break;
      }
      case "cartesia":
        open = () => new CartesiaTextToSpeech({
          fetch: this.#options.fetch,
          apiKey: execution.getKey ?? (() => this.#options.getKey?.("cartesia")),
          beforeRequest: execution.beforeRequest,
          signal: execution.signal,
          onRequestComplete: execution.observe,
          now: this.#options.now,
        }).open(selected);
        break;
      case "openai":
        open = () => {
          if (!this.#options.createSocket) throw new ProviderError("PROVIDER_UNAVAILABLE");
          return new OpenAITextToSpeech({
            createSocket: this.#options.createSocket,
            apiKey: execution.getKey ?? (() => this.#options.getKey?.("openai")),
            beforeRequest: execution.beforeRequest,
            now: this.#options.now,
          }).open(selected, { signal: execution.signal });
        };
        break;
      default:
        throw new ProviderError("CONFIG_UNSUPPORTED");
    }
    return {
      open: async (override = {}) => {
        if (this.#closed) throw new Error("Speech output factory is closed");
        if (Object.keys(override).length) throw new Error("Selected output configuration is fixed");
        execution.signal?.throwIfAborted();
        return open();
      },
    };
  }

  /** Call after operation/session settlement. Never loads an unused graph. */
  async quiesce(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT): Promise<void> {
    if (config.provider.kind === "local") await this.#local.quiesce(config);
  }

  /** Hosts close sessions first. The local factory waits for native work tails. */
  async close(): Promise<void> {
    this.#closed = true;
    await this.#local.close();
  }
}
