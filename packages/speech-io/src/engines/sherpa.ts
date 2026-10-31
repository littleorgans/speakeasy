import { EventEmitter } from "node:events";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import type {
  EmptyEvent,
  EndpointConfig,
  EndpointMode,
  FinalEvent,
  STTConfig,
  STTSession,
  VoiceToText,
} from "../contract.ts";
import { ensureAsset, type AssetSpec, type FileSetSpec } from "./assets.ts";
import type { SilenceGate } from "./offline-session.ts";
import { resultDetail, resultText } from "./result.ts";
import type {
  Hotwords,
  OnlineRecognizer,
  OnlineRecognizerConfig,
  OnlineStream,
  SherpaBinding,
} from "./sherpa-api.ts";
import {
  resolveModelPaths,
  resolveSherpaModel,
  type SherpaModel,
  type SherpaModelId,
  type StreamingSherpaModel,
} from "./sherpa-models.ts";

/**
 * SherpaEngine: primary streaming STT engine.
 *
 * Uses sherpa-onnx's native N-API addon, injected as `binding`, with an
 * English streaming Zipformer transducer selected from the model registry
 * (see sherpa-models.ts). The model streams partials, uses sherpa's built-in
 * endpoint detector, and keeps all inference in-process.
 */

export const SHERPA_SAMPLE_RATE = 16_000;
export const SHERPA_FEATURE_DIM = 80;
/** CPU threads per sherpa recognizer; onnxruntime engines match it. */
export const SHERPA_NUM_THREADS = 2;
/**
 * Silence pushed instantly at flush() so the encoder's pending chunk and
 * right context fill without waiting for real trailing audio. Sized for the
 * registry's largest chunk as a worst case (~640ms for a chunk-16 model at 4x
 * subsampling on 10ms features) plus right context, with headroom, so it also
 * covers the smaller-context default. Decoding the extra silence costs only a
 * few encoder steps, so flush stays well under the 200ms budget.
 */
const FLUSH_PADDING_MS = 1_200;
const FLUSH_PADDING_SAMPLES = (SHERPA_SAMPLE_RATE * FLUSH_PADDING_MS) / 1_000;
const DEFAULT_ENDPOINT: Required<EndpointConfig> = {
  mode: "eager",
  minTrailingSilenceMs: 200,
  minUtteranceMs: 20_000,
};

export type SherpaEngineOptions = {
  model: SherpaModelId;
  /** Folder holding downloaded archives and extracted models. */
  modelRoot: string;
  binding: SherpaBinding;
  /**
   * Streaming models only: contextual biasing, which switches decoding to
   * modified_beam_search. Absent means greedy search.
   */
  hotwords?: Hotwords;
  /** Offline models only: the silence gate every session runs before decode. */
  silenceGate?: SilenceGate;
};

/** Shared by the streaming and offline engines. */
export class SherpaModelAsset<M extends SherpaModel> {
  readonly model: M;
  readonly paths: AssetSpec | FileSetSpec;
  readonly #binding: SherpaBinding;

  constructor(model: M, options: SherpaEngineOptions) {
    this.model = model;
    this.paths = resolveModelPaths(model, options.modelRoot);
    this.#binding = options.binding;
  }

  get label(): string {
    return `sherpa-onnx-node-${this.#binding.version}:${this.model.name}`;
  }

  file(name: string): string {
    return join(this.paths.dir, name);
  }

  async ensure(): Promise<void> {
    await ensureAsset(this.paths);
  }
}

export function assertSampleRate(engine: string, config?: STTConfig): void {
  if (config?.sampleRate && config.sampleRate !== SHERPA_SAMPLE_RATE) {
    throw new Error(
      `${engine} expects ${SHERPA_SAMPLE_RATE} Hz frames, received ${config.sampleRate} Hz`,
    );
  }
}

export function samplesToMs(samples: number): number {
  return (samples * 1_000) / SHERPA_SAMPLE_RATE;
}

export class SherpaEngine implements VoiceToText {
  readonly #asset: SherpaModelAsset<StreamingSherpaModel>;
  readonly #binding: SherpaBinding;
  readonly #hotwords: Hotwords | undefined;

  constructor(options: SherpaEngineOptions) {
    const model = resolveSherpaModel(options.model);
    if (model.kind !== "streaming") {
      throw new Error(
        `SherpaEngine needs a streaming model; ${options.model} is offline (use SherpaOfflineEngine)`,
      );
    }
    this.#asset = new SherpaModelAsset(model, options);
    this.#binding = options.binding;
    this.#hotwords = options.hotwords;
  }

  get label(): string {
    return this.#asset.label;
  }

  get model(): StreamingSherpaModel {
    return this.#asset.model;
  }

  get threads(): number {
    return SHERPA_NUM_THREADS;
  }

  async prepare(): Promise<void> {
    await this.#asset.ensure();
  }

  async open(config?: STTConfig): Promise<STTSession> {
    assertSampleRate("SherpaEngine", config);
    await this.#asset.ensure();
    const endpoint = normalizeEndpoint(config?.endpoint);
    const hotwords = this.#hotwords;
    if (hotwords) {
      // Loud on purpose: hotwords flip the decoder to modified_beam_search and
      // must never be a silent change to the measured baseline.
      console.warn(
        `[sherpa] HOTWORDS ACTIVE: ${hotwords.phrases.length} terms, score=${hotwords.score}, decoding=modified_beam_search (baseline is greedy). terms: ${hotwords.phrases.join(", ")}`,
      );
    }
    const recognizer = new this.#binding.OnlineRecognizer(
      createRecognizerConfig(this.#asset, endpoint, hotwords),
    );
    return new SherpaSession(recognizer, endpoint);
  }
}

class SherpaSession extends EventEmitter implements STTSession {
  #recognizer: OnlineRecognizer;
  #stream: OnlineStream;
  #endpoint: Required<EndpointConfig>;
  #lastPartial = "";
  /** Samples of this utterance, excluding flush padding. */
  #samples = 0;
  /** Samples the stream has accepted since creation, padding included. */
  #streamSamples = 0;
  #utteranceStartSamples = 0;
  #closed = false;

  constructor(recognizer: OnlineRecognizer, endpoint: Required<EndpointConfig>) {
    super();
    this.#recognizer = recognizer;
    this.#stream = recognizer.createStream();
    this.#endpoint = endpoint;
  }

  pushAudio(frame: Float32Array): void {
    if (this.#closed) {
      throw new Error("Cannot push audio after end()");
    }
    const startedAt = performance.now();
    this.#stream.acceptWaveform({ samples: frame, sampleRate: SHERPA_SAMPLE_RATE });
    this.#samples += frame.length;
    this.#streamSamples += frame.length;
    this.#decodeReady();
    this.#emitPartial();
    this.#maybeAutoCommit(startedAt);
  }

  flush(): void {
    if (this.#closed) {
      return;
    }
    const startedAt = performance.now();
    // Commit fast without closing the session: the streaming zipformer holds
    // the last word until its chunk/right-context is satisfied, so push
    // synthetic silence instantly (never inputFinished(), which would end the
    // stream), decode everything pending, then commit. reset() inside
    // #commit() leaves the stream accepting audio for the next utterance.
    this.#stream.acceptWaveform({
      samples: new Float32Array(FLUSH_PADDING_SAMPLES),
      sampleRate: SHERPA_SAMPLE_RATE,
    });
    this.#streamSamples += FLUSH_PADDING_SAMPLES;
    this.#decodeReady();
    this.#commit({ endpoint: false, explicit: true, startedAt });
  }

  reset(): void {
    this.#recognizer.reset(this.#stream);
    this.#lastPartial = "";
    this.#samples = 0;
    this.#utteranceStartSamples = this.#streamSamples;
  }

  async end(): Promise<void> {
    if (this.#closed) {
      return;
    }
    const startedAt = performance.now();
    this.#closed = true;
    this.#stream.inputFinished();
    this.#decodeReady();
    this.#commit({ endpoint: true, explicit: false, startedAt });
  }

  #decodeReady(): void {
    while (this.#recognizer.isReady(this.#stream)) {
      this.#recognizer.decode(this.#stream);
    }
  }

  #emitPartial(): void {
    const text = resultText(this.#recognizer.getResult(this.#stream));
    if (text && text !== this.#lastPartial) {
      this.#lastPartial = text;
      this.emit("partial", { text });
    }
  }

  #maybeAutoCommit(startedAt: number): void {
    if (
      this.#endpoint.mode !== "manual" &&
      this.#recognizer.isEndpoint(this.#stream)
    ) {
      this.#commit({ endpoint: true, explicit: false, startedAt });
    }
  }

  /**
   * `explicit` commits (flush) always answer: `final`, or `empty` when no text
   * decoded. Endpoint and end() commits stay silent without text, since the
   * endpoint detector also fires on pure silence.
   */
  #commit(commit: { endpoint: boolean; explicit: boolean; startedAt: number }): void {
    const { text, tokens } = resultDetail(
      this.#recognizer.getResult(this.#stream),
      this.#utteranceStartSamples / SHERPA_SAMPLE_RATE,
    );
    const timing = {
      audioMs: samplesToMs(this.#samples),
      decodeMs: performance.now() - commit.startedAt,
    };
    if (commit.endpoint) {
      this.emit("endpoint", {});
    }
    if (text) {
      this.emit("final", { text, tokens, ...timing } satisfies FinalEvent);
    } else if (commit.explicit) {
      this.emit("empty", timing satisfies EmptyEvent);
    }
    this.reset();
  }
}

function createRecognizerConfig(
  asset: SherpaModelAsset<StreamingSherpaModel>,
  endpoint: Required<EndpointConfig>,
  hotwords?: Hotwords,
): OnlineRecognizerConfig {
  const { model } = asset;
  const endpointEnabled = endpoint.mode !== "manual";
  return {
    featConfig: {
      sampleRate: SHERPA_SAMPLE_RATE,
      featureDim: SHERPA_FEATURE_DIM,
    },
    modelConfig: {
      transducer: {
        encoder: asset.file(model.encoder),
        decoder: asset.file(model.decoder),
        joiner: asset.file(model.joiner),
      },
      tokens: asset.file(model.tokens),
      numThreads: SHERPA_NUM_THREADS,
      provider: "cpu",
      debug: 0,
      // bpe modelingUnit only when the model ships a bpe.model; several
      // streaming zipformers decode straight from tokens.txt without one.
      ...(model.bpe ? { modelingUnit: "bpe", bpeVocab: asset.file(model.bpe) } : {}),
    },
    // Hotwords bias only under modified_beam_search; greedy ignores them, so
    // the decoder switches only when hotwords are actually passed.
    decodingMethod: hotwords ? "modified_beam_search" : "greedy_search",
    maxActivePaths: 4,
    enableEndpoint: endpointEnabled ? 1 : 0,
    rule1MinTrailingSilence: endpoint.minTrailingSilenceMs / 1_000,
    rule2MinTrailingSilence: endpoint.minTrailingSilenceMs / 1_000,
    rule3MinUtteranceLength: endpoint.minUtteranceMs / 1_000,
    hotwordsFile: hotwords?.file ?? "",
    hotwordsScore: hotwords?.score ?? 1.5,
    ruleFsts: "",
    ruleFars: "",
    blankPenalty: 0,
  };
}

function normalizeEndpoint(endpoint?: EndpointConfig): Required<EndpointConfig> {
  const mode = endpoint?.mode ?? DEFAULT_ENDPOINT.mode;
  validateEndpointMode(mode);

  return {
    mode,
    minTrailingSilenceMs:
      endpoint?.minTrailingSilenceMs ??
      DEFAULT_ENDPOINT.minTrailingSilenceMs,
    minUtteranceMs:
      endpoint?.minUtteranceMs ?? DEFAULT_ENDPOINT.minUtteranceMs,
  };
}

function validateEndpointMode(mode: string): asserts mode is EndpointMode {
  if (mode !== "eager" && mode !== "turn-aware" && mode !== "manual") {
    throw new Error(`Unsupported endpoint mode ${mode}`);
  }
}
