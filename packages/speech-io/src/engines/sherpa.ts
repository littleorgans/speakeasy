import { EventEmitter } from "node:events";
import { createRequire } from "node:module";
import type {
  OnlineRecognizer as OnlineRecognizerClass,
  OnlineRecognizerConfig,
  OnlineRecognizerResult,
  OnlineStream,
} from "sherpa-onnx-node";
import type {
  EndpointConfig,
  EndpointMode,
  STTConfig,
  STTSession,
  VoiceToText,
} from "../contract.ts";
import { ensureAsset } from "./assets.ts";
import { loadHotwords, type Hotwords } from "./hotwords.ts";
import {
  DEFAULT_SHERPA_MODEL,
  resolveModelPaths,
  resolveSherpaModel,
  SHERPA_ROOT,
  type SherpaModel,
  type SherpaModelId,
  type SherpaModelPaths,
} from "./sherpa-models.ts";

/**
 * SherpaEngine: primary streaming STT engine for the benchmark spike.
 *
 * Uses sherpa-onnx-node's native N-API addon with an English streaming
 * Zipformer transducer selected from the model registry (see
 * sherpa-models.ts). The model streams partials, uses sherpa's built-in
 * endpoint detector, and keeps all inference in-process for Electron main.
 */

const SAMPLE_RATE = 16_000;
const FEATURE_DIM = 80;
/**
 * Silence pushed instantly at flush() so the encoder's pending chunk and
 * right context fill without waiting for real trailing audio. Sized for the
 * registry's largest chunk as a worst case (~640ms for a chunk-16 model at 4x
 * subsampling on 10ms features) plus right context, with headroom, so it also
 * covers the smaller-context default. Decoding the extra silence costs only a
 * few encoder steps, so flush stays well under the 200ms budget.
 */
const FLUSH_PADDING_MS = 1_200;
const FLUSH_PADDING_SAMPLES = (SAMPLE_RATE * FLUSH_PADDING_MS) / 1_000;
const DEFAULT_ENDPOINT: Required<EndpointConfig> = {
  mode: "eager",
  minTrailingSilenceMs: 200,
  minUtteranceMs: 20_000,
};
const require = createRequire(import.meta.url);
const sherpa = require("sherpa-onnx-node") as typeof import("sherpa-onnx-node");

export class SherpaEngine implements VoiceToText {
  #model: SherpaModel;
  #paths: SherpaModelPaths;

  constructor(model: SherpaModelId = DEFAULT_SHERPA_MODEL) {
    this.#model = resolveSherpaModel(model);
    this.#paths = resolveModelPaths(this.#model);
  }

  get label(): string {
    return `sherpa-onnx-node-${sherpa.version}:${this.#model.name}`;
  }

  async prepare(): Promise<void> {
    await ensureModel(this.#paths);
  }

  async open(config?: STTConfig): Promise<STTSession> {
    if (config?.sampleRate && config.sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `SherpaEngine expects ${SAMPLE_RATE} Hz frames, received ${config.sampleRate} Hz`,
      );
    }
    await ensureModel(this.#paths);
    const endpoint = normalizeEndpoint(config?.endpoint);
    const hotwords = await loadHotwords();
    if (hotwords) {
      // Loud on purpose: hotwords flip the decoder to modified_beam_search and
      // must never be a silent change to the measured baseline.
      console.warn(
        `[sherpa] HOTWORDS ACTIVE: ${hotwords.phrases.length} terms from ./hotwords.txt, score=${hotwords.score}, decoding=modified_beam_search (baseline is greedy). terms: ${hotwords.phrases.join(", ")}`,
      );
    }
    const recognizer = new sherpa.OnlineRecognizer(
      createRecognizerConfig(this.#paths, endpoint, hotwords),
    );
    return new SherpaSession(recognizer, endpoint);
  }
}

class SherpaSession extends EventEmitter implements STTSession {
  #recognizer: OnlineRecognizerClass;
  #stream: OnlineStream;
  #endpoint: Required<EndpointConfig>;
  #lastPartial = "";
  #closed = false;

  constructor(
    recognizer: OnlineRecognizerClass,
    endpoint: Required<EndpointConfig>,
  ) {
    super();
    this.#recognizer = recognizer;
    this.#stream = recognizer.createStream();
    this.#endpoint = endpoint;
  }

  pushAudio(frame: Float32Array): void {
    if (this.#closed) {
      throw new Error("Cannot push audio after end()");
    }
    this.#stream.acceptWaveform({ samples: frame, sampleRate: SAMPLE_RATE });
    this.#decodeReady();
    this.#emitPartial();
    this.#maybeAutoCommit();
  }

  flush(): void {
    if (this.#closed) {
      return;
    }
    // Commit fast without closing the session: the streaming zipformer holds
    // the last word until its chunk/right-context is satisfied, so push
    // synthetic silence instantly (never inputFinished(), which would end the
    // stream), decode everything pending, then commit. reset() inside
    // #commit() leaves the stream accepting audio for the next utterance.
    this.#stream.acceptWaveform({
      samples: new Float32Array(FLUSH_PADDING_SAMPLES),
      sampleRate: SAMPLE_RATE,
    });
    this.#decodeReady();
    this.#commit(false);
  }

  reset(): void {
    this.#recognizer.reset(this.#stream);
    this.#lastPartial = "";
  }

  async end(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    this.#stream.inputFinished();
    this.#decodeReady();
    this.#commit(true);
  }

  #decodeReady(): void {
    while (this.#recognizer.isReady(this.#stream)) {
      this.#recognizer.decode(this.#stream);
    }
  }

  #emitPartial(): void {
    const text = this.#currentText();
    if (text && text !== this.#lastPartial) {
      this.#lastPartial = text;
      this.emit("partial", { text });
    }
  }

  #maybeAutoCommit(): void {
    if (
      this.#endpoint.mode !== "manual" &&
      this.#recognizer.isEndpoint(this.#stream)
    ) {
      this.#commit(true);
    }
  }

  #commit(emitEndpoint: boolean): void {
    const text = this.#currentText();
    if (emitEndpoint) {
      this.emit("endpoint", {});
    }
    if (text) {
      this.emit("final", { text });
    }
    this.reset();
  }

  #currentText(): string {
    return normalizeResultText(this.#recognizer.getResult(this.#stream));
  }
}

async function ensureModel(paths: SherpaModelPaths): Promise<void> {
  await ensureAsset({
    url: paths.url,
    archive: paths.archive,
    extractTo: SHERPA_ROOT,
    sentinel: paths.encoder,
  });
}

function createRecognizerConfig(
  paths: SherpaModelPaths,
  endpoint: Required<EndpointConfig>,
  hotwords?: Hotwords,
): OnlineRecognizerConfig {
  const endpointEnabled = endpoint.mode !== "manual";
  return {
    featConfig: {
      sampleRate: SAMPLE_RATE,
      featureDim: FEATURE_DIM,
    },
    modelConfig: {
      transducer: {
        encoder: paths.encoder,
        decoder: paths.decoder,
        joiner: paths.joiner,
      },
      tokens: paths.tokens,
      numThreads: 2,
      provider: "cpu",
      debug: 0,
      // bpe modelingUnit only when the model ships a bpe.model; several
      // streaming zipformers decode straight from tokens.txt without one.
      ...(paths.bpe ? { modelingUnit: "bpe", bpeVocab: paths.bpe } : {}),
    },
    // Hotwords bias only under modified_beam_search; greedy ignores them, so
    // the decoder switches only when hotwords are actually loaded.
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

function normalizeResultText(result: OnlineRecognizerResult): string {
  return (result.text ?? "").trim();
}
