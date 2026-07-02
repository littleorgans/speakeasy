import { EventEmitter } from "node:events";
import { mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
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
import { downloadFile, extractTarBz2, hasNonEmptyFile } from "./assets.ts";

/**
 * SherpaEngine: primary streaming STT engine for the benchmark spike.
 *
 * Uses sherpa-onnx-node's native N-API addon with the English streaming
 * Zipformer transducer. The model streams partials, uses sherpa's built-in
 * endpoint detector, and keeps all inference in-process for Electron main.
 */

const SAMPLE_RATE = 16_000;
const FEATURE_DIM = 80;
/**
 * Silence pushed instantly at flush() so the encoder's pending chunk and
 * right context fill without waiting for real trailing audio. Sized for one
 * full chunk (~640ms for chunk-16 at 4x subsampling on 10ms features) plus
 * right context, with headroom. Decoding the extra silence costs only a few
 * encoder steps, so flush stays well under the 200ms budget.
 */
const FLUSH_PADDING_MS = 1_200;
const FLUSH_PADDING_SAMPLES = (SAMPLE_RATE * FLUSH_PADDING_MS) / 1_000;
const MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26";
const MODEL_URL =
  `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2`;
const SHERPA_ROOT = join(process.cwd(), "models", "sherpa");
const MODEL_DIR = join(SHERPA_ROOT, MODEL_NAME);
const MODEL_ARCHIVE = join(SHERPA_ROOT, `${MODEL_NAME}.tar.bz2`);
const ENCODER = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const DECODER = "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const JOINER = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const TOKENS = "tokens.txt";
const BPE_MODEL = "bpe.model";
const DEFAULT_ENDPOINT: Required<EndpointConfig> = {
  mode: "eager",
  minTrailingSilenceMs: 200,
  minUtteranceMs: 20_000,
};
const require = createRequire(import.meta.url);
const sherpa = require("sherpa-onnx-node") as typeof import("sherpa-onnx-node");

export class SherpaEngine implements VoiceToText {
  get label(): string {
    return `sherpa-onnx-node-${sherpa.version}:${MODEL_NAME}:int8`;
  }

  async prepare(): Promise<void> {
    await ensureModel();
  }

  async open(config?: STTConfig): Promise<STTSession> {
    if (config?.sampleRate && config.sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `SherpaEngine expects ${SAMPLE_RATE} Hz frames, received ${config.sampleRate} Hz`,
      );
    }
    await ensureModel();
    const endpoint = normalizeEndpoint(config?.endpoint);
    const recognizer = new sherpa.OnlineRecognizer(
      createRecognizerConfig(endpoint),
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

async function ensureModel(): Promise<void> {
  const encoderPath = join(MODEL_DIR, ENCODER);
  if (await hasNonEmptyFile(encoderPath)) {
    return;
  }

  await mkdir(SHERPA_ROOT, { recursive: true });
  if (!(await hasNonEmptyFile(MODEL_ARCHIVE))) {
    await downloadFile(MODEL_URL, MODEL_ARCHIVE);
  }
  await extractTarBz2(MODEL_ARCHIVE, SHERPA_ROOT);
}

function createRecognizerConfig(
  endpoint: Required<EndpointConfig>,
): OnlineRecognizerConfig {
  const endpointEnabled = endpoint.mode !== "manual";
  return {
    featConfig: {
      sampleRate: SAMPLE_RATE,
      featureDim: FEATURE_DIM,
    },
    modelConfig: {
      transducer: {
        encoder: join(MODEL_DIR, ENCODER),
        decoder: join(MODEL_DIR, DECODER),
        joiner: join(MODEL_DIR, JOINER),
      },
      tokens: join(MODEL_DIR, TOKENS),
      numThreads: 2,
      provider: "cpu",
      debug: 0,
      modelingUnit: "bpe",
      bpeVocab: join(MODEL_DIR, BPE_MODEL),
    },
    decodingMethod: "greedy_search",
    maxActivePaths: 4,
    enableEndpoint: endpointEnabled ? 1 : 0,
    rule1MinTrailingSilence: endpoint.minTrailingSilenceMs / 1_000,
    rule2MinTrailingSilence: endpoint.minTrailingSilenceMs / 1_000,
    rule3MinUtteranceLength: endpoint.minUtteranceMs / 1_000,
    hotwordsFile: "",
    hotwordsScore: 1.5,
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
