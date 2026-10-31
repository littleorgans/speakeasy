/**
 * The slice of sherpa-onnx-node 1.13.3 the STT engines call, declared here so
 * the engines take the native binding by injection and never import the
 * package. Callers load it with `sherpa-binding.ts::loadSherpaBinding` or, in
 * hosts that ship the native separately, their own `createRequire`.
 */

export type Waveform = {
  samples: Float32Array;
  sampleRate: number;
};

/** Prepared contextual biasing supplied by the host. */
export type Hotwords = {
  /** Cleaned phrases, one per hotword. */
  phrases: string[];
  /** Path to the cleaned file handed to sherpa. */
  file: string;
  score: number;
};

/**
 * Recognizer result JSON. Streaming fills `ys_probs` and `start_time`;
 * offline fills `ys_log_probs` and `durations` for models that report them.
 * Both are natural log probabilities, one per token.
 */
export type RecognizerResult = {
  text?: string;
  tokens?: string[];
  timestamps?: number[];
  ys_probs?: number[];
  ys_log_probs?: number[];
  durations?: number[];
  start_time?: number;
};

export type FeatureConfig = {
  sampleRate: number;
  featureDim: number;
};

export type TransducerModelConfig = {
  encoder: string;
  decoder: string;
  joiner: string;
};

export type OnlineRecognizerConfig = {
  featConfig: FeatureConfig;
  modelConfig: {
    transducer: TransducerModelConfig;
    tokens: string;
    numThreads?: number;
    provider?: string;
    debug?: boolean | number;
    modelType?: string;
    modelingUnit?: string;
    bpeVocab?: string;
  };
  decodingMethod?: string;
  maxActivePaths?: number;
  enableEndpoint?: boolean | number;
  rule1MinTrailingSilence?: number;
  rule2MinTrailingSilence?: number;
  rule3MinUtteranceLength?: number;
  hotwordsFile?: string;
  hotwordsScore?: number;
  ruleFsts?: string;
  ruleFars?: string;
  blankPenalty?: number;
};

export type OfflineMoonshineModelConfig = {
  preprocessor: string;
  encoder: string;
  uncachedDecoder: string;
  cachedDecoder: string;
};

export type OfflineCanaryModelConfig = {
  encoder: string;
  decoder: string;
  srcLang: string;
  tgtLang: string;
  usePnc: number;
};

export type OfflineQwen3AsrModelConfig = {
  convFrontend: string;
  encoder: string;
  decoder: string;
  /** Tokenizer directory. */
  tokenizer: string;
  hotwords?: string;
};

export type OfflineRecognizerConfig = {
  featConfig: FeatureConfig;
  modelConfig: {
    transducer?: TransducerModelConfig;
    moonshine?: OfflineMoonshineModelConfig;
    canary?: OfflineCanaryModelConfig;
    qwen3Asr?: OfflineQwen3AsrModelConfig;
    /** Empty for models whose vocabulary lives elsewhere (Qwen3 ASR). */
    tokens: string;
    numThreads?: number;
    provider?: string;
    debug?: boolean | number;
    modelType?: string;
  };
  decodingMethod?: string;
};

export interface OnlineStream {
  acceptWaveform(waveform: Waveform): void;
  inputFinished(): void;
}

export interface OnlineRecognizer {
  createStream(): OnlineStream;
  isReady(stream: OnlineStream): boolean;
  decode(stream: OnlineStream): void;
  isEndpoint(stream: OnlineStream): boolean;
  reset(stream: OnlineStream): void;
  getResult(stream: OnlineStream): RecognizerResult;
}

export interface OfflineStream {
  acceptWaveform(waveform: Waveform): void;
}

export interface OfflineRecognizer {
  createStream(): OfflineStream;
  decodeAsync(stream: OfflineStream): Promise<RecognizerResult>;
}

export type SherpaBinding = {
  version: string;
  OnlineRecognizer: new (config: OnlineRecognizerConfig) => OnlineRecognizer;
  OfflineRecognizer: new (config: OfflineRecognizerConfig) => OfflineRecognizer;
};
