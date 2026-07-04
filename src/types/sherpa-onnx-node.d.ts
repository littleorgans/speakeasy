declare module "sherpa-onnx-node" {
  export type Waveform = {
    samples: Float32Array;
    sampleRate: number;
  };

  export type OnlineRecognizerResult = {
    text?: string;
    tokens?: string[];
    timestamps?: number[];
    json?: string;
  };

  export type OnlineRecognizerConfig = {
    featConfig: {
      sampleRate: number;
      featureDim: number;
    };
    modelConfig: {
      transducer: {
        encoder: string;
        decoder: string;
        joiner: string;
      };
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

  export class OnlineStream {
    acceptWaveform(waveform: Waveform): void;
    inputFinished(): void;
  }

  export class OnlineRecognizer {
    constructor(config: OnlineRecognizerConfig);
    createStream(): OnlineStream;
    isReady(stream: OnlineStream): boolean;
    decode(stream: OnlineStream): void;
    isEndpoint(stream: OnlineStream): boolean;
    reset(stream: OnlineStream): void;
    getResult(stream: OnlineStream): OnlineRecognizerResult;
  }

  export type OfflineTtsVitsModelConfig = {
    model?: string;
    lexicon?: string;
    tokens?: string;
    dataDir?: string;
    noiseScale?: number;
    noiseScaleW?: number;
    lengthScale?: number;
  };

  export type OfflineTtsKokoroModelConfig = {
    model?: string;
    voices?: string;
    tokens?: string;
    dataDir?: string;
    lengthScale?: number;
    lexicon?: string;
    lang?: string;
  };

  export type OfflineTtsModelConfig = {
    vits?: OfflineTtsVitsModelConfig;
    kokoro?: OfflineTtsKokoroModelConfig;
    numThreads?: number;
    debug?: boolean | number;
    provider?: string;
  };

  export type OfflineTtsConfig = {
    model?: OfflineTtsModelConfig;
    maxNumSentences?: number;
    silenceScale?: number;
  };

  export type GeneratedAudio = {
    samples: Float32Array;
    sampleRate: number;
  };

  export type TtsRequest = {
    text: string;
    sid: number;
    speed: number;
    enableExternalBuffer?: boolean;
  };

  export type TtsProgress = {
    samples: Float32Array;
    progress: number;
  };

  export class OfflineTts {
    constructor(config: OfflineTtsConfig);
    static createAsync(config: OfflineTtsConfig): Promise<OfflineTts>;
    numSpeakers: number;
    sampleRate: number;
    generate(request: TtsRequest): GeneratedAudio;
    generateAsync(
      request: TtsRequest & {
        /** Streaming chunk callback; return 0/false to stop generation. */
        onProgress?: (info: TtsProgress) => number | boolean | void;
      },
    ): Promise<GeneratedAudio>;
  }

  export function writeWave(filename: string, audio: Waveform): boolean;

  export const version: string;
}
