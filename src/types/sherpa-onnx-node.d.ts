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

  export const version: string;
}
