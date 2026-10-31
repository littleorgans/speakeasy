/**
 * Ambient types for the TTS paths that require sherpa-onnx-node directly. The
 * STT engines take the binding by injection; their types live in
 * engines/sherpa-api.ts.
 */
declare module "sherpa-onnx-node" {
  export type Waveform = {
    samples: Float32Array;
    sampleRate: number;
  };

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
