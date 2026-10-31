/**
 * Ambient types for the TTS paths that require sherpa-onnx-node directly. The
 * STT engines take the binding by injection; their types live in
 * engines/sherpa-api.ts. Checked against 2.0.1, which ignores the old
 * `dataDir` (espeak-ng data) field, so it is not declared.
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
    noiseScale?: number;
    noiseScaleW?: number;
    lengthScale?: number;
  };

  export type OfflineTtsKokoroModelConfig = {
    model?: string;
    voices?: string;
    tokens?: string;
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

  /** Per call settings; when present they replace the request's sid and speed. */
  export type GenerationConfig = {
    sid?: number;
    speed?: number;
    silenceScale?: number;
    /** Caller phonemized input: one non-empty codepoint array per sentence. */
    phonemeCodepoints?: number[][];
  };

  export type TtsRequest = {
    /** Empty when `generationConfig.phonemeCodepoints` carries the input. */
    text: string;
    sid: number;
    speed: number;
    enableExternalBuffer?: boolean;
    generationConfig?: GenerationConfig;
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
    /** The espeak voice the model metadata names, e.g. "en-us". */
    lang: string;
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
