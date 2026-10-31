/** speak-easy declarations for the vendored whistle.js; see NOTICE. */
import type { OrtBackend } from "./backend.js";
import type { Pack } from "./pack.js";

export const LANGS: readonly string[];

export class Tokenizer {
  constructor(tok: unknown);
}

export class Engram {
  constructor(pack: Pack);
}

export type WhistleContext = {
  tok: Tokenizer;
  /** Mel filterbank, 257 x 80. */
  fb: Float32Array;
  engram: Engram;
};

export type TranscribeOptions = {
  timestamps?: boolean;
  /** One of LANGS; omitted means language ID on the first step. */
  language?: string;
  keywords?: string[];
};

export type TranscribeWord = {
  word: string;
  /** Mean softmax probability of the word's tokens, rounded to 3 places. */
  probability: number;
  /** Seconds, at 80 ms resolution; present when timestamps were asked. */
  start?: number;
  end?: number;
};

export type TranscribeResult = {
  text: string;
  language: string | null;
  words: TranscribeWord[];
};

export function transcribe(
  backend: OrtBackend,
  ctx: WhistleContext,
  pcm: Float32Array,
  opts?: TranscribeOptions,
): Promise<TranscribeResult>;
