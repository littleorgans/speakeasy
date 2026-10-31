import type { RecognizerResult } from "./sherpa-api.ts";

/**
 * Greedy CTC decoding of graph logits into a sherpa shaped result, so the
 * shared `resultDetail` maps it to final text and tokens.
 *
 * Collapse follows transformers v5.16.0 `models/parakeet/tokenization_parakeet.py`
 * `ParakeetTokenizer._decode`: argmax per frame, group consecutive repeats,
 * drop the blank (the pad token). Each kept token takes the time of its first
 * frame and the log softmax of its logit there, both read from the model's
 * own output. Detokenization follows the Hugging Face tokenizers `ByteLevel`
 * decoder named in the export's `tokenizer.json`: token characters map back
 * to bytes through the GPT-2 byte to unicode table, then decode as UTF-8.
 */

export type CtcLogits = {
  /** Row major `[frames, vocab]`. */
  data: Float32Array;
  frames: number;
  vocab: number;
};

/** id to token bytes, from a tokenizers `BPE` model with a `ByteLevel` decoder. */
export class ByteLevelVocab {
  readonly blank: number;
  readonly #bytes: Uint8Array[];

  constructor(bytes: Uint8Array[], blank: number) {
    this.#bytes = bytes;
    this.blank = blank;
  }

  get size(): number {
    return this.#bytes.length;
  }

  /** Validate a tokenizer.json document; the blank is its pad token. */
  static fromTokenizerJson(json: string, blankToken: string): ByteLevelVocab {
    const document = JSON.parse(json) as TokenizerJson;
    if (document.model?.type !== "BPE" || document.decoder?.type !== "ByteLevel") {
      throw new Error("tokenizer.json must hold a BPE model with a ByteLevel decoder");
    }
    const vocab = document.model.vocab ?? {};
    const blank = vocab[blankToken];
    if (blank === undefined) {
      throw new Error(`tokenizer.json has no ${blankToken} token`);
    }
    const bytes: Uint8Array[] = [];
    for (const [token, id] of Object.entries(vocab)) {
      bytes[id] = id === blank ? new Uint8Array(0) : tokenBytes(token);
    }
    if (bytes.length !== Object.keys(vocab).length || bytes.includes(undefined as never)) {
      throw new Error("tokenizer.json vocabulary ids must be dense from 0");
    }
    return new ByteLevelVocab(bytes, blank);
  }

  decode(ids: readonly number[]): string {
    return this.decodeTokens(ids).join("");
  }

  /** Incomplete UTF-8 bytes carry into the token that completes the character. */
  decodeTokens(ids: readonly number[]): string[] {
    const decoder = new TextDecoder("utf-8");
    return ids.map((id, index) =>
      decoder.decode(this.#token(id), { stream: index < ids.length - 1 }),
    );
  }

  #token(id: number): Uint8Array {
    const bytes = this.#bytes[id];
    if (!bytes) {
      throw new Error(`token id ${id} is outside the ${this.size} token vocabulary`);
    }
    return bytes;
  }
}

/** Greedy CTC over logits; `frameSeconds` is the audio one output frame spans. */
export function greedyCtc(
  logits: CtcLogits,
  vocab: ByteLevelVocab,
  frameSeconds: number,
): RecognizerResult {
  if (logits.vocab !== vocab.size || logits.data.length !== logits.frames * logits.vocab) {
    throw new Error(
      `logits [${logits.frames}, ${logits.vocab}] do not match the ${vocab.size} token vocabulary`,
    );
  }
  const ids: number[] = [];
  const timestamps: number[] = [];
  const logProbs: number[] = [];
  let previous = vocab.blank;
  for (let frame = 0; frame < logits.frames; frame += 1) {
    const row = logits.data.subarray(frame * logits.vocab, (frame + 1) * logits.vocab);
    const id = argMax(row);
    if (id !== previous && id !== vocab.blank) {
      ids.push(id);
      timestamps.push(frame * frameSeconds);
      logProbs.push(row[id]! - logSumExp(row));
    }
    previous = id;
  }
  const tokens = vocab.decodeTokens(ids);
  return {
    text: tokens.join(""),
    tokens,
    timestamps,
    ys_log_probs: logProbs,
  };
}

type TokenizerJson = {
  model?: { type?: string; vocab?: Record<string, number> };
  decoder?: { type?: string };
};

/** GPT-2 `bytes_to_unicode`, inverted: printable stand in character to byte. */
const UNICODE_TO_BYTE = (() => {
  const map = new Map<string, number>();
  let shifted = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    const printable =
      (byte >= 0x21 && byte <= 0x7e) || (byte >= 0xa1 && byte <= 0xac) || (byte >= 0xae && byte <= 0xff);
    map.set(String.fromCodePoint(printable ? byte : 256 + shifted++), byte);
  }
  return map;
})();

function tokenBytes(token: string): Uint8Array {
  return Uint8Array.from(Array.from(token), (char) => {
    const byte = UNICODE_TO_BYTE.get(char);
    if (byte === undefined) {
      throw new Error(`token ${JSON.stringify(token)} is not byte level encoded`);
    }
    return byte;
  });
}

function argMax(row: Float32Array): number {
  let best = 0;
  for (let index = 1; index < row.length; index += 1) {
    if (row[index]! > row[best]!) {
      best = index;
    }
  }
  return best;
}

function logSumExp(row: Float32Array): number {
  let max = -Infinity;
  for (const value of row) {
    max = Math.max(max, value);
  }
  let sum = 0;
  for (const value of row) {
    sum += Math.exp(value - max);
  }
  return max + Math.log(sum);
}
