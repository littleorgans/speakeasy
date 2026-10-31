import type { TdtDecoding } from "./ort-tdt-models.ts";
import type { RecognizerResult } from "./sherpa-api.ts";

/**
 * Greedy token and duration (TDT) transducer decoding, shared by every
 * OrtTdtEngine layout, plus the SentencePiece vocabulary that turns its
 * tokens into text.
 *
 * The loop is onnx-asr's `_TransducerAsr._decoding` (asr.py, 0.12.0), the
 * reference the Phonon-2 card names: argmax the token logits and the duration
 * logits, keep a non blank token and advance the prediction network, move by
 * the duration, and step one frame on a blank of duration 0 or after
 * `maxSymbolsPerFrame` tokens on one frame. Fluid's runner (`greedy_tdt`)
 * also stops on its end of text token. Results take sherpa's shape so the
 * shared `resultDetail` maps them.
 */

const WORD_START = "▁";
const CONTROL_PIECE = /^<(unk|pad|blk|\|.*\|)>$/;

/** One decoder step source: the graphs behind a TDT model. */
export interface TdtStepper {
  /** Token then duration logits for an encoder frame under the current context. */
  joint(frame: number): Promise<Float32Array>;
  /** Make a non blank token the prediction network's new context. */
  emit(token: number): Promise<void>;
}

export type TdtToken = { id: number; frame: number; logProb: number };

export async function greedyTdt(
  stepper: TdtStepper,
  frames: number,
  decoding: TdtDecoding,
): Promise<TdtToken[]> {
  const { vocabSize, blankId, durations, maxSymbolsPerFrame, eosId } = decoding;
  const tokens: TdtToken[] = [];
  let frame = 0;
  let emitted = 0;
  while (frame < frames) {
    const logits = await stepper.joint(frame);
    const token = argMax(logits, 0, vocabSize);
    const step = durations[argMax(logits, vocabSize, vocabSize + durations.length)]!;
    if (token === eosId) {
      break;
    }
    if (token !== blankId) {
      tokens.push({ id: token, frame, logProb: logSoftmaxAt(logits, vocabSize, token) });
      await stepper.emit(token);
      emitted += 1;
    }
    if (step > 0) {
      frame += step;
      emitted = 0;
    } else if (token === blankId || emitted === maxSymbolsPerFrame) {
      frame += 1;
      emitted = 0;
    }
  }
  return tokens;
}

function argMax(values: Float32Array, from: number, to: number): number {
  let best = from;
  for (let index = from + 1; index < to; index += 1) {
    if (values[index]! > values[best]!) {
      best = index;
    }
  }
  return best - from;
}

function logSoftmaxAt(logits: Float32Array, size: number, index: number): number {
  let max = -Infinity;
  for (let i = 0; i < size; i += 1) {
    max = Math.max(max, logits[i]!);
  }
  let sum = 0;
  for (let i = 0; i < size; i += 1) {
    sum += Math.exp(logits[i]! - max);
  }
  return logits[index]! - max - Math.log(sum);
}

/** Pieces indexed by token id. */
export type Vocabulary = readonly string[];

/** onnx-asr vocab.txt: one "piece id" per line. */
export function parseVocabTxt(text: string): Vocabulary {
  const pieces: string[] = [];
  for (const line of text.split("\n")) {
    const at = line.lastIndexOf(" ");
    if (at > 0) {
      pieces[Number(line.slice(at + 1))] = line.slice(0, at);
    }
  }
  return pieces;
}

/** Hugging Face tokenizer.json: the model's piece to id map. */
export function parseTokenizerJson(text: string): Vocabulary {
  const json = JSON.parse(text) as { model?: { vocab?: Record<string, number> } };
  const vocab = json.model?.vocab;
  if (!vocab) {
    throw new Error("tokenizer.json has no model.vocab");
  }
  const pieces: string[] = [];
  for (const [piece, id] of Object.entries(vocab)) {
    pieces[id] = piece;
  }
  return pieces;
}

/**
 * Text plus per token detail in sherpa's result shape. Control pieces
 * (`<unk>`, `<pad>`, `<blk>`, `<|...|>`) are dropped, the rule the Phonon-2
 * card gives; a word start marker becomes a leading space.
 */
export function tdtResult(
  tokens: readonly TdtToken[],
  vocabulary: Vocabulary,
  frameMs: number,
): RecognizerResult {
  const kept = tokens.filter((token) => {
    const piece = vocabulary[token.id];
    return piece !== undefined && !CONTROL_PIECE.test(piece);
  });
  const pieces = kept.map((token) => vocabulary[token.id]!.replaceAll(WORD_START, " "));
  return {
    text: pieces.join("").trim(),
    tokens: pieces,
    timestamps: kept.map((token) => (token.frame * frameMs) / 1_000),
    ys_log_probs: kept.map((token) => token.logProb),
  };
}
