import type { CtcLogits } from "./ctc-decode.ts";

/**
 * Fixtures for the CTC engine tests. Byte level tokens: "Ġ" is the space
 * byte; "Ã" and "©" are the two UTF-8 bytes of "é" (0xC3 0xA9).
 */
export const TINY_TOKENIZER = JSON.stringify({
  model: {
    type: "BPE",
    vocab: { "<|blank|>": 0, open: 1, "Ġthe": 2, "Ġfile": 3, "'s": 4, "Ã": 5, "©": 6 },
  },
  decoder: { type: "ByteLevel" },
});

/** One frame per id, that id scoring `peak` and every other id 0. */
export function logitsFor(ids: number[], size: number, peak = 5): CtcLogits {
  const data = new Float32Array(ids.length * size);
  ids.forEach((id, frame) => {
    data[frame * size + id] = peak;
  });
  return { data, frames: ids.length, vocab: size };
}
