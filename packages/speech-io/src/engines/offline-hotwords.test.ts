import assert from "node:assert/strict";
import { test } from "node:test";
import { prepareOfflineHotwords } from "./offline-hotwords.ts";
import { SHERPA_MODELS, type NemoTransducerModel } from "./sherpa-models.ts";

function encode(pieces: string[], phrases: string[], firstScoredPieceId = 2) {
  const model: NemoTransducerModel = {
    ...SHERPA_MODELS["en-parakeet-ultra-int8"],
    hotwordTokenizer: { firstScoredPieceId },
  };
  return prepareOfflineHotwords(model,
    [...pieces, "<blk>"].map((piece, id) => `${piece} ${id}\n`).join(""),
    { phrases, score: 2 });
}

test("scored suffix encoding uses reconstructed negative scores rather than token IDs", () => {
  // ▁a + b = -6; ▁ + ab = -1. Longest prefix and positive IDs choose wrongly.
  const result = encode(["<unk>", "<s>", "▁", "ab", "▁a", "a", "b", "A", "."], ["ab", "A ab."]);
  assert.deepEqual(result.encodings, [["▁", "ab"], ["▁", "A", "▁", "ab", "."]]);
  assert.equal(result.vocabulary, "<unk>\t0\n<s>\t0\n▁\t-0\nab\t-1\n▁a\t-2\na\t-3\nb\t-4\nA\t-5\n.\t-6\n");
});

test("native equal scores choose the shorter first piece and preserve case and boundaries", () => {
  // ▁ + ab = -3; ▁a + b = -3. Native chooses the shorter ▁ prefix.
  const pieces = ["<unk>", "<s>", "▁", "▁a", "b", "ab", "A", ".", "é"];
  const result = encode(pieces, ["ab", "A\t ab.", "é"]);
  assert.deepEqual(result.encodings, [["▁", "ab"], ["▁", "A", "▁", "ab", "."], ["▁", "é"]]);
});

test("a native fallback route fails even when another complete segmentation exists", () => {
  // The zero cost suffix fallback makes ▁a + unknown win over ▁ + ab.
  assert.throws(() => encode(["<unk>", "<s>", "▁a", "▁", "ab"], ["ab"]), /coverage/);
});

test("fallback bytes and selected control pieces are rejected even with vocabulary membership", () => {
  const pieces = ["<unk>", "<s>", "▁", "a", "<0xE2>"];
  assert.throws(() => encode(pieces, ["a☃"]), /coverage/);
  assert.throws(() => encode(pieces, ["a<0xE2>"]), /control/);
});
