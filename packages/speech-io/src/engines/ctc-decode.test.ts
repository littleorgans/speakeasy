import assert from "node:assert/strict";
import { test } from "node:test";
import { ByteLevelVocab, greedyCtc } from "./ctc-decode.ts";
import { logitsFor, TINY_TOKENIZER } from "./ctc-test-support.ts";
import { resultDetail } from "./result.ts";

const vocab = ByteLevelVocab.fromTokenizerJson(TINY_TOKENIZER, "<|blank|>");

test("greedy CTC groups repeats, drops blanks and keeps blank separated repeats", () => {
  const result = greedyCtc(logitsFor([0, 1, 1, 0, 2, 2, 0, 2, 3, 0], vocab.size), vocab, 0.08);
  assert.equal(result.text, "open the the file");
  assert.deepEqual(result.tokens, ["open", " the", " the", " file"]);
  assert.deepEqual(
    result.timestamps?.map((seconds) => Math.round(seconds * 1_000)),
    [80, 320, 560, 640],
  );
});

test("each token carries the log softmax of its first frame", () => {
  const result = greedyCtc(logitsFor([1], vocab.size, 5), vocab, 0.08);
  const expected = 5 - Math.log(Math.exp(5) + (vocab.size - 1));
  assert.ok(Math.abs(result.ys_log_probs![0]! - expected) < 1e-6);
});

test("all blank frames decode to no text", () => {
  const result = greedyCtc(logitsFor([0, 0, 0], vocab.size), vocab, 0.08);
  assert.equal(result.text, "");
  assert.deepEqual(result.tokens, []);
});

test("bytes join across tokens before UTF-8 decoding", () => {
  assert.equal(vocab.decode([1, 5, 6, 4]), "opené's");
});

test("greedy CTC token text joins bytes across tokens before UTF-8 decoding", () => {
  const result = greedyCtc(logitsFor([5, 6], vocab.size), vocab, 0.08);
  assert.equal(result.text, "é");
  assert.ok(result.tokens!.every((token) => !token.includes("\uFFFD")));
  assert.equal(result.tokens!.join(""), result.text);
  const detail = resultDetail(result);
  assert.deepEqual(
    detail.tokens.map((token) => [token.text, token.startMs]),
    [["", 0], ["é", 80]],
  );
  assert.deepEqual(detail.tokens.map((token) => token.logProb), result.ys_log_probs);
});

test("logits that disagree with the vocabulary are refused", () => {
  assert.throws(() => greedyCtc(logitsFor([1], vocab.size + 1), vocab, 0.08), /do not match/);
});

test("only a BPE tokenizer with a ByteLevel decoder and the blank token loads", () => {
  const wordPiece = JSON.stringify({ model: { type: "WordPiece", vocab: {} }, decoder: { type: "WordPiece" } });
  assert.throws(() => ByteLevelVocab.fromTokenizerJson(wordPiece, "<|blank|>"), /BPE model/);
  assert.throws(() => ByteLevelVocab.fromTokenizerJson(TINY_TOKENIZER, "<pad>"), /no <pad> token/);
  const gappy = JSON.stringify({ model: { type: "BPE", vocab: { "<|blank|>": 0, a: 2 } }, decoder: { type: "ByteLevel" } });
  assert.throws(() => ByteLevelVocab.fromTokenizerJson(gappy, "<|blank|>"), /dense/);
});
