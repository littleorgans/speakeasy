import assert from "node:assert/strict";
import { test } from "node:test";
import {
  greedyTdt,
  parseTokenizerJson,
  parseVocabTxt,
  tdtResult,
  type TdtStepper,
} from "./ort-tdt-decode.ts";
import type { TdtDecoding } from "./ort-tdt-models.ts";

const DECODING: TdtDecoding = {
  vocabSize: 5,
  blankId: 4,
  durations: [0, 1, 2, 3, 4],
  maxSymbolsPerFrame: 2,
  frameMs: 80,
  predictor: { layers: 1, hidden: 1 },
};

/** Logits that pick `token` and the duration at `durationIndex`. */
function step(token: number, durationIndex: number, size = DECODING.vocabSize): Float32Array {
  const logits = new Float32Array(size + DECODING.durations.length);
  logits[token] = 10;
  logits[size + durationIndex] = 10;
  return logits;
}

function scripted(steps: Float32Array[]) {
  const calls: string[] = [];
  const stepper: TdtStepper = {
    joint: async (frame) => {
      calls.push(`joint ${frame}`);
      const next = steps.shift();
      if (!next) throw new Error("script exhausted");
      return next;
    },
    emit: async (token) => {
      calls.push(`emit ${token}`);
    },
  };
  return { calls, stepper };
}

test("greedyTdt follows token durations and steps past blanks", async () => {
  const { calls, stepper } = scripted([
    step(1, 2), // token 1 at frame 0, jump 2
    step(4, 0), // blank, duration 0: step one frame
    step(2, 0), // token 2 at frame 3, stay
    step(3, 1), // token 3 at frame 3, then frame 4
    step(4, 3), // blank jumps past the end
  ]);
  const tokens = await greedyTdt(stepper, 5, DECODING);
  assert.deepEqual(tokens.map((token) => [token.id, token.frame]), [[1, 0], [2, 3], [3, 3]]);
  assert.deepEqual(calls, [
    "joint 0", "emit 1", "joint 2", "joint 3", "emit 2", "joint 3", "emit 3", "joint 4",
  ]);
});

test("greedyTdt advances after maxSymbolsPerFrame tokens on one frame", async () => {
  const { stepper } = scripted([step(1, 0), step(2, 0), step(3, 1)]);
  const tokens = await greedyTdt(stepper, 2, DECODING);
  assert.deepEqual(tokens.map((token) => [token.id, token.frame]), [[1, 0], [2, 0], [3, 1]]);
});

test("greedyTdt stops on the end of text token", async () => {
  const { stepper } = scripted([step(1, 1), step(0, 1)]);
  const tokens = await greedyTdt(stepper, 10, { ...DECODING, eosId: 0 });
  assert.deepEqual(tokens.map((token) => token.id), [1]);
});

test("greedyTdt reports the token's log softmax over token logits only", async () => {
  const logits = new Float32Array([0, 2, 0, 0, 0, 0, 0, 9, 0, 0]);
  const { stepper } = scripted([logits]);
  const [token] = await greedyTdt(stepper, 1, DECODING);
  const expected = 2 - Math.log(4 + Math.exp(2));
  assert.ok(Math.abs(token!.logProb - expected) < 1e-9);
});

test("tdtResult drops control pieces and marks word starts", () => {
  const vocabulary = ["<unk>", "▁Open", "ing", "<|endoftext|>", "▁the", ".", "<blk>"];
  const result = tdtResult(
    [0, 1, 2, 3, 4, 5].map((id, index) => ({ id, frame: index * 2, logProb: -id })),
    vocabulary,
    80,
  );
  assert.equal(result.text, "Opening the.");
  assert.deepEqual(result.tokens, [" Open", "ing", " the", "."]);
  assert.deepEqual(result.timestamps, [0.16, 0.32, 0.64, 0.8]);
  assert.deepEqual(result.ys_log_probs, [-1, -2, -4, -5]);
});

test("vocab.txt and tokenizer.json give the same pieces", () => {
  const fromTxt = parseVocabTxt("<unk> 0\n▁a 1\nb c 2\n<blk> 3\n");
  assert.deepEqual(fromTxt, ["<unk>", "▁a", "b c", "<blk>"]);
  const fromJson = parseTokenizerJson(
    JSON.stringify({ model: { vocab: { "<unk>": 0, "▁a": 1, "b c": 2 } } }),
  );
  assert.deepEqual(fromJson, fromTxt.slice(0, 3));
  assert.throws(() => parseTokenizerJson("{}"), /model.vocab/);
});
