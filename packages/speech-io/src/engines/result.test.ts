import assert from "node:assert/strict";
import { test } from "node:test";
import { resultDetail, resultText } from "./result.ts";

test("text is trimmed and stripped of control characters", () => {
  assert.equal(resultText({ text: "  open\u0000 the\u0007 file \n" }), "open the file");
  assert.equal(resultText({}), "");
});

test("tokens keep their leading space so words group on it", () => {
  const detail = resultDetail({
    text: " HELLO WORLD",
    tokens: [" HE", "LLO", " WOR", "LD"],
    timestamps: [0.12, 0.2, 0.48, 0.56],
    start_time: 1.5,
  });
  assert.equal(detail.text, "HELLO WORLD");
  assert.deepEqual(
    detail.tokens.map((token) => token.text),
    [" HE", "LLO", " WOR", "LD"],
  );
  const words = detail.tokens
    .map((token) => token.text)
    .join("")
    .trim()
    .split(" ");
  assert.deepEqual(words, ["HELLO", "WORLD"]);
  assert.deepEqual(
    detail.tokens.map((token) => token.startMs),
    [1_620, 1_700, 1_980, 2_060],
  );
});

test("streaming ys_probs and offline ys_log_probs become per token log probs", () => {
  const streaming = resultDetail({
    text: "hi",
    tokens: [" h", "i"],
    timestamps: [0, 0.04],
    ys_probs: [-0.1, -2.3],
  });
  assert.deepEqual(
    streaming.tokens.map((token) => token.logProb),
    [-0.1, -2.3],
  );
  const offline = resultDetail({
    text: "hi",
    tokens: [" h", "i"],
    timestamps: [0, 0.04],
    ys_log_probs: [-0.5, -0.25],
    durations: [0.04, 0.04],
  });
  assert.deepEqual(
    offline.tokens.map((token) => token.logProb),
    [-0.5, -0.25],
  );
});

test("log probs that do not line up with tokens are dropped, not guessed", () => {
  const detail = resultDetail({
    text: "hi",
    tokens: [" h", "i"],
    timestamps: [0, 0.04],
    ys_probs: [-0.1],
  });
  assert.deepEqual(
    detail.tokens.map((token) => token.logProb),
    [null, null],
  );
});

test("Moonshine shape: tokens without timestamps yield no tokens", () => {
  const detail = resultDetail({
    text: " Open the file.",
    tokens: [" Open", " the", " file", "."],
    timestamps: [],
  });
  assert.deepEqual(detail, { text: "Open the file.", tokens: [] });
});

test("silence yields empty text and no tokens", () => {
  assert.deepEqual(resultDetail({ text: "", tokens: [], timestamps: [] }), {
    text: "",
    tokens: [],
  });
});

test("token times are utterance offsets when the stream clock runs on", () => {
  const detail = resultDetail(
    { text: "hi", tokens: [" h", "i"], timestamps: [1.36, 1.4], start_time: 2.56 },
    3.2,
  );
  assert.deepEqual(
    detail.tokens.map((token) => token.startMs),
    [720, 760],
  );
});
