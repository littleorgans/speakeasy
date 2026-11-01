import assert from "node:assert/strict";
import { test } from "node:test";
import { decodePcm16 } from "./pcm.ts";

test("decodePcm16 preserves the Realtime signed little endian conversion", () => {
  const bytes = Buffer.from([0, 128, 255, 127, 0, 0, 0, 64, 0, 192, 1, 0, 255, 255]);
  const base64 = bytes.toString("base64");
  assert.deepEqual([...decodePcm16(base64)], [-1, 32767 / 32768, 0, 0.5, -0.5, 1 / 32768, -1 / 32768]);
  assert.equal(bytes.toString("base64"), base64);
  assert.deepEqual(decodePcm16(""), new Float32Array());
});

test("decodePcm16 rejects malformed base64 and incomplete PCM samples", () => {
  for (const value of ["!AAA", "AA A=", "AA\nA=", "AAA", "AA==", "A===", "AAB=", "AAA=trailing"]) {
    assert.throws(() => decodePcm16(value), /Invalid PCM16/);
  }
});
