import assert from "node:assert/strict";
import { test } from "node:test";
import { decodeWavFrames, decodeWavSamples, encodeWavFloat32 } from "./wav.ts";

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function chunk(id: string, payload: Uint8Array): Uint8Array {
  const bytes = new Uint8Array(8 + payload.length + payload.length % 2);
  bytes.set(new TextEncoder().encode(id));
  view(bytes).setUint32(4, payload.length, true);
  bytes.set(payload, 8);
  return bytes;
}

function riff(...chunks: Uint8Array[]): Uint8Array {
  const bytes = new Uint8Array(12 + chunks.reduce((size, part) => size + part.length, 0));
  bytes.set(new TextEncoder().encode("RIFF"));
  view(bytes).setUint32(4, bytes.length - 8, true);
  bytes.set(new TextEncoder().encode("WAVE"), 8);
  let offset = 12;
  for (const part of chunks) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return bytes;
}

function floatFixture(): Uint8Array {
  return encodeWavFloat32(new Float32Array([0.25, -0.5, 2]), 16_000);
}

test("strict float replay preserves finite sample bits and count without padding", () => {
  const bits = new Uint32Array([
    0, 0x80000000, 0x3f800000, 0xbf800000, 0x3eaaaaab,
    0x40000000, 0xc0a00000, 1, 0x80000001, 0x007fffff,
    0x00800000, 0x7f7fffff, 0xff7fffff,
  ]);
  const sampleStorage = new Float32Array(bits.length + 2);
  new Uint32Array(sampleStorage.buffer).set(bits, 1);
  const samples = sampleStorage.subarray(1, -1);
  const encoded = encodeWavFloat32(samples, 16_000);
  // Neither sample views nor byte views need to start at their buffer's origin.
  const storage = new Uint8Array(encoded.length + 7).fill(0xff);
  storage.set(encoded, 3);
  const bytes = storage.subarray(3, 3 + encoded.length);
  const before = storage.slice();
  const decoded = decodeWavSamples(bytes);

  assert.equal(decoded.length, samples.length);
  assert.deepEqual(new Uint32Array(decoded.buffer, decoded.byteOffset, decoded.length), bits);
  assert.ok(Object.is(decoded[1], -0));
  assert.deepEqual(encodeWavFloat32(decoded, 16_000), encoded);
  decoded[0] = 0.5;
  assert.deepEqual(storage, before, "decoded samples own their storage");
});

test("strict float replay permits an empty take", () => {
  assert.equal(decodeWavSamples(encodeWavFloat32(new Float32Array(), 16_000)).length, 0);
});

test("legacy float decoding keeps clipping and padded frames", () => {
  const audio = decodeWavFrames(encodeWavFloat32(new Float32Array([2, -3, -0]), 16_000));
  assert.deepEqual(audio.samples, new Float32Array([1, -1, -0]));
  assert.equal(audio.durationMs, 3 / 16);
  assert.equal(audio.frames.length, 1);
  assert.equal(audio.frames[0]!.length, 320);
  assert.deepEqual(audio.frames[0]!.slice(0, 3), audio.samples);
  assert.ok(audio.frames[0]!.slice(3).every((sample) => sample === 0));
});

test("strict replay rejects nonfinite floats while legacy behavior remains", () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    const bytes = encodeWavFloat32(new Float32Array([0, value]), 16_000);
    assert.throws(() => decodeWavSamples(bytes), /nonfinite.*1/i);
    const legacy = decodeWavFrames(bytes).samples[1];
    assert.ok(Number.isNaN(value) ? Number.isNaN(legacy) : legacy === Math.sign(value));
  }
});

test("strict and legacy decoding retain PCM16 scaling and original sample count", () => {
  const format = floatFixture().slice(20, 36);
  view(format).setUint16(0, 1, true);
  view(format).setUint32(8, 32_000, true);
  view(format).setUint16(12, 2, true);
  view(format).setUint16(14, 16, true);
  const payload = new Uint8Array(6);
  [-32768, 0, 32767].forEach((sample, index) => view(payload).setInt16(index * 2, sample, true));
  const bytes = riff(chunk("fmt ", format), chunk("data", payload));
  const expected = new Float32Array([-1, 0, 32767 / 32768]);
  assert.deepEqual(decodeWavSamples(bytes), expected);
  const legacy = decodeWavFrames(bytes, 10);
  assert.deepEqual(legacy.samples, expected);
  assert.equal(legacy.frames[0]!.length, 160);

  const partial = riff(chunk("fmt ", format), chunk("data", payload.slice(0, 3)));
  assert.throws(() => decodeWavSamples(partial), /whole.*sample/i);
  assert.deepEqual(decodeWavFrames(partial).samples, expected.slice(0, 1));
});

test("strict replay handles odd unknown chunks with RIFF padding", () => {
  const bytes = floatFixture();
  const rebuilt = riff(
    chunk("JUNK", new Uint8Array([7])),
    bytes.slice(12, 36),
    chunk("LIST", new Uint8Array([1, 2, 3])),
    bytes.slice(36),
    chunk("JUNK", new Uint8Array([4])),
  );
  assert.deepEqual(decodeWavSamples(rebuilt), new Float32Array([0.25, -0.5, 2]));
});

test("strict replay rejects inconsistent RIFF bounds without changing legacy tolerance", () => {
  for (const delta of [-4, 4]) {
    const bytes = floatFixture();
    view(bytes).setUint32(4, bytes.length - 8 + delta, true);
    assert.throws(() => decodeWavSamples(bytes), /RIFF.*length/);
    assert.equal(decodeWavFrames(bytes).samples.length, 3);
  }
});

test("strict replay rejects invalid RIFF headers", () => {
  assert.throws(() => decodeWavSamples(new Uint8Array(11)), /too short/);
  for (const offset of [0, 8]) {
    const bytes = floatFixture();
    bytes[offset] = 0;
    assert.throws(() => decodeWavSamples(bytes), /RIFF\/WAVE/);
  }
});

test("strict replay rejects duplicate format or data chunks", () => {
  const bytes = floatFixture();
  const format = bytes.slice(12, 36);
  const data = bytes.slice(36);
  assert.throws(() => decodeWavSamples(riff(format, format, data)), /duplicate.*fmt/i);
  assert.throws(() => decodeWavSamples(riff(format, data, data)), /duplicate.*data/i);
  // Legacy decoding continues to select the last data chunk.
  const replacement = encodeWavFloat32(new Float32Array([-0.25]), 16_000).slice(36);
  assert.deepEqual(decodeWavFrames(riff(format, data, replacement)).samples, new Float32Array([-0.25]));
});

test("strict replay rejects missing or short required chunks", () => {
  const bytes = floatFixture();
  assert.throws(() => decodeWavSamples(riff(bytes.slice(36))), /missing a fmt/);
  assert.throws(() => decodeWavSamples(riff(bytes.slice(12, 36))), /missing a data/);
  assert.throws(() => decodeWavSamples(riff(chunk("fmt ", new Uint8Array(14)), bytes.slice(36))), /fmt chunk is too short/);
});

test("strict replay rejects truncated chunks, headers and missing odd padding", () => {
  const bytes = floatFixture();
  const oversizedData = bytes.slice();
  view(oversizedData).setUint32(40, 16, true);
  assert.throws(() => decodeWavSamples(oversizedData), /chunk data exceeds/);

  const oversizedFormat = bytes.slice();
  view(oversizedFormat).setUint32(16, bytes.length, true);
  assert.throws(() => decodeWavSamples(oversizedFormat), /chunk fmt .*exceeds/);

  const partialHeader = riff(bytes.slice(12), new Uint8Array([1, 2, 3]));
  assert.throws(() => decodeWavSamples(partialHeader), /chunk header/);
  assert.equal(decodeWavFrames(partialHeader).samples.length, 3);

  const missingPad = riff(bytes.slice(12), chunk("JUNK", new Uint8Array([1])).slice(0, -1));
  assert.throws(() => decodeWavSamples(missingPad), /padding/);
});

test("strict replay rejects partial float samples instead of dropping bytes", () => {
  const bytes = floatFixture();
  const partial = riff(bytes.slice(12, 36), chunk("data", bytes.slice(44, 49)));
  assert.throws(() => decodeWavSamples(partial), /whole.*sample/i);
  assert.deepEqual(decodeWavFrames(partial).samples, new Float32Array([0.25]));
});

test("strict replay validates declared format and sample layout", () => {
  const mutations: [number, number, number, RegExp][] = [
    [24, 8000, 4, /16000 Hz/],
    [22, 2, 2, /mono/],
    [20, 6, 2, /PCM or float/],
    [34, 64, 2, /32 bits/],
    [32, 2, 2, /blockAlign/],
    [28, 1, 4, /byteRate/],
  ];
  for (const [offset, value, width, message] of mutations) {
    const bytes = floatFixture();
    if (width === 4) view(bytes).setUint32(offset, value, true);
    else view(bytes).setUint16(offset, value, true);
    assert.throws(() => decodeWavSamples(bytes), message);
  }
});
