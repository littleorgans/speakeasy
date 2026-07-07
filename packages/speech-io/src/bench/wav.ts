import { readFile } from "node:fs/promises";

const RIFF = "RIFF";
const WAVE = "WAVE";
const FORMAT_CHUNK = "fmt ";
const DATA_CHUNK = "data";
const PCM_FORMAT = 1;
const FLOAT_FORMAT = 3;
const TARGET_SAMPLE_RATE = 16_000;
const TARGET_CHANNELS = 1;
const DEFAULT_FRAME_MS = 20;

export type WavAudio = {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  durationMs: number;
  samples: Float32Array;
  frames: Float32Array[];
};

type FormatChunk = {
  audioFormat: number;
  channels: number;
  sampleRate: number;
  byteRate: number;
  blockAlign: number;
  bitsPerSample: number;
};

export async function readWavFrames(
  path: string,
  frameMs = DEFAULT_FRAME_MS,
): Promise<WavAudio> {
  const bytes = await readFile(path);
  return decodeWavFrames(bytes, frameMs);
}

export function decodeWavFrames(
  bytes: Uint8Array,
  frameMs = DEFAULT_FRAME_MS,
): WavAudio {
  if (frameMs <= 0) {
    throw new Error(`frameMs must be positive, received ${frameMs}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assertRiffWave(view);

  let format: FormatChunk | undefined;
  let dataOffset: number | undefined;
  let dataLength: number | undefined;

  for (const chunk of iterChunks(view)) {
    if (chunk.id === FORMAT_CHUNK) {
      format = readFormatChunk(view, chunk.offset, chunk.size);
    } else if (chunk.id === DATA_CHUNK) {
      dataOffset = chunk.offset;
      dataLength = chunk.size;
    }
  }

  if (!format) {
    throw new Error("WAV is missing a fmt chunk");
  }
  if (dataOffset === undefined || dataLength === undefined) {
    throw new Error("WAV is missing a data chunk");
  }
  validateFormat(format);

  const samples = decodeSamples(view, dataOffset, dataLength, format);
  const frameSize = Math.round((format.sampleRate * frameMs) / 1_000);
  const frames = splitFrames(samples, frameSize);
  const durationMs = (samples.length / format.sampleRate) * 1_000;

  return {
    sampleRate: format.sampleRate,
    channels: format.channels,
    bitsPerSample: format.bitsPerSample,
    durationMs,
    samples,
    frames,
  };
}

const PCM16_HEADER_BYTES = 44;
const PCM16_BYTES_PER_SAMPLE = 2;

/** Encode Float32 samples as a mono 16-bit PCM WAV file. */
export function encodeWavPcm16(
  samples: Float32Array,
  sampleRate: number,
): Uint8Array {
  const dataBytes = samples.length * PCM16_BYTES_PER_SAMPLE;
  const bytes = new Uint8Array(PCM16_HEADER_BYTES + dataBytes);
  const view = new DataView(bytes.buffer);

  writeAscii(view, 0, RIFF);
  view.setUint32(4, PCM16_HEADER_BYTES - 8 + dataBytes, true);
  writeAscii(view, 8, WAVE);
  writeAscii(view, 12, FORMAT_CHUNK);
  view.setUint32(16, 16, true);
  view.setUint16(20, PCM_FORMAT, true);
  view.setUint16(22, TARGET_CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * TARGET_CHANNELS * PCM16_BYTES_PER_SAMPLE, true);
  view.setUint16(32, TARGET_CHANNELS * PCM16_BYTES_PER_SAMPLE, true);
  view.setUint16(34, PCM16_BYTES_PER_SAMPLE * 8, true);
  writeAscii(view, 36, DATA_CHUNK);
  view.setUint32(40, dataBytes, true);

  for (let index = 0; index < samples.length; index += 1) {
    view.setInt16(
      PCM16_HEADER_BYTES + index * PCM16_BYTES_PER_SAMPLE,
      Math.round(clampSample(samples[index]!) * 32_767),
      true,
    );
  }
  return bytes;
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function assertRiffWave(view: DataView): void {
  if (view.byteLength < 12) {
    throw new Error("WAV is too short to contain a RIFF header");
  }
  const riff = readAscii(view, 0, 4);
  const wave = readAscii(view, 8, 4);
  if (riff !== RIFF || wave !== WAVE) {
    throw new Error("Expected a RIFF/WAVE file");
  }
}

function* iterChunks(
  view: DataView,
): Generator<{ id: string; offset: number; size: number }> {
  let cursor = 12;
  while (cursor + 8 <= view.byteLength) {
    const id = readAscii(view, cursor, 4);
    const size = view.getUint32(cursor + 4, true);
    const offset = cursor + 8;
    const next = offset + size + (size % 2);
    if (offset + size > view.byteLength) {
      throw new Error(`WAV chunk ${id} exceeds file length`);
    }
    yield { id, offset, size };
    cursor = next;
  }
}

function readFormatChunk(
  view: DataView,
  offset: number,
  size: number,
): FormatChunk {
  if (size < 16) {
    throw new Error("WAV fmt chunk is too short");
  }
  return {
    audioFormat: view.getUint16(offset, true),
    channels: view.getUint16(offset + 2, true),
    sampleRate: view.getUint32(offset + 4, true),
    byteRate: view.getUint32(offset + 8, true),
    blockAlign: view.getUint16(offset + 12, true),
    bitsPerSample: view.getUint16(offset + 14, true),
  };
}

function validateFormat(format: FormatChunk): void {
  if (format.sampleRate !== TARGET_SAMPLE_RATE) {
    throw new Error(
      `Expected ${TARGET_SAMPLE_RATE} Hz WAV, received ${format.sampleRate} Hz`,
    );
  }
  if (format.channels !== TARGET_CHANNELS) {
    throw new Error(
      `Expected mono WAV, received ${format.channels} channels`,
    );
  }
  if (format.audioFormat !== PCM_FORMAT && format.audioFormat !== FLOAT_FORMAT) {
    throw new Error(
      `Expected PCM or float WAV, received format ${format.audioFormat}`,
    );
  }
  if (format.audioFormat === FLOAT_FORMAT && format.bitsPerSample !== 32) {
    throw new Error("Float WAV must use 32 bits per sample");
  }
  if (
    format.audioFormat === PCM_FORMAT &&
    ![8, 16, 24, 32].includes(format.bitsPerSample)
  ) {
    throw new Error(
      `Unsupported PCM bit depth ${format.bitsPerSample}; expected 8, 16, 24, or 32`,
    );
  }

  const expectedBlockAlign = (format.channels * format.bitsPerSample) / 8;
  if (format.blockAlign !== expectedBlockAlign) {
    throw new Error(
      `Invalid WAV blockAlign ${format.blockAlign}; expected ${expectedBlockAlign}`,
    );
  }
  const expectedByteRate = format.sampleRate * format.blockAlign;
  if (format.byteRate !== expectedByteRate) {
    throw new Error(
      `Invalid WAV byteRate ${format.byteRate}; expected ${expectedByteRate}`,
    );
  }
}

function decodeSamples(
  view: DataView,
  dataOffset: number,
  dataLength: number,
  format: FormatChunk,
): Float32Array {
  const bytesPerSample = format.bitsPerSample / 8;
  const sampleCount = Math.floor(dataLength / bytesPerSample);
  const samples = new Float32Array(sampleCount);

  for (let index = 0; index < sampleCount; index += 1) {
    const offset = dataOffset + index * bytesPerSample;
    samples[index] = readSample(view, offset, format);
  }

  return samples;
}

function readSample(
  view: DataView,
  offset: number,
  format: FormatChunk,
): number {
  if (format.audioFormat === FLOAT_FORMAT) {
    return clampSample(view.getFloat32(offset, true));
  }

  switch (format.bitsPerSample) {
    case 8:
      return clampSample((view.getUint8(offset) - 128) / 128);
    case 16:
      return clampSample(view.getInt16(offset, true) / 32_768);
    case 24:
      return clampSample(readInt24(view, offset) / 8_388_608);
    case 32:
      return clampSample(view.getInt32(offset, true) / 2_147_483_648);
    default:
      throw new Error(`Unsupported bit depth ${format.bitsPerSample}`);
  }
}

function readInt24(view: DataView, offset: number): number {
  const unsigned =
    view.getUint8(offset) |
    (view.getUint8(offset + 1) << 8) |
    (view.getUint8(offset + 2) << 16);
  return unsigned & 0x80_0000 ? unsigned | 0xff_00_00_00 : unsigned;
}

function splitFrames(
  samples: Float32Array,
  frameSize: number,
): Float32Array[] {
  const frames: Float32Array[] = [];
  for (let offset = 0; offset < samples.length; offset += frameSize) {
    const frame = samples.slice(offset, offset + frameSize);
    frames.push(
      frame.length === frameSize ? frame : padFrame(frame, frameSize),
    );
  }
  return frames;
}

function padFrame(frame: Float32Array, frameSize: number): Float32Array {
  const padded = new Float32Array(frameSize);
  padded.set(frame);
  return padded;
}

function clampSample(sample: number): number {
  return Math.max(-1, Math.min(1, sample));
}

function readAscii(view: DataView, offset: number, length: number): string {
  let text = "";
  for (let index = 0; index < length; index += 1) {
    text += String.fromCharCode(view.getUint8(offset + index));
  }
  return text;
}
