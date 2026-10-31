import type { MelConfig } from "./ort-tdt-models.ts";

/**
 * Fluid Speech Mini front end, ported from the pinned runner
 * (fluid-speech-mini-onnx transcribe.py `log_mel`, `Model.subsampled` and
 * `Model.encode`): the Hugging Face ParakeetFeatureExtractor recipe of
 * pre-emphasis, a centred 512-point STFT with a symmetric Hann window, power
 * through a mel filter bank, log(x + 2^-24), then per band normalisation over
 * the valid frames. Precision follows the reference: float64 FFT, float32
 * everywhere else.
 */

const LOG_ZERO_GUARD = 2 ** -24;
const NORM_EPS = 1e-5;

export type MelFilters = {
  bands: number;
  bins: number;
  /** Row major [bands, bins]. */
  weights: Float32Array;
};

export type LogMel = {
  /** Row major [frames, bands]; frames past `valid` are zero. */
  features: Float32Array;
  frames: number;
  valid: number;
};

export function logMel(
  audio: Float32Array,
  config: MelConfig,
  filters: MelFilters,
): LogMel {
  const { nFft, hop, win, preemphasis } = config;
  const bins = nFft / 2 + 1;
  if (filters.bins !== bins) {
    throw new Error(`Mel filters have ${filters.bins} bins; nFft ${nFft} needs ${bins}`);
  }
  const x = preemphasize(audio, preemphasis);
  const window = hannWindow(nFft, win);
  const half = nFft / 2;
  const frames = 1 + Math.floor(x.length / hop);
  const valid = Math.min(Math.floor(x.length / hop), frames);
  const bands = filters.bands;
  const mel = new Float32Array(frames * bands);
  const fft = fftPlan(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const power = new Float32Array(bins);
  const spans = filterSpans(filters);
  for (let frame = 0; frame < frames; frame += 1) {
    // np.pad(x, nFft / 2): zeros outside the signal.
    const start = frame * hop - half;
    for (let n = 0; n < nFft; n += 1) {
      const index = start + n;
      re[n] = index >= 0 && index < x.length ? x[index]! * window[n]! : 0;
      im[n] = 0;
    }
    fft(re, im);
    for (let k = 0; k < bins; k += 1) {
      power[k] = re[k]! * re[k]! + im[k]! * im[k]!;
    }
    for (let band = 0; band < bands; band += 1) {
      const [from, to] = spans[band]!;
      let sum = 0;
      for (let k = from; k < to; k += 1) {
        sum += power[k]! * filters.weights[band * bins + k]!;
      }
      mel[frame * bands + band] = Math.log(Math.fround(sum) + LOG_ZERO_GUARD);
    }
  }
  normalize(mel, bands, frames, valid);
  return { features: mel, frames, valid };
}

function preemphasize(audio: Float32Array, coefficient: number): Float32Array {
  if (!coefficient || audio.length === 0) {
    return audio;
  }
  const out = new Float32Array(audio.length);
  const scale = Math.fround(coefficient);
  out[0] = audio[0]!;
  for (let n = 1; n < audio.length; n += 1) {
    out[n] = audio[n]! - Math.fround(scale * audio[n - 1]!);
  }
  return out;
}

/** Symmetric Hann of `win` samples, centred in `nFft`, rounded to float32. */
function hannWindow(nFft: number, win: number): Float64Array {
  const window = new Float64Array(nFft);
  const left = Math.floor((nFft - win) / 2);
  for (let n = 0; n < win; n += 1) {
    window[left + n] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (win - 1)));
  }
  return window;
}

/** Each band's nonzero bin range: triangular banks are sparse. */
function filterSpans(filters: MelFilters): [number, number][] {
  const spans: [number, number][] = [];
  for (let band = 0; band < filters.bands; band += 1) {
    const row = filters.weights.subarray(band * filters.bins, (band + 1) * filters.bins);
    let from = row.findIndex((weight) => weight !== 0);
    let to = row.findLastIndex((weight) => weight !== 0) + 1;
    if (from < 0) {
      from = 0;
      to = 0;
    }
    spans.push([from, to]);
  }
  return spans;
}

/** Per band (x - mean) / (std + eps) over valid frames, sample variance. */
function normalize(mel: Float32Array, bands: number, frames: number, valid: number): void {
  for (let band = 0; band < bands; band += 1) {
    let sum = 0;
    for (let frame = 0; frame < valid; frame += 1) {
      sum += mel[frame * bands + band]!;
    }
    const mean = Math.fround(sum / valid);
    let squares = 0;
    for (let frame = 0; frame < valid; frame += 1) {
      const delta = mel[frame * bands + band]! - mean;
      squares += delta * delta;
    }
    const std = Math.fround(Math.sqrt(squares / Math.max(valid - 1, 1)));
    for (let frame = 0; frame < frames; frame += 1) {
      const index = frame * bands + band;
      mel[index] = frame < valid ? (mel[index]! - mean) / (std + NORM_EPS) : 0;
    }
  }
}

type Fft = (re: Float64Array, im: Float64Array) => void;

/** In place iterative radix 2 complex FFT of a power of two size. */
function fftPlan(size: number): Fft {
  if (size < 2 || (size & (size - 1)) !== 0) {
    throw new Error(`FFT size must be a power of two, received ${size}`);
  }
  const reversed = new Uint32Array(size);
  const bits = Math.log2(size);
  for (let index = 0; index < size; index += 1) {
    let value = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      value |= ((index >> bit) & 1) << (bits - 1 - bit);
    }
    reversed[index] = value;
  }
  const cos = new Float64Array(size / 2);
  const sin = new Float64Array(size / 2);
  for (let index = 0; index < size / 2; index += 1) {
    cos[index] = Math.cos((2 * Math.PI * index) / size);
    sin[index] = -Math.sin((2 * Math.PI * index) / size);
  }
  return (re, im) => {
    for (let index = 0; index < size; index += 1) {
      const other = reversed[index]!;
      if (other > index) {
        [re[index], re[other]] = [re[other]!, re[index]!];
        [im[index], im[other]] = [im[other]!, im[index]!];
      }
    }
    for (let length = 2; length <= size; length *= 2) {
      const stride = size / length;
      const half = length / 2;
      for (let start = 0; start < size; start += length) {
        for (let k = 0; k < half; k += 1) {
          const wr = cos[k * stride]!;
          const wi = sin[k * stride]!;
          const a = start + k;
          const b = a + half;
          const tr = re[b]! * wr - im[b]! * wi;
          const ti = re[b]! * wi + im[b]! * wr;
          re[b] = re[a]! - tr;
          im[b] = im[a]! - ti;
          re[a] = re[a]! + tr;
          im[a] = im[a]! + ti;
        }
      }
    }
  };
}

/** Output frames of `layers` stride 2 convolutions (reference `subsampled`). */
export function subsampledFrames(
  frames: number,
  subsampling: { kernel: number; stride: number; layers: number },
): number {
  const pad = Math.floor((subsampling.kernel - 1) / 2) * 2 - subsampling.kernel;
  let out = frames;
  for (let layer = 0; layer < subsampling.layers; layer += 1) {
    out = Math.floor((out + pad) / subsampling.stride + 1);
  }
  return out;
}

export type FrontendChunk = {
  /** Feature frame range fed to the front end. */
  start: number;
  end: number;
  /** Leading output frames to drop (one frame of left context). */
  skip: number;
  /** Output frames kept. */
  count: number;
};

/**
 * Time chunks for the convolution front end (reference `encode`): each
 * chunk carries one output frame of context on both sides, which is dropped,
 * so the concatenation equals one whole clip pass with smaller activations.
 */
export function frontendChunks(
  frames: number,
  chunkFrames: number,
  subsampling: { kernel: number; stride: number; layers: number },
): FrontendChunk[] {
  const factor = subsampling.stride ** subsampling.layers;
  const total = subsampledFrames(frames, subsampling);
  const chunks: FrontendChunk[] = [];
  for (let first = 0; first < total; first += chunkFrames) {
    const last = Math.min(first + chunkFrames, total);
    chunks.push({
      start: first ? factor * (first - 1) : 0,
      end: Math.min(frames, factor * (last + 1)),
      skip: first ? 1 : 0,
      count: last - first,
    });
  }
  return chunks;
}

/** Parse a little endian float32 or float64 2D numpy .npy file. */
export function parseMelFilters(bytes: Uint8Array): MelFilters {
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (bytes[0] !== 0x93 || magic !== "NUMPY") {
    throw new Error("Mel filters are not a .npy file");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6]!;
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(
    bytes.subarray(headerStart, headerStart + headerLength),
  );
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const shape = /'shape':\s*\((\d+),\s*(\d+)\)/.exec(header);
  if (/'fortran_order':\s*True/.test(header) || !shape) {
    throw new Error(`Unsupported .npy layout: ${header.trim()}`);
  }
  const bands = Number(shape[1]);
  const bins = Number(shape[2]);
  const offset = headerStart + headerLength;
  const weights = new Float32Array(bands * bins);
  const width = descr === "<f4" ? 4 : descr === "<f8" ? 8 : 0;
  if (!width) {
    throw new Error(`Unsupported .npy dtype ${descr ?? "unknown"}`);
  }
  for (let index = 0; index < weights.length; index += 1) {
    const at = offset + index * width;
    weights[index] = width === 4 ? view.getFloat32(at, true) : view.getFloat64(at, true);
  }
  return { bands, bins, weights };
}
