/**
 * Granite Speech 5.0 CTC feature frontend: 16 kHz audio to the stacked log
 * mel plus delta features the ONNX graph takes as `input_features`.
 *
 * Port of transformers v5.16.0
 * `models/granite_speech5/feature_extraction_granite_speech5.py`
 * `GraniteSpeech5FeatureExtractor._extract_features` (same math as the IBM
 * checkpoint's `processing_ctc_conformer.py` `CtcConformerProcessor._frontend`)
 * with the defaults in the export's `preprocessor_config.json`, and the
 * torchaudio defaults it relies on: `MelSpectrogram` (periodic Hann window
 * centred in the FFT, reflect padded centred frames, power 2, HTK mel scale,
 * no filter norm) and `compute_deltas` (replicate padded, win 3).
 *
 * Per utterance: right pad to an even mel frame count, log10 mel floored 8
 * decades below the utterance maximum, scaled `/4 + 1`, deltas appended, and
 * pairs of frames stacked into one 320 wide feature frame.
 */

const SAMPLE_RATE = 16_000;
const N_FFT = 512;
const WIN_LENGTH = 400;
const HOP_LENGTH = 160;
const N_MELS = 80;
const STACK = 2;
const LOGMEL_FLOOR = 8;
const MEL_MIN_POWER = 1e-10;
const N_FREQS = N_FFT / 2 + 1;
const FRAME_WIDTH = N_MELS * 2;

export const GRANITE_FEATURE_DIM = FRAME_WIDTH * STACK;
/** Shortest input the graph accepts (its README graph contract). */
export const GRANITE_MIN_FEATURE_FRAMES = 4;
/** Audio one feature frame spans: two 10 ms mel frames. */
export const GRANITE_FEATURE_FRAME_MS = (STACK * HOP_LENGTH * 1_000) / SAMPLE_RATE;

export type GraniteFeatures = {
  /** Row major `[frames, GRANITE_FEATURE_DIM]`. */
  data: Float32Array;
  frames: number;
};

/** Feature frames `audio` yields: ceil(mel frames / 2). */
export function graniteFeatureFrames(samples: number): number {
  return Math.ceil(Math.floor(samples / HOP_LENGTH) / STACK);
}

export function graniteFeatures(audio: Float32Array): GraniteFeatures {
  const frames = graniteFeatureFrames(audio.length);
  if (frames < GRANITE_MIN_FEATURE_FRAMES) {
    throw new Error(
      `Granite needs at least ${GRANITE_MIN_FEATURE_FRAMES} feature frames; ${audio.length} samples give ${frames}`,
    );
  }
  const melFrames = frames * STACK;
  const padded = rightPad(audio, (melFrames - 1) * HOP_LENGTH + 1);
  const logmel = logMel(padded, melFrames);
  const data = new Float32Array(frames * GRANITE_FEATURE_DIM);
  for (let frame = 0; frame < melFrames; frame += 1) {
    const out = frame * FRAME_WIDTH;
    const previous = Math.max(frame - 1, 0) * N_MELS;
    const next = Math.min(frame + 1, melFrames - 1) * N_MELS;
    for (let mel = 0; mel < N_MELS; mel += 1) {
      data[out + mel] = logmel[frame * N_MELS + mel]!;
      data[out + N_MELS + mel] = (logmel[next + mel]! - logmel[previous + mel]!) / 2;
    }
  }
  return { data, frames };
}

function rightPad(audio: Float32Array, length: number): Float32Array {
  if (audio.length >= length) {
    return audio;
  }
  const padded = new Float32Array(length);
  padded.set(audio);
  return padded;
}

/** Floored, scaled log10 mel energies, row major `[melFrames, N_MELS]`. */
function logMel(audio: Float32Array, melFrames: number): Float64Array {
  const logmel = new Float64Array(melFrames * N_MELS);
  const real = new Float64Array(N_FFT);
  const imag = new Float64Array(N_FFT);
  const power = new Float64Array(N_FREQS);
  let max = -Infinity;
  for (let frame = 0; frame < melFrames; frame += 1) {
    const start = frame * HOP_LENGTH - N_FFT / 2;
    for (let index = 0; index < N_FFT; index += 1) {
      real[index] = reflect(audio, start + index) * WINDOW[index]!;
      imag[index] = 0;
    }
    fft(real, imag);
    for (let bin = 0; bin < N_FREQS; bin += 1) {
      power[bin] = real[bin]! * real[bin]! + imag[bin]! * imag[bin]!;
    }
    for (let mel = 0; mel < N_MELS; mel += 1) {
      const filter = MEL_FILTERS[mel]!;
      let energy = 0;
      for (let bin = 0; bin < filter.weights.length; bin += 1) {
        energy += filter.weights[bin]! * power[filter.start + bin]!;
      }
      const value = Math.log10(Math.max(energy, MEL_MIN_POWER));
      logmel[frame * N_MELS + mel] = value;
      max = Math.max(max, value);
    }
  }
  const floor = max - LOGMEL_FLOOR;
  for (let index = 0; index < logmel.length; index += 1) {
    logmel[index] = Math.max(logmel[index]!, floor) / 4 + 1;
  }
  return logmel;
}

/** torch reflect padding: mirror about the first and last sample. */
function reflect(audio: Float32Array, index: number): number {
  const last = audio.length - 1;
  const mirrored = index < 0 ? -index : index > last ? 2 * last - index : index;
  return audio[mirrored]!;
}

/** Periodic Hann of WIN_LENGTH, zero padded to N_FFT and centred, as torch.stft. */
const WINDOW = (() => {
  const window = new Float64Array(N_FFT);
  const offset = (N_FFT - WIN_LENGTH) / 2;
  for (let index = 0; index < WIN_LENGTH; index += 1) {
    window[offset + index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / WIN_LENGTH);
  }
  return window;
})();

type MelFilter = { start: number; weights: Float64Array };

/** torchaudio `melscale_fbanks`, HTK scale, no norm, kept as nonzero spans. */
const MEL_FILTERS: MelFilter[] = (() => {
  const toMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const toHz = (mel: number) => 700 * (10 ** (mel / 2595) - 1);
  const melMax = toMel(SAMPLE_RATE / 2);
  const points = Array.from({ length: N_MELS + 2 }, (_, index) =>
    toHz((melMax * index) / (N_MELS + 1)),
  );
  return Array.from({ length: N_MELS }, (_, mel) => {
    const [low, centre, high] = [points[mel]!, points[mel + 1]!, points[mel + 2]!];
    const weights: number[] = [];
    let start = -1;
    for (let bin = 0; bin < N_FREQS; bin += 1) {
      const hz = (bin * SAMPLE_RATE) / 2 / (N_FREQS - 1);
      const weight = Math.max(0, Math.min((hz - low) / (centre - low), (high - hz) / (high - centre)));
      if (weight > 0) {
        start = start < 0 ? bin : start;
        weights[bin - start] = weight;
      }
    }
    return { start: Math.max(start, 0), weights: Float64Array.from(weights, (weight) => weight ?? 0) };
  });
})();

const TWIDDLE_COS = Float64Array.from({ length: N_FFT / 2 }, (_, index) =>
  Math.cos((-2 * Math.PI * index) / N_FFT),
);
const TWIDDLE_SIN = Float64Array.from({ length: N_FFT / 2 }, (_, index) =>
  Math.sin((-2 * Math.PI * index) / N_FFT),
);

const BIT_REVERSED = (() => {
  const bits = Math.log2(N_FFT);
  return Uint16Array.from({ length: N_FFT }, (_, index) => {
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | ((index >> bit) & 1);
    }
    return reversed;
  });
})();

/** In place iterative radix 2 FFT of length N_FFT. */
function fft(real: Float64Array, imag: Float64Array): void {
  for (let index = 0; index < N_FFT; index += 1) {
    const swap = BIT_REVERSED[index]!;
    if (swap > index) {
      [real[index], real[swap]] = [real[swap]!, real[index]!];
      [imag[index], imag[swap]] = [imag[swap]!, imag[index]!];
    }
  }
  for (let size = 2; size <= N_FFT; size *= 2) {
    const half = size / 2;
    const stride = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let offset = 0; offset < half; offset += 1) {
        const cos = TWIDDLE_COS[offset * stride]!;
        const sin = TWIDDLE_SIN[offset * stride]!;
        const even = start + offset;
        const odd = even + half;
        const oddReal = real[odd]! * cos - imag[odd]! * sin;
        const oddImag = real[odd]! * sin + imag[odd]! * cos;
        real[odd] = real[even]! - oddReal;
        imag[odd] = imag[even]! - oddImag;
        real[even] = real[even]! + oddReal;
        imag[even] = imag[even]! + oddImag;
      }
    }
  }
}
