/** Radix 2 complex FFT shared by the owned mel frontends (Granite, TDT). */

export type Fft = (re: Float64Array, im: Float64Array) => void;

/** In place iterative radix 2 complex FFT of a power of two size. */
export function fftPlan(size: number): Fft {
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
