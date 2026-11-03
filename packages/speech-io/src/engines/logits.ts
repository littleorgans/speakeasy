/**
 * Selection and normalization over a range `[from, to)` of a logits row,
 * shared by the greedy decoders. Ties keep the first index; a NaN never wins
 * argMax unless it sits at `from`, and any NaN in range makes the log sums NaN.
 */

/** Index of the first maximum in `[from, to)`, relative to `from`; 0 when empty. */
export function argMax(values: Float32Array, from = 0, to = values.length): number {
  let best = from;
  for (let index = from + 1; index < to; index += 1) {
    if (values[index]! > values[best]!) {
      best = index;
    }
  }
  return best - from;
}

/** `log(sum(exp(values)))` over `[from, to)`. */
export function logSumExp(values: Float32Array, from = 0, to = values.length): number {
  const [max, sum] = shiftedExpSum(values, from, to);
  return max + Math.log(sum);
}

/** Log softmax over `[from, to)` of the value at `index`, relative to `from`. */
export function logSoftmaxAt(
  values: Float32Array,
  index: number,
  from = 0,
  to = values.length,
): number {
  const [max, sum] = shiftedExpSum(values, from, to);
  return values[from + index]! - max - Math.log(sum);
}

/** The range maximum and `sum(exp(value - max))`. */
function shiftedExpSum(values: Float32Array, from: number, to: number): [number, number] {
  let max = -Infinity;
  for (let index = from; index < to; index += 1) {
    max = Math.max(max, values[index]!);
  }
  let sum = 0;
  for (let index = from; index < to; index += 1) {
    sum += Math.exp(values[index]! - max);
  }
  return [max, sum];
}
