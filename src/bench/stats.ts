/** Small numeric helpers shared by the bench harness. */

export function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot compute median of an empty array");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle]!;
  }
  return (sorted[middle - 1]! + sorted[middle]!) / 2;
}

export function medianOptional(
  values: Array<number | undefined>,
): number | undefined {
  const present = values.filter(
    (value): value is number => value !== undefined,
  );
  return present.length > 0 ? median(present) : undefined;
}
