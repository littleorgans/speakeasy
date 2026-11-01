/** Base64 mono signed PCM16 little endian to Float32, with no source mutation. */
export function decodePcm16(base64: string): Float32Array {
  // Buffer.from silently ignores invalid base64 and trailing partial samples.
  // Reject those before using the existing Realtime responder's conversion.
  if (
    typeof base64 !== "string" ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(base64)
  ) {
    throw new Error("Invalid PCM16 base64");
  }
  const bytes = Buffer.from(base64, "base64");
  if (bytes.length % 2 !== 0 || bytes.toString("base64") !== base64) {
    throw new Error("Invalid PCM16 bytes");
  }
  const samples = new Float32Array(bytes.length / 2);
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = bytes.readInt16LE(i * 2) / 32768;
  }
  return samples;
}
