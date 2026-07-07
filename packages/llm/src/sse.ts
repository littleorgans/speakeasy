/**
 * Minimal Server-Sent Events reader for the OpenAI-compatible streaming shape.
 *
 * Pure over a stream of text chunks so it is unit-testable without a network:
 * it buffers across chunk boundaries, splits on newlines, and yields the payload
 * of each `data:` field. Interpreting a payload (delta / [DONE] / error) is the
 * caller's job, keeping this vendor-neutral.
 */
export async function* parseSSEStream(
  chunks: AsyncIterable<string>,
): AsyncGenerator<string> {
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += chunk;
    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      const payload = readDataLine(line);
      if (payload !== undefined) {
        yield payload;
      }
    }
  }
  // Flush a final line that arrived without a trailing newline.
  const payload = readDataLine(buffer);
  if (payload !== undefined) {
    yield payload;
  }
}

/** Return the payload of a `data:` line, or undefined for any other line. */
function readDataLine(line: string): string | undefined {
  const trimmed = line.replace(/\r$/, "");
  if (!trimmed.startsWith("data:")) {
    return undefined;
  }
  return trimmed.slice("data:".length).trimStart();
}
