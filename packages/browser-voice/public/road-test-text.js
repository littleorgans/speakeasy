/** Plain text for the road test page: durations, sizes and model cards. */

export function formatDuration(ms) {
  if (typeof ms !== "number" || !Number.isFinite(ms) || ms < 0) return null;
  return ms < 1_000 ? `${Math.round(ms)} ms` : `${(ms / 1_000).toFixed(2)} s`;
}

export function formatBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return null;
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

/**
 * Where it runs · size · runtime. The name is the list entry itself; whether
 * words show while you speak is the Live words choice, said once there.
 */
export function inputCard(setup) {
  const ready = setup.readiness?.ready === true;
  return [
    "Local",
    formatBytes(setup.readiness?.sizeBytes) ?? "size unknown",
    setup.card?.runtime ?? "runtime unknown",
    ready ? "" : `needs setup: ${setup.readiness?.reason ?? "not configured"}`,
  ].filter(Boolean).join(" · ");
}

export function outputCard(model) {
  return [
    model.provider?.kind === "local" ? "Local" : `Provider ${model.provider?.kind}`,
    formatBytes(model.sizeBytes) ?? "size unknown",
    model.runtime ?? "runtime unknown",
    model.streaming === "sentence" ? "streams by sentence" : model.streaming === "pcm" ? "streams audio" : "",
    model.readiness?.state === "ready" ? "" : `needs setup: ${model.readiness?.reason ?? model.readiness?.state}`,
  ].filter(Boolean).join(" · ");
}

/**
 * What makes a take's live words, from the stream and settings the host
 * applied: "Live words: Parakeet Ultra MLX stream, updates every 800 ms, the
 * last 5.12 s may change." A final from another model replaces them, and a
 * final model that cannot stream is named as giving final text only.
 */
export function liveLine(live, final) {
  const finalOnly = final?.card?.transcript === "final";
  if (!live) return finalOnly ? `Live words off. ${final.label} gives final text only.` : "Live words off.";
  const phrases = live.stream.settings.map((setting) =>
    setting.options.find((option) => option.value === live.settings[setting.id])?.phrase).filter(Boolean);
  const replaced = final && final.id !== live.setup.id
    ? ` ${final.label} writes the final text${finalOnly ? " only" : ""}.` : "";
  return `Live words: ${[`${live.setup.label} stream`, ...phrases].join(", ")}.${replaced}`;
}

/** The browser's reported output path delays in ms; null where it reports none. */
export function outputLatency(context) {
  const ms = (seconds) => Number.isFinite(seconds) ? seconds * 1_000 : null;
  return context ? { outputMs: ms(context.outputLatency), baseMs: ms(context.baseLatency) } : null;
}

/** Reads an NDJSON comparison stream, calling onEvent for each line. */
export async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  for (;;) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    let newline;
    while ((newline = pending.indexOf("\n")) !== -1) {
      const line = pending.slice(0, newline);
      pending = pending.slice(newline + 1);
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (pending.trim()) onEvent(JSON.parse(pending));
}
