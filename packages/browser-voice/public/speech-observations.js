/** Durations keep their own boundaries; missing values never become zero. */
export function formatMs(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${Math.round(value)} ms` : "Unavailable";
}

export function formatObservation(value) {
  if (!value || value.status === "unavailable") return `Unavailable${value?.reason ? `: ${value.reason}` : ""}`;
  const duration = formatMs(value.ms);
  if (duration === "Unavailable" || typeof value.boundary !== "string" || !value.boundary.trim()) return "Unavailable";
  if (value.status === "measured" && ["server", "worker", "browser"].includes(value.clock)) {
    return `${duration} (measured; ${value.clock}; ${value.boundary})`;
  }
  if (value.status === "provider-reported" && typeof value.metric === "string" && value.metric.trim()) {
    return `${duration} (provider reported; ${value.metric}; ${value.boundary})`;
  }
  return "Unavailable";
}

export function transcriptStatus(requested, returned) {
  return typeof returned !== "string" ? "unavailable" : returned === requested ? "same" : "different";
}
