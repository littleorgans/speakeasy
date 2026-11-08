import { trailItems, trailSummary } from "/live-trail.js";
import { formatDuration, liveSettingsNote } from "/road-test-text.js";
import { cleanedText, cleanupNote, cleanupTime } from "/text-cleanup.js";

/**
 * Take timings and earlier takes: the figures each take shows, the history
 * table's columns and a collapsed trail per earlier take.
 */

/** Earlier takes columns; the transcript spans a line of its own, and narrow screens label each cell with these. */
export const HISTORY_COLUMNS = [
  { label: "Speech in", cell: (run) => [run.setup.label, [finalNote(run), liveNote(run)].filter(Boolean).join(" · ")] },
  { label: "Voice", cell: (run) => [run.output ? run.voiceLabel : null] },
  { label: "Transcript", kind: "said-cell", cell: (run) => [cleanedText(run.cleanup) ?? (run.input.error || run.input.text || "Nothing heard"),
    cleanupNote(run.cleanup)], trail: true },
  { label: "Press to first words", kind: "num", cell: (run) => [formatDuration(pressToFirstWords(run))] },
  { label: "Release to text", kind: "num", cell: (run) => [formatDuration(run.input.finalMs),
    run.input.loadMs != null ? `Model start ${formatDuration(run.input.loadMs)}` : null] },
  { label: "Cleanup", kind: "num", cell: (run) => [cleanupTime(run.cleanup),
    run.cleanup?.loadMs != null ? `Model start ${formatDuration(run.cleanup.loadMs)}` : null] },
  { label: "Text to first audio", kind: "num", cell: (run) => [formatDuration(run.output?.firstAudioMs),
    run.output?.loadMs != null ? `Model start ${formatDuration(run.output.loadMs)}` : null] },
  { label: "Release to voice", kind: "num", cell: (run) => [formatDuration(releaseToVoice(run))] },
];

/**
 * From the press to the first words on screen: live words where they came,
 * else the final text. Only a take with live words on shows it, and a take
 * where no words came shows none.
 */
export function pressToFirstWords(run) {
  if (!run?.liveChoice) return null;
  const firstWordsAt = run.live?.firstWordsAt ?? (run.input?.text?.trim() ? run.input.finalAt : null);
  return firstWordsAt != null && run.pressedAt != null ? firstWordsAt - run.pressedAt : null;
}

/** A take's live choice with the settings the host applied, where it reported them; null with live words off. */
export function appliedLive(run) {
  const choice = run.liveChoice;
  return choice && { ...choice, settings: run.live?.stream?.settings ?? choice.settings };
}

/** An earlier take's live words: the live model where it differs from the final one, and its settings. */
function liveNote(run) {
  if (!run.liveChoice) return null;
  return [liveDiffers(run) ? `Live words ${run.liveChoice.setup.label}` : null, liveSettingsNote(appliedLive(run))]
    .filter(Boolean).join(" · ");
}

/** A take's final settings as the host applied them: "Names gentle"; null before its text or without any. */
export function finalNote(run) {
  const settings = run.input?.settings;
  return settings && run.setup.final ? liveSettingsNote({ stream: run.setup.final, settings }) : null;
}

/** Whether another model than the final one made the take's live words. */
export function liveDiffers(run) {
  return Boolean(run.liveChoice && run.liveChoice.setup.id !== run.setup.id);
}

/** An earlier take's trail, collapsed to its one line summary. */
export function historyTrail(run) {
  const details = document.createElement("details");
  details.className = "trail";
  const summary = document.createElement("summary");
  summary.className = "note";
  summary.textContent = trailSummary(run.live.updates);
  const list = document.createElement("ol");
  list.replaceChildren(...trailItems(document, run.live.updates, run.input?.text ?? null, formatDuration));
  details.append(summary, list);
  return details;
}

/** From the owner's release to the voice's first sound at the output. */
export function releaseToVoice(run) {
  const firstAudioMs = run?.output?.firstAudioMs;
  if (firstAudioMs == null || run.voiceRequestedAt == null || run.releasedAt == null) return null;
  return run.voiceRequestedAt + firstAudioMs - run.releasedAt;
}

/** Fills a timing list; missing values show a quiet dash, and a note names the figure's model. */
export function describe(list, entries) {
  const rows = entries.map(([label, value, kind, note]) => {
    const row = document.createElement("div");
    if (kind) row.className = kind;
    const term = document.createElement("dt");
    term.textContent = label;
    if (note) {
      const quiet = document.createElement("span");
      quiet.className = "note";
      quiet.textContent = note;
      term.append(quiet);
    }
    const detail = document.createElement("dd");
    detail.textContent = value ?? "–";
    if (value == null) detail.className = "missing";
    row.append(term, detail);
    return row;
  });
  list.replaceChildren(...rows);
}
