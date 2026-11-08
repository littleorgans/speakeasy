import { diffWords, segmentMarks } from "/live-trail.js";
import { formatDuration, inputCard } from "/road-test-text.js";
import { accepts, fill, settingControls } from "/setting-controls.js";

/**
 * Text cleanup under Speech in: an optional model that tidies each take's
 * final once it shows. The voice never waits for it and reads the final. The
 * cleaned text replaces the final on screen, changed words glow once, and a
 * quiet line keeps the final. Each model's settings persist in this browser;
 * the choice starts Off on every load.
 */

export const CLEANUP_KEY = "speak-easy.road-test.cleanup";
// The host's bounds on earlier takes (src/text-cleanup.ts): at most 3, 4096 UTF8 bytes in total.
const EARLIER_TAKES = 3;
const EARLIER_BYTES = 4_096;
// D1: S1-mini runs its card's fixed prompt and declares no settings; one note says so.
const MODEL_NOTES = { "superwhisper/s1-mini": "S1-mini uses its card's fixed prompt; it takes no instruction, names or earlier takes." };

export class TextCleanup {
  constructor({ document, onChange = () => {}, onLanded = () => {} }) {
    this.document = document;
    this.onChange = onChange;
    this.onLanded = onLanded;
    this.select = document.getElementById("cleanup-model");
    this.card = document.getElementById("cleanup-card");
    this.settings = document.getElementById("cleanup-settings");
    this.lineElement = document.getElementById("cleanup-line");
    // A browser that refuses storage keeps the settings for this page only.
    try { this.storage = globalThis.localStorage ?? null; } catch { this.storage = null; }
    this.saved = readSaved(this.storage);
    this.models = [];
    this.unavailable = "";
    this.controls = null;
    this.defaults = new Map();
    this.serial = 0;
    this.select.addEventListener("change", () => { this.showModel(); this.onChange(); });
  }

  /** The cleanup models and their readiness; a model that is not ready is listed but disabled. */
  async load() {
    try {
      const response = await fetch("/prompt-reader/cleanups");
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `cleanups answered ${response.status}`);
      this.models = body.models;
    } catch (error) {
      this.unavailable = error.message;
    }
    // D3: Off on every load; only the settings persist.
    fill(this.select, [{ value: "off", label: "Off" }, ...this.models.map((model) => ({ value: model.id, label: model.label,
      disabled: model.readiness?.ready !== true }))], "off");
    this.showModel();
  }

  model() {
    return this.models.find((model) => model.id === this.select.value);
  }

  /**
   * The chosen model's card and settings at their stored values; Off shows
   * only why a model cannot be chosen.
   */
  showModel() {
    const model = this.model();
    const blocked = this.models.filter((entry) => entry.readiness?.ready !== true)
      .map((entry) => `${entry.label} needs setup: ${entry.readiness?.reason ?? "not configured"}`);
    this.card.textContent = model ? inputCard(model)
      : this.unavailable ? `Cleanup unavailable: ${this.unavailable}` : blocked.join(" · ");
    this.card.hidden = !this.card.textContent;
    const declared = model?.settings ?? [];
    this.controls = declared.length ? settingControls(this.document, declared, () => this.changed()) : null;
    this.defaults = new Map();
    const columns = declared.map((setting) => setting.lines ? this.passage(setting) : this.controls.column(setting.id));
    const note = model && MODEL_NOTES[model.id];
    if (note) columns.push(Object.assign(this.document.createElement("p"), { className: "quiet", textContent: note }));
    this.settings.replaceChildren(...columns);
    this.settings.hidden = columns.length === 0;
    if (!this.controls) return;
    const stored = this.saved[model.id];
    for (const setting of declared) {
      if (stored && typeof stored === "object" && Object.hasOwn(stored, setting.id) && accepts(setting, stored[setting.id])) {
        this.controls.write(setting.id, stored[setting.id]);
      }
    }
    this.markDefaults(model);
  }

  /** A passage setting, a textarea, with a Default button that restores its declared text. */
  passage(setting) {
    const wrap = this.document.createElement("div");
    wrap.className = "passage";
    const button = this.document.createElement("button");
    button.type = "button";
    button.textContent = "Default";
    button.addEventListener("click", () => { this.controls.write(setting.id, setting.default); this.changed(); });
    this.defaults.set(setting.id, button);
    wrap.append(this.controls.column(setting.id), button);
    return wrap;
  }

  /** Stores the values off their default, written on every change. */
  changed() {
    const model = this.model();
    const values = this.controls.read();
    this.saved[model.id] = Object.fromEntries(model.settings.filter((setting) => values[setting.id] !== setting.default)
      .map((setting) => [setting.id, values[setting.id]]));
    try { this.storage?.setItem(CLEANUP_KEY, JSON.stringify({ settings: this.saved })); } catch {}
    this.markDefaults(model);
    this.onChange();
  }

  markDefaults(model) {
    const values = this.controls.read();
    for (const [id, button] of this.defaults) button.disabled = values[id] === model.settings.find((setting) => setting.id === id).default;
  }

  /** The model and settings a press uses; null when cleanup is Off. */
  choice() {
    const model = this.model();
    return model?.readiness?.ready ? { model, settings: this.controls?.read() ?? null } : null;
  }

  /**
   * Sends a take's final for cleanup unless its choice is Off or the final is
   * blank. Never awaited: the reply lands on the run, then onLanded(run).
   */
  start(run, text, history) {
    const choice = run.cleanupChoice;
    if (!choice || !text.trim()) return;
    const cleanup = run.cleanup = { modelId: choice.model.id, label: choice.model.label, final: text, state: "pending" };
    const startedAt = performance.now();
    void this.request(choice, text, choice.settings?.earlier ? earlierTakes(history, run) : []).then((reply) => {
      if (reply.status === "complete") {
        Object.assign(cleanup, { state: "complete", text: reply.text ?? "", ms: performance.now() - startedAt,
          loadMs: reply.firstUse ? reply.loadMs ?? null : null, processingMs: reply.processingMs ?? null });
      } else Object.assign(cleanup, { state: "skipped", reason: reply.reason ?? reply.status });
      this.onLanded(run);
    });
  }

  /** One cleanup request; every answer but complete, a refusal included, comes back skipped with its reason. */
  async request(choice, text, earlier) {
    const body = { requestId: `cleanup-${Date.now()}-${++this.serial}`, modelId: choice.model.id, text,
      ...(choice.settings ? { settings: choice.settings } : {}), ...(earlier.length ? { earlier } : {}) };
    try {
      const response = await fetch("/prompt-reader/cleanups", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify(body) });
      const reply = await response.json().catch(() => ({}));
      if (response.ok) return reply;
      return { status: "skipped", reason: response.status === 429 ? "busy" : reply.error ?? `host answered ${response.status}` };
    } catch (error) {
      return { status: "skipped", reason: error.message };
    }
  }

  /** The latest take's cleaned text in place of its final; changed words glow once. */
  show(run, element) {
    const text = cleanedText(run.cleanup);
    // D2: strict, so case and punctuation fixes glow too.
    if (text) element.replaceChildren(segmentMarks(this.document, diffWords(run.cleanup.final, text, { strict: true }),
      { sides: ["ins"], className: "changed" }));
  }

  /** The quiet line under the latest take; before cleanup, removed and replaced words are struck through. */
  line(run) {
    const cleanup = run?.cleanup;
    const note = cleanupNote(cleanup);
    this.lineElement.hidden = !note;
    const span = (textContent) => Object.assign(this.document.createElement("span"), { textContent });
    this.lineElement.replaceChildren(...(revised(cleanup) ? [span("Before cleanup: "),
      segmentMarks(this.document, diffWords(cleanup.final, cleanup.text, { strict: true }), { sides: ["del"] })]
      : note ? [span(note)] : []));
  }

  /** Speech in figures: final shown to cleaned shown, named by its model, and the model's first load. */
  timings(run) {
    const choice = run ? run.cleanupChoice : this.choice();
    if (!choice) return [];
    const cleanup = run?.cleanup;
    return [["Cleanup", cleanupTime(cleanup), null, choice.model.label],
      ...(cleanup?.loadMs != null ? [["Cleanup start", formatDuration(cleanup.loadMs)]] : [])];
  }
}

/**
 * Up to three recent earlier takes, oldest first, within the host's bytes:
 * each cleaned where it was, else its final. A Read again counts once.
 */
export function earlierTakes(history, run) {
  const seen = new Set();
  const texts = [];
  let bytes = 0;
  for (const earlier of history) {
    if (earlier === run || seen.has(earlier.input) || texts.length === EARLIER_TAKES) continue;
    seen.add(earlier.input);
    const text = cleanedText(earlier.cleanup) ?? earlier.input?.text?.trim();
    if (!text) continue;
    bytes += new TextEncoder().encode(text).length;
    if (bytes > EARLIER_BYTES) break;
    texts.push(text);
  }
  return texts.reverse();
}

/** The cleaned text a take shows; null while pending, skipped, or when cleanup removed every word. */
export function cleanedText(cleanup) {
  return cleanup?.state === "complete" && cleanup.text.trim() ? cleanup.text : null;
}

/** A take's cleanup in words, for the line under it and its earlier row. */
export function cleanupNote(cleanup) {
  if (!cleanup) return null;
  if (cleanup.state === "pending") return `Cleaning up with ${cleanup.label}`;
  if (cleanup.state === "skipped") return `Cleanup skipped: ${cleanup.reason}`;
  if (!cleanedText(cleanup)) return "Cleanup removed every word";
  return revised(cleanup) ? `Before cleanup: ${cleanup.final}` : "Cleanup changed nothing";
}

/** A take's cleanup time: final shown to cleaned shown, or skipped; null while pending or without cleanup. */
export function cleanupTime(cleanup) {
  return cleanup?.state === "skipped" ? "skipped" : formatDuration(cleanup?.ms);
}

/** Whether the cleaned text differs from the final in any word, case or punctuation. */
function revised(cleanup) {
  return Boolean(cleanedText(cleanup)) && diffWords(cleanup.final, cleanup.text, { strict: true })
    .some((segment) => segment.kind !== "same");
}

/** The stored settings by model, or none where the stored value is not an object or not JSON. */
function readSaved(storage) {
  try {
    const settings = JSON.parse(storage?.getItem(CLEANUP_KEY) ?? "{}")?.settings;
    return settings && typeof settings === "object" && !Array.isArray(settings) ? settings : {};
  } catch {
    return {};
  }
}
