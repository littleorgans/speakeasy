import { fill, settingControls } from "/setting-controls.js";

/**
 * The chosen voice's settings under Speech out: the main controls in one
 * row, sampling and fine detail in a closed Sampling disclosure, and a
 * Preset that fills them. Only values that differ from the declared default
 * are sent; the host fills the rest and echoes every value it applied. Greedy
 * makes its stack's temperature, top p and top k step aside: they return to
 * their defaults and grey out.
 */

/** Filled into the controls, never sent. Soft is the taming candidate, chosen by clip length only. */
export const PRESETS = [
  { id: "stock", label: "Stock", values: {} },
  { id: "soft", label: "Soft", values: { temperature: 0.7, topP: 0.9, repetitionPenalty: 1.15 } },
];
const PRESET_NOTE = "Soft: temperature 0.7, top p 0.9, repetition penalty 1.15, picked by clip length, not by ear. Fills the controls.";
const SEED_MODES = [{ value: "fixed", label: "Fixed" }, { value: "new", label: "New each reading" }];
const NOTE_TEXT_CHARS = 40;

/**
 * A reading's settings by label, only those that differ from their default,
 * plus any always listed (a drawn seed): "seed 7, temperature 0.7".
 */
export function settingsNote(settings, applied, always = []) {
  return (settings ?? []).filter((setting) => applied && Object.hasOwn(applied, setting.id) &&
    (applied[setting.id] !== setting.default || always.includes(setting.id))).map((setting) => {
    const value = applied[setting.id];
    if (setting.kind === "text") {
      return `${setting.label.toLowerCase()} "${value.length > NOTE_TEXT_CHARS ? `${value.slice(0, NOTE_TEXT_CHARS)}…` : value}"`;
    }
    const option = setting.kind === "choice" ? setting.options.find((entry) => entry.value === value) : null;
    return `${setting.label.toLowerCase()} ${(option?.label ?? String(value)).replace(/ \(stock\)$/, "").toLowerCase()}`;
  }).join(", ");
}

export class VoiceSettings {
  constructor({ document, root, onChange = () => {} }) {
    this.document = document;
    this.root = root;
    this.onChange = onChange;
    this.model = null;
    this.controls = null;
  }

  /** Renders the model's settings at their defaults; hides the row for a model without any. */
  show(model) {
    this.model = model;
    const settings = model?.settings ?? [];
    this.root.hidden = settings.length === 0;
    if (!settings.length) { this.controls = null; this.root.replaceChildren(); return; }
    this.controls = settingControls(this.document, settings, () => this.changed());
    const main = this.document.createElement("div");
    main.className = "voice-columns";
    main.append(...settings.filter((setting) => setting.group !== "sampling").map((setting) => this.controls.column(setting.id)),
      this.presetColumn());
    const seed = this.controls.column("seed");
    if (seed) {
      const [label, input, note] = [...seed.children];
      this.seedMode = this.document.createElement("select");
      this.seedMode.setAttribute("aria-label", "Seed mode");
      fill(this.seedMode, SEED_MODES, "fixed");
      this.seedMode.addEventListener("change", () => this.changed());
      const pair = this.document.createElement("span");
      pair.className = "pair";
      pair.append(this.seedMode, input);
      seed.replaceChildren(label, pair, note);
    }
    const sampling = this.document.createElement("details");
    sampling.className = "sampling";
    const summary = this.document.createElement("summary");
    summary.className = "label";
    summary.textContent = "Sampling";
    const columns = this.document.createElement("div");
    columns.className = "voice-columns";
    columns.append(...settings.filter((setting) => setting.group === "sampling").map((setting) => this.controls.column(setting.id)));
    sampling.append(summary, columns);
    this.root.replaceChildren(main, sampling);
    this.changed(false);
  }

  presetColumn() {
    const column = this.document.createElement("label");
    column.className = "setting";
    const label = this.document.createElement("span");
    label.className = "label";
    label.textContent = "Preset";
    this.preset = this.document.createElement("select");
    this.preset.addEventListener("change", () => this.applyPreset(this.preset.value));
    const note = this.document.createElement("span");
    note.className = "quiet";
    note.textContent = PRESET_NOTE;
    column.append(label, this.preset, note);
    return column;
  }

  /** Sets the preset's sampling values and returns the rest of its stack to default. */
  applyPreset(id) {
    const preset = PRESETS.find((entry) => entry.id === id);
    if (!preset || !this.controls) return;
    for (const setting of this.model.settings.filter((entry) => entry.group === "sampling")) {
      this.controls.write(setting.id, preset.values[setting.id] ?? setting.default);
    }
    this.changed();
  }

  /** Greedy steps its stack aside; the preset shows which one the controls match, else Custom. */
  changed(notify = true) {
    const values = this.controls.read();
    for (const setting of this.model.settings) {
      if (!setting.disabledBy) continue;
      const aside = values[setting.disabledBy] === true;
      if (aside) this.controls.write(setting.id, setting.default);
      this.controls.disable(setting.id, aside);
    }
    const now = this.controls.read();
    const sampling = this.model.settings.filter((setting) => setting.group === "sampling");
    const match = PRESETS.find((preset) => sampling.every((setting) => now[setting.id] === (preset.values[setting.id] ?? setting.default)));
    fill(this.preset, [...PRESETS.map(({ id, label }) => ({ value: id, label })), ...(match ? [] : [{ value: "custom", label: "Custom" }])],
      match?.id ?? "custom");
    if (this.seedMode) this.controls.disable("seed", this.seedMode.value === "new");
    if (notify) this.onChange();
  }

  /**
   * The reading's settings to send: only values off their default. New each
   * reading draws a seed and writes it into the field, so switching to Fixed
   * pins a take worth keeping.
   */
  request() {
    if (!this.controls) return null;
    const drawn = this.seedMode?.value === "new";
    if (drawn) this.controls.write("seed", Math.floor(Math.random() * 2 ** 32));
    const values = this.controls.read();
    const settings = Object.fromEntries(this.model.settings
      .filter((setting) => values[setting.id] !== setting.default).map((setting) => [setting.id, values[setting.id]]));
    return { settings, declared: this.model.settings, always: drawn ? ["seed"] : [] };
  }
}
