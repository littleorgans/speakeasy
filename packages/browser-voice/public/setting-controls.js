/** The road test page's selects: option lists and setting columns. */

export function fill(select, entries, value) {
  select.replaceChildren(...entries.map((entry) => {
    const option = new Option(entry.label, entry.value);
    option.disabled = Boolean(entry.disabled);
    return option;
  }));
  if (entries.some((entry) => entry.value === value)) select.value = value;
}

/**
 * One column per setting, each at today's value with its one line note
 * (src/live-settings.ts, src/final-settings.ts); selects keyed by setting id.
 */
export function settingColumns(settings, onChange) {
  const selects = new Map();
  const columns = settings.map((setting) => {
    const column = document.createElement("label");
    column.className = "setting";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = setting.label;
    const select = document.createElement("select");
    fill(select, setting.options.map((option) => ({ value: String(option.value), label: option.label })), String(setting.default));
    select.addEventListener("change", onChange);
    const note = document.createElement("span");
    note.className = "quiet";
    note.textContent = setting.note;
    column.append(label, select, note);
    selects.set(setting.id, select);
    return column;
  });
  return { columns, selects };
}

/** Each setting's chosen value, its default where it has no select. */
export function chosenSettings(settings, selects) {
  return Object.fromEntries(settings.map((setting) => [setting.id, Number(selects.get(setting.id)?.value ?? setting.default)]));
}

/** Columns for `settings`; `onChange` runs after any control changes. */
export function settingControls(document, settings, onChange = () => {}) {
  const controls = new Map();
  const columns = settings.map((setting) => {
    const column = document.createElement("label");
    column.className = "setting";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = setting.label;
    const control = createControl(document, setting);
    control.addEventListener(setting.kind === "choice" ? "change" : "input", () => onChange(setting.id));
    const note = document.createElement("span");
    note.className = "quiet";
    note.textContent = setting.note;
    column.append(label, control, note);
    controls.set(setting.id, { setting, control, column });
    return column;
  });
  return {
    columns,
    /** Every setting's current value, typed as declared; an unusable number reads as its default. */
    read() {
      return Object.fromEntries([...controls.values()].map(({ setting, control }) => [setting.id, readValue(setting, control)]));
    },
    write(id, value) {
      const entry = controls.get(id);
      if (entry) entry.control.value = controlValue(entry.setting, value);
    },
    disable(id, disabled) {
      const entry = controls.get(id);
      if (entry) entry.control.disabled = disabled;
    },
    column(id) {
      return controls.get(id)?.column;
    },
  };
}

function createControl(document, setting) {
  if (setting.kind === "choice") {
    const select = document.createElement("select");
    fill(select, setting.options.map((option, index) => ({ value: String(index), label: option.label })),
      controlValue(setting, setting.default));
    return select;
  }
  const input = document.createElement("input");
  if (setting.kind === "text") {
    input.type = "text";
    input.maxLength = setting.maxLength;
  } else {
    input.type = "number";
    Object.assign(input, { min: String(setting.min), max: String(setting.max), step: "1" });
  }
  input.value = controlValue(setting, setting.default);
  return input;
}

function controlValue(setting, value) {
  if (setting.kind === "choice") return String(Math.max(0, setting.options.findIndex((option) => option.value === value)));
  return String(value);
}

function readValue(setting, control) {
  if (setting.kind === "choice") return setting.options[Number(control.value)]?.value ?? setting.default;
  if (setting.kind === "text") return String(control.value ?? "").slice(0, setting.maxLength);
  const number = Number(control.value);
  return Number.isSafeInteger(number) && number >= setting.min && number <= setting.max ? number : setting.default;
}
