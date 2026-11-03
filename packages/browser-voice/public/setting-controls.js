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
