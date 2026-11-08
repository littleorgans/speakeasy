import { settingControls } from "/setting-controls.js";

/**
 * The Names control under Speech in: the host's one names list
 * (src/names.ts), comma separated. It loads the saved list, saves an edit
 * when the field is committed (blur or Enter), then shows the list the host
 * saved: trimmed, blanks and repeats removed. Its note says whether the list
 * is saved; after a failed save, takes keep the saved list and the note
 * names it.
 */

export const NAMES_ROUTE = "/road-test/names";
// The host's bounds (src/names.ts): 64 names of at most 128 UTF8 bytes, joined with ", ".
const NAMES_LENGTH = 64 * 130;
const NAMES_SETTING = { id: "names", label: "Names", kind: "text", default: "", maxLength: NAMES_LENGTH, note: "" };

export class NamesControl {
  constructor({ document, root }) {
    this.controls = settingControls(document, [NAMES_SETTING]);
    this.input = this.controls.control("names");
    this.input.disabled = true;
    this.input.addEventListener("change", () => void this.commit());
    root.replaceChildren(this.controls.column("names"));
    this.saved = null;
    this.serial = 0;
    this.pending = 0;
    this.saves = Promise.resolve();
    this.describe("Loading the names list");
  }

  /** The saved list, or why it is unavailable; never rejects. */
  async load() {
    try {
      this.show(await namesReply(await fetch(NAMES_ROUTE)));
    } catch (error) {
      this.describe(`Names unavailable: ${error.message}`);
    }
  }

  /** Saves the field's names; an unchanged list only tidies the field. */
  async commit() {
    if (!this.saved) return;
    const names = this.input.value.split(",").map((name) => name.trim()).filter(Boolean);
    if (!this.pending && names.length === this.saved.names.length && names.every((name, index) => name === this.saved.names[index])) {
      this.show(this.saved);
      return;
    }
    const serial = ++this.serial;
    this.pending++;
    this.describe("Saving");
    try {
      // PUTs arrive in edit order, including a revert while an earlier save is pending.
      const save = this.saves.catch(() => {}).then(async () => namesReply(await fetch(NAMES_ROUTE, {
        method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ names }) })));
      this.saves = save;
      const saved = await save;
      this.saved = saved;
      if (serial === this.serial) this.show(saved);
    } catch (error) {
      if (serial !== this.serial) return;
      const kept = this.saved.names.length ? this.saved.names.join(", ") : "no names";
      this.describe(`Not saved: ${error.message}. Takes still use the saved list: ${kept}`);
    } finally { this.pending--; }
  }

  show(saved) {
    this.saved = saved;
    this.input.value = saved.names.join(", ");
    this.input.disabled = false;
    this.describe(saved.names.length ? "Saved. Every model that takes names uses this list from its next take."
      : "Saved: no names. Models that take names run without them.");
  }

  describe(text) {
    this.controls.note("names", text);
  }
}

/** The host's list from a names reply, or its error. */
async function namesReply(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error ?? `host answered ${response.status}`);
  if (!Array.isArray(body.names)) throw new Error("the host sent no names list");
  return body;
}
