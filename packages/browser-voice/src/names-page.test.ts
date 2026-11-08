import assert from "node:assert/strict";
import { test } from "node:test";
import { canonicalNames, namesReply, pageHarness, until, type FetchReply } from "./road-test-page-harness.ts";
import { DEFAULT_NAMES } from "./names.ts";

/** The Names column under Speech in: its label, field and note. */
function names(harness: ReturnType<typeof pageHarness>) {
  const [column] = harness.elements.get("names").children;
  const [label, field, note] = column.children;
  return { label, field, note };
}
/** Types a list into the field and commits it, as blur or Enter does, then waits for the save to settle. */
async function commit(harness: ReturnType<typeof pageHarness>, value: string) {
  const { field, note } = names(harness);
  field.value = value;
  field.listeners.get("change")();
  await until(() => note.textContent !== "Saving");
}

test("the Names control shows the saved list under Speech in, saves a committed edit and shows it canonical", async () => {
  const harness = pageHarness();
  await harness.page.start();
  const { label, field, note } = names(harness);
  assert.equal(label.textContent, "Names");
  assert.equal(field.type, "text");
  assert.equal(field.value, DEFAULT_NAMES.join(", "));
  assert.equal(field.disabled, false);
  assert.match(note.textContent, /^Saved\./);
  await commit(harness, " Qwen, , GPT-SOL,Qwen ");
  assert.deepEqual(harness.namesSaves, [{ names: ["Qwen", "GPT-SOL", "Qwen"] }]);
  assert.equal(field.value, "Qwen, GPT-SOL", "the field shows the list the host saved");
  assert.match(note.textContent, /^Saved\. Every model that takes names uses this list from its next take\.$/);
  await commit(harness, "Qwen,GPT-SOL");
  assert.equal(harness.namesSaves.length, 1, "an unchanged list is not saved again");
  assert.equal(field.value, "Qwen, GPT-SOL");
  await commit(harness, "");
  assert.deepEqual(harness.namesSaves.at(-1), { names: [] });
  assert.equal(note.textContent, "Saved: no names. Models that take names run without them.");
});

test("a refused save keeps the saved list in use and says so; the field keeps the edit to fix", async () => {
  const harness = pageHarness({ saveNames: async (body) => body.names.includes("x".repeat(129))
    ? { ok: false, status: 400, json: async () => ({ error: `name "${"x".repeat(129)}" exceeds 128 UTF8 bytes` }) }
    : canonicalNames(body) });
  await harness.page.start();
  const { field, note } = names(harness);
  await commit(harness, `Grok, ${"x".repeat(129)}`);
  assert.match(note.textContent, /^Not saved: name "x+" exceeds 128 UTF8 bytes\. Takes still use the saved list: Grok, Claude, Codex, GPT, GPT-SOL, TM, Hibiki, Qwen$/);
  assert.equal(field.value, `Grok, ${"x".repeat(129)}`);
  await commit(harness, "Grok");
  assert.equal(field.value, "Grok");
  assert.match(note.textContent, /^Saved\./);
});

test("when the names route fails the field stays disabled and the note says why; the page still starts", async () => {
  const harness = pageHarness({ names: { ok: false, status: 500, json: async () => ({ error: "Local speech request failed" }) } });
  await harness.page.start();
  const { field, note } = names(harness);
  assert.equal(field.disabled, true);
  assert.equal(note.textContent, "Names unavailable: Local speech request failed");
  assert.equal(harness.elements.get("status").textContent, "Ready");
});

test("saves reach the host in edit order and only the latest edit replaces the field", async () => {
  const replies: Array<(reply: FetchReply) => void> = [];
  const harness = pageHarness({ saveNames: () => new Promise<FetchReply>((resolve) => replies.push(resolve)) });
  await harness.page.start();
  const { field, note } = names(harness);
  field.value = "Grok";
  field.listeners.get("change")();
  field.value = "Qwen";
  field.listeners.get("change")();
  await until(() => replies.length >= 1);
  assert.equal(replies.length, 1, "the second PUT waits for the first to finish");
  replies[0]!(namesReply(["Grok"]));
  await until(() => replies.length === 2);
  assert.equal(field.value, "Qwen", "an earlier reply must not overwrite a newer edit");
  replies[1]!(namesReply(["Qwen"]));
  await until(() => note.textContent.startsWith("Saved"));
  assert.equal(field.value, "Qwen");
});

test("reverting to the saved names while a PUT is pending still saves the revert", async () => {
  const replies: Array<(reply: FetchReply) => void> = [];
  const harness = pageHarness({ saveNames: () => new Promise<FetchReply>((resolve) => replies.push(resolve)) });
  await harness.page.start();
  const { field } = names(harness);
  field.value = "Grok";
  const first = harness.page.names.commit();
  field.value = DEFAULT_NAMES.join(", ");
  const second = harness.page.names.commit();
  await until(() => replies.length >= 1);
  replies[0]!(namesReply(["Grok"]));
  await first;
  assert.deepEqual(harness.namesSaves, [{ names: ["Grok"] }, { names: [...DEFAULT_NAMES] }]);
  replies[1]!(namesReply([...DEFAULT_NAMES]));
  await second;
  assert.equal(field.value, DEFAULT_NAMES.join(", "));
});

test("a failed latest edit reports the preceding successful save as the list still in use", async () => {
  let finish!: (reply: FetchReply) => void;
  const harness = pageHarness({ saveNames: (body) => body.names[0] === "Grok"
    ? new Promise<FetchReply>((resolve) => { finish = resolve; })
    : Promise.resolve({ ok: false, status: 400, json: async () => ({ error: "refused edit" }) }) });
  await harness.page.start();
  const { field, note } = names(harness);
  field.value = "Grok";
  const first = harness.page.names.commit();
  field.value = "Qwen";
  const second = harness.page.names.commit();
  await until(() => !!finish);
  finish(namesReply(["Grok"]));
  await Promise.all([first, second]);
  assert.equal(note.textContent, "Not saved: refused edit. Takes still use the saved list: Grok");
  assert.equal(field.value, "Qwen", "the refused edit stays available to fix");
});
