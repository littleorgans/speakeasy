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

test("a later save supersedes an earlier one still in flight", async () => {
  const replies: Array<(reply: FetchReply) => void> = [];
  const harness = pageHarness({ saveNames: () => new Promise<FetchReply>((resolve) => replies.push(resolve)) });
  await harness.page.start();
  const { field, note } = names(harness);
  field.value = "Grok";
  field.listeners.get("change")();
  field.value = "Qwen";
  field.listeners.get("change")();
  await until(() => replies.length === 2);
  replies[1]!(namesReply(["Qwen"]));
  await until(() => field.value === "Qwen" && note.textContent.startsWith("Saved"));
  replies[0]!(namesReply(["Grok"]));
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(field.value, "Qwen", "the earlier reply lands late and is ignored");
});
