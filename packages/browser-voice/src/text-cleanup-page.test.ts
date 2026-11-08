import assert from "node:assert/strict";
import { test } from "node:test";
import * as trail from "../public/live-trail.js";
import { choose, cleanupReply, CLEANUP_MODELS, control, FM_INSTRUCTION, listed, nodes, pageHarness, take, until, type FetchReply }
  from "./road-test-page-harness.ts";

// The design's storage key (design section 5, D3).
const CLEANUP_KEY = "speak-easy.road-test.cleanup";
const FM = "Apple Foundation Models";
const S1 = "S1-mini by Superwhisper";

/** One complete final row with this text. */
const final = (text: string) => [{ type: "row", row: { setupId: "mlx-parakeet/v3-greedy", status: "complete", text,
  timings: { firstUse: false, processingMs: 90 } } }];

/** Cleanup replies the test answers by hand, in request order. */
function heldReplies() {
  const held: Array<{ body: any; answer: (reply: FetchReply) => void }> = [];
  const reply = (body: any) => new Promise<FetchReply>((answer) => held.push({ body, answer }));
  return { held, reply };
}

/** The text a node shows, its children's text joined. */
const shown = (node: any): string => node.children.length ? node.children.map(shown).join("") : node.textContent;
const marked = (node: any, tag: string) => nodes(node).filter((entry) => entry.tag === tag);
const pick = (harness: ReturnType<typeof pageHarness>, label: string) => choose(harness.elements.get("cleanup-model"), label);

test("cleanup is Off on every load; Off sends no request and changes nothing on the take", async () => {
  const harness = pageHarness({}, final("clear the screen"));
  await harness.page.start();
  const select = harness.elements.get("cleanup-model");
  assert.equal(select.value, "off");
  assert.deepEqual(select.options.map((option: any) => option.label), ["Off", S1, FM]);
  assert.equal(harness.elements.get("cleanup-settings").hidden, true, "Off shows nothing under the select");
  assert.equal(harness.elements.get("cleanup-card").hidden, true);
  await take(harness.page, 100);
  await until(() => harness.page.phase === "idle");
  assert.equal(harness.cleanups.length, 0);
  assert.equal(harness.page.run.cleanup, undefined);
  assert.equal(harness.elements.get("cleanup-line").hidden, true);
  assert.ok(!listed(harness.elements.get("input-timings")).some((row: string) => row.startsWith("Cleanup")));
});

test("a model that is not ready is disabled, and the card says why while cleanup is Off", async () => {
  const blocked = { ...CLEANUP_MODELS[0], readiness: { ready: false, reason: "setup runtime file missing" } };
  const harness = pageHarness({ cleanups: [blocked, CLEANUP_MODELS[1]] });
  await harness.page.start();
  assert.deepEqual(harness.elements.get("cleanup-model").options.map((option: any) => option.disabled), [false, true, false]);
  assert.equal(harness.elements.get("cleanup-card").textContent, `${S1} needs setup: setup runtime file missing`);
  assert.equal(harness.elements.get("cleanup-card").hidden, false);
});

test("Foundation Models sends the final, instruction and names; earlier takes only with the toggle on", async () => {
  const harness = pageHarness({}, final("clear the screen"));
  await harness.page.start();
  pick(harness, FM);
  const settings = harness.elements.get("cleanup-settings");
  assert.equal(control(settings, "Instruction").tag, "textarea", "a text setting with lines is a textarea");
  assert.equal(control(settings, "Instruction").rows, 6);
  assert.equal(control(settings, "Names").type, "text");
  await take(harness.page, 100);
  await until(() => harness.page.run.cleanup?.state === "complete" && harness.page.phase === "idle");
  const [first] = harness.cleanups;
  assert.equal(first.modelId, "apple/foundation-models");
  assert.equal(first.text, "clear the screen");
  assert.deepEqual(first.settings, { instruction: FM_INSTRUCTION, names: "Grok, Claude, Codex, GPT, TM", earlier: false });
  assert.equal("earlier" in first, false, "no earlier takes while the toggle is off");
  assert.match(first.requestId, /^[A-Za-z0-9_.:-]{1,128}$/);

  choose(control(settings, "Earlier takes as context"), "On");
  await take(harness.page, 2_000);
  await until(() => harness.cleanups.length === 2 && harness.page.run.cleanup?.state === "complete" && harness.page.phase === "idle");
  assert.deepEqual(harness.cleanups[1].earlier, ["Clear the screen."], "an earlier take goes in cleaned");
  harness.elements.get("read-again").listeners.get("click")();
  await until(() => harness.spoken.length === 3 && harness.page.phase === "idle");
  await take(harness.page, 4_000);
  await until(() => harness.cleanups.length === 3 && harness.page.run.cleanup?.state === "complete" && harness.page.phase === "idle");
  assert.deepEqual(harness.cleanups[2].earlier, ["Clear the screen.", "Clear the screen."],
    "oldest first, and a Read again run's take counts once");
});

test("the voice reads the final while cleanup is held; the cleaned text then replaces it, changed words marked", async () => {
  const { held, reply } = heldReplies();
  const harness = pageHarness({ cleanupReply: reply }, final("so grok said it was ready"));
  await harness.page.start();
  pick(harness, FM);
  await take(harness.page, 100);
  await until(() => held.length === 1 && harness.page.phase === "idle");
  assert.equal(harness.spoken[0].text, "so grok said it was ready", "the voice never waits and reads the final");
  assert.equal(harness.elements.get("transcript").textContent, "so grok said it was ready");
  assert.equal(shown(harness.elements.get("cleanup-line")), `Cleaning up with ${FM}`);
  const pending = harness.elements.get("input-timings").children.find((row: any) => row.children[0].textContent === "Cleanup");
  assert.deepEqual([pending.children[0].children[0].textContent, pending.children[1].textContent], [FM, "–"],
    "pending shows a dash, named by its model");

  held[0]!.answer(cleanupReply(held[0]!.body, "So Grok said it was ready.", { loadMs: 900, firstUse: true }));
  await until(() => harness.page.run.cleanup.state === "complete");
  const transcript = harness.elements.get("transcript");
  assert.equal(shown(transcript), "So Grok said it was ready. ");
  const changed = marked(transcript, "ins");
  assert.deepEqual(changed.map((node) => [node.textContent, node.className, node.title]),
    [["So Grok", "changed", "so grok"], ["ready.", "changed", "ready"]], "case and punctuation fixes glow (D2)");
  const line = harness.elements.get("cleanup-line");
  assert.equal(shown(line), "Before cleanup: so grok said it was ready ");
  assert.deepEqual(marked(line, "del").map((node) => node.textContent), ["so grok", "ready"]);
  const timings = listed(harness.elements.get("input-timings"));
  assert.ok(timings.some((row: string) => /^Cleanup.*: \d+ ms$/.test(row)), timings.join(" | "));
  assert.ok(timings.includes("Cleanup start: 900 ms"), "a first load shows as cleanup start");
  assert.equal(harness.spoken.length, 1, "the cleaned text never reaches the voice");
  assert.equal(harness.page.run.input.text, "so grok said it was ready", "the final is never changed");
});

test("a skipped or refused cleanup keeps the final and names the reason", async () => {
  const replies = [
    async (body: any) => ({ ok: true, status: 200, json: async () => ({ requestId: body.requestId, modelId: body.modelId,
      status: "skipped", code: "unavailable", reason: "Apple Intelligence is off" }) }),
    async () => ({ ok: false, status: 429, json: async () => ({ error: "cleanup is busy" }) }),
    async () => ({ ok: true, status: 200, json: async () => ({ status: "complete", text: "  " }) }),
  ];
  const harness = pageHarness({ cleanupReply: (body) => replies.shift()!(body) }, final("clear the screen"));
  await harness.page.start();
  pick(harness, FM);
  for (const [at, line] of [[100, "Cleanup skipped: Apple Intelligence is off"], [2_000, "Cleanup skipped: busy"],
    [4_000, "Cleanup removed every word"]] as const) {
    await take(harness.page, at);
    await until(() => harness.page.run.cleanup && harness.page.run.cleanup.state !== "pending" && harness.page.phase === "idle");
    assert.equal(harness.elements.get("transcript").textContent, "clear the screen");
    assert.equal(harness.elements.get("transcript").children.length, 0, "the final stays as it was shown");
    assert.equal(shown(harness.elements.get("cleanup-line")), line);
  }
  const [, skipped] = harness.elements.get("history").children;
  assert.deepEqual(skipped.children[2].children.map((part: any) => part.textContent),
    ["clear the screen", "Cleanup skipped: Apple Intelligence is off"]);
  assert.equal(skipped.children[5].children[0].textContent, "skipped");
});

test("a reply after the next take updates its earlier row in place, and an open trail stays open", async () => {
  const { held, reply } = heldReplies();
  const harness = pageHarness({ transcript: "partials", cleanupReply: reply }, final("clear the screen"));
  await harness.page.start();
  pick(harness, FM);
  await take(harness.page, 100);
  await until(() => held.length === 1 && harness.page.phase === "idle");
  const first = harness.page.run;
  first.live.updates.push({ atMs: 300, words: "clear the" });
  await take(harness.page, 2_000);
  await until(() => held.length === 2 && harness.page.phase === "idle");
  const before = harness.elements.get("history").children[0];
  assert.deepEqual(before.children[2].children.slice(0, 2).map((part: any) => part.textContent),
    ["clear the screen", `Cleaning up with ${FM}`]);
  const details = before.children[2].children.find((part: any) => part.tag === "details");
  details.open = true;

  held[0]!.answer(cleanupReply(held[0]!.body, "Clear the screen.", { loadMs: 1_200, firstUse: true }));
  await until(() => first.cleanup.state === "complete");
  const after = harness.elements.get("history").children[0];
  assert.notEqual(after, before, "the earlier row is rebuilt");
  assert.deepEqual(after.children[2].children.slice(0, 2).map((part: any) => part.textContent),
    ["Clear the screen.", "Before cleanup: clear the screen"]);
  assert.equal(after.children[2].children.find((part: any) => part.tag === "details").open, true, "the open trail stays open");
  assert.match(after.children[5].children[0].textContent, /^\d+ ms$/);
  assert.equal(after.children[5].children[1].textContent, "Model start 1.20 s");
  assert.equal(harness.elements.get("transcript").textContent, "clear the screen", "the latest take is left alone");
  assert.equal(harness.elements.get("transcript").children.length, 0);
});

test("S1-mini shows its card and one note, and sends no settings", async () => {
  const harness = pageHarness({}, final("clear the screen"));
  await harness.page.start();
  pick(harness, S1);
  assert.equal(harness.elements.get("cleanup-card").textContent, "Local · 1.2 GB · mlx-lm on MLX");
  const settings = harness.elements.get("cleanup-settings");
  assert.equal(settings.hidden, false);
  assert.deepEqual(settings.children.map((child: any) => child.textContent),
    ["S1-mini uses its card's fixed prompt; it takes no instruction, names or earlier takes."]);
  await take(harness.page, 100);
  await until(() => harness.page.run.cleanup?.state === "complete" && harness.page.phase === "idle");
  assert.deepEqual(Object.keys(harness.cleanups[0]).sort(), ["modelId", "requestId", "text"]);
  // A fast reply can land before the comparison stream ends; either way the cleaned text shows in place.
  assert.equal(shown(harness.elements.get("transcript")), "Clear the screen. ");
});

test("settings persist per model; stored values the model no longer offers fall back to defaults", async () => {
  const stored = { [CLEANUP_KEY]: JSON.stringify({ settings: { "apple/foundation-models": {
    instruction: "Fix names only.", names: "x".repeat(401), earlier: "yes" } } }) };
  const harness = pageHarness({ stored });
  await harness.page.start();
  assert.equal(harness.elements.get("cleanup-model").value, "off", "the choice is never remembered (D3)");
  pick(harness, FM);
  const settings = harness.elements.get("cleanup-settings");
  assert.equal(control(settings, "Instruction").value, "Fix names only.");
  assert.equal(control(settings, "Names").value, "Grok, Claude, Codex, GPT, TM", "too long: the default");
  assert.equal(control(settings, "Earlier takes as context").value, "0", "not offered: the default");
  const names = control(settings, "Names");
  names.value = "Siobhan, Grok";
  names.listeners.get("input")();
  assert.deepEqual(JSON.parse(harness.storage.getItem(CLEANUP_KEY)!),
    { settings: { "apple/foundation-models": { instruction: "Fix names only.", names: "Siobhan, Grok" } } });
  const restore = nodes(settings).find((node) => node.tag === "button" && node.textContent === "Default");
  assert.equal(restore.disabled, false);
  restore.listeners.get("click")();
  assert.equal(control(settings, "Instruction").value, FM_INSTRUCTION, "Default restores the instruction");
  assert.equal(restore.disabled, true);
  assert.deepEqual(JSON.parse(harness.storage.getItem(CLEANUP_KEY)!),
    { settings: { "apple/foundation-models": { names: "Siobhan, Grok" } } });

  const broken = pageHarness({ stored: { [CLEANUP_KEY]: "{not json" } });
  await broken.page.start();
  pick(broken, FM);
  assert.equal(control(broken.elements.get("cleanup-settings"), "Instruction").value, FM_INSTRUCTION, "bad JSON: defaults");
});

test("strict word diffs count case and punctuation; the trail's default diff is unchanged", () => {
  const kinds = (before: string, after: string, options?: { strict?: boolean }) =>
    trail.diffWords(before, after, options).map((segment) => segment.kind);
  assert.deepEqual(kinds("so grok said", "So Grok said."), ["same"]);
  assert.deepEqual(kinds("so grok said", "So Grok said.", { strict: true }), ["replaced"]);
  assert.deepEqual(kinds("eight, n", "eight, nine"), ["same", "inserted"], "the default treats a word still being written as growth");
  assert.deepEqual(kinds("eight, n", "eight, nine", { strict: true }), ["same", "replaced"]);
  assert.deepEqual(kinds("um I mean the blue one", "The blue one.", { strict: true }), ["replaced", "same", "replaced"]);
});
