import assert from "node:assert/strict";
import { test } from "node:test";
import { applyRewrites, REWRITE_RULES } from "./replace.ts";

test("fixes the corpus proper-noun mis-splits", () => {
  assert.equal(
    applyRewrites("Navigate to Little Organs project"),
    "Navigate to littleorgans project",
  );
  assert.equal(applyRewrites("Open crown browser"), "Open chrome browser");
  assert.equal(applyRewrites("Close pain ten"), "Close pane 10");
});

test("is case-insensitive and preserves surrounding text", () => {
  assert.equal(applyRewrites("CROWN"), "chrome");
  assert.equal(applyRewrites("go to sleep"), "go to sleep");
  assert.equal(applyRewrites("clear the screen"), "clear the screen");
});

test("matches whole words only", () => {
  assert.equal(applyRewrites("painting the fence"), "painting the fence");
  assert.equal(applyRewrites("tenderness"), "tenderness");
});

test("over-trigger rules fire on legitimate words (documents brittleness)", () => {
  // crown/pain/ten are real words; the map cannot tell mishear from intent.
  assert.equal(applyRewrites("the crown jewels"), "the chrome jewels");
  assert.equal(applyRewrites("in pain"), "in pane");
  assert.equal(applyRewrites("ten apples"), "10 apples");
});

test("ruleset flags the over-trigger risks", () => {
  const risky = REWRITE_RULES.filter((r) => r.overTrigger).map((r) => r.from);
  assert.deepEqual(risky.sort(), ["crown", "pain", "ten"]);
});
