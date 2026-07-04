import assert from "node:assert/strict";
import { test } from "node:test";
import { applyRules, DEFAULT_RULES } from "./rules.ts";

test("fixes the corpus proper-noun mis-hears", () => {
  assert.equal(
    applyRules("Navigate to Little Organs project"),
    "Navigate to littleorgans project",
  );
  assert.equal(applyRules("Open crown browser"), "Open chrome browser");
  assert.equal(applyRules("Close pain 10"), "Close pane 10");
});

test("is case-insensitive and passes through non-matching text", () => {
  assert.equal(applyRules("CROWN"), "chrome");
  assert.equal(applyRules("clear the screen"), "clear the screen");
  assert.equal(applyRules(""), "");
});

test("matches whole words only", () => {
  assert.equal(applyRules("painting the fence"), "painting the fence");
  assert.equal(applyRules("crowned"), "crowned");
});

test("is idempotent", () => {
  const once = applyRules("Open crown browser");
  assert.equal(applyRules(once), once);
});

test("over-trigger rules are flagged in the ruleset", () => {
  const risky = DEFAULT_RULES.filter((r) => r.overTrigger).map((r) => r.from);
  assert.deepEqual(risky.sort(), ["crown", "pain"]);
  // documents brittleness: fires on legitimate words too
  assert.equal(applyRules("the crown jewels"), "the chrome jewels");
});

test("custom rules override the default set", () => {
  assert.equal(
    applyRules("deploy to prod", [{ from: "prod", to: "production" }]),
    "deploy to production",
  );
});
