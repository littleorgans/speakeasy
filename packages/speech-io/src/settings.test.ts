import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveSettings, SettingsError, type VoiceSetting } from "./settings.ts";

const DECLARED: readonly VoiceSetting[] = [
  { id: "temperature", label: "Temperature", note: "n", kind: "choice", default: 0.9, disabledBy: "greedy",
    options: [{ value: 0.5, label: "0.5" }, { value: 0.9, label: "0.9" }] },
  { id: "greedy", label: "Greedy", note: "n", kind: "choice", default: false,
    options: [{ value: false, label: "Off" }, { value: true, label: "On" }] },
  { id: "instructions", label: "Instruction", note: "n", kind: "text", default: "", maxLength: 5, lines: 3 },
  { id: "seed", label: "Seed", note: "n", kind: "integer", default: 1, min: 0, max: 100 },
];

test("resolveSettings fills every default and keeps offered values of each kind", () => {
  assert.deepEqual(resolveSettings(DECLARED), { temperature: 0.9, greedy: false, instructions: "", seed: 1 });
  assert.deepEqual(resolveSettings(DECLARED, { temperature: 0.5, instructions: "calm", seed: 100 }),
    { temperature: 0.5, greedy: false, instructions: "calm", seed: 100 });
  assert.ok(Object.isFrozen(resolveSettings(DECLARED)));
  assert.deepEqual(resolveSettings([]), {});
});

test("resolveSettings refuses unknown ids, unoffered values, wrong kinds and changes that step aside", () => {
  for (const value of [null, [], "x", { other: 1 }, { temperature: 0.7 }, { temperature: "0.9" }, { greedy: 1 },
    { instructions: "too long" }, { instructions: 3 }, { seed: 1.5 }, { seed: -1 }, { seed: 101 }, { seed: "2" },
    { greedy: true, temperature: 0.5 }]) {
    assert.throws(() => resolveSettings(DECLARED, value), SettingsError, JSON.stringify(value));
  }
  assert.deepEqual(resolveSettings(DECLARED, { greedy: true }).temperature, 0.9, "a stepped aside setting keeps its default");
});
