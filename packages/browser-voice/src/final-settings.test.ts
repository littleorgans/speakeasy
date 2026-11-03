import assert from "node:assert/strict";
import { test } from "node:test";
import { FINAL_SETUPS, finalSetup, resolveFinalSettings } from "./final-settings.ts";
import { getSetup } from "./setup-registry.ts";

const COHERE = "cohere/transcribe" as const;

test("Cohere's one final setting is Names: Off by default, then Gentle and Strong", () => {
  assert.deepEqual(FINAL_SETUPS.map((setup) => setup.setupId), [COHERE]);
  const [names] = finalSetup(COHERE)!.settings;
  assert.equal(names!.label, "Names");
  assert.equal(names!.note, "Biases Cohere toward Grok, Claude, Codex, GPT, TM; can turn similar words into these names");
  assert.deepEqual(names!.options.map((option) => [option.label, option.value]), [["Off", 0], ["Gentle", 1.25], ["Strong", 2.5]]);
  assert.equal(names!.default, 0, "Off is today's output");
  const pinned = getSetup(COHERE).controls.namesBias as { default: number; values: number[] };
  assert.deepEqual([pinned.default, pinned.values], [names!.default, names!.options.map((option) => option.value)],
    "the manifest pins the offered values");
  assert.equal(finalSetup("hex/vocabulary"), undefined);
});

test("final settings fill every selected setup's defaults and refuse anything not offered", () => {
  assert.deepEqual(resolveFinalSettings([COHERE]), { [COHERE]: { namesBias: 0 } });
  assert.deepEqual(resolveFinalSettings(["mlx-parakeet/v3-greedy", COHERE], { [COHERE]: { namesBias: 1.25 } }),
    { [COHERE]: { namesBias: 1.25 } });
  assert.deepEqual(resolveFinalSettings(["mlx-parakeet/v3-greedy"]), {}, "a setup without final settings sends none");
  assert.throws(() => resolveFinalSettings(["mlx-parakeet/v3-greedy"], { [COHERE]: { namesBias: 2.5 } }), /no final settings/);
  assert.throws(() => resolveFinalSettings([COHERE], { "hex/unified": {} }), /no final settings/);
  assert.throws(() => resolveFinalSettings([COHERE], { [COHERE]: { namesBias: 1 } }), /unsupported namesBias/);
  assert.throws(() => resolveFinalSettings([COHERE], { [COHERE]: { namesBias: "2.5" } }), /unsupported namesBias/);
  assert.throws(() => resolveFinalSettings([COHERE], { [COHERE]: { language: "en" } }), /unknown/);
  assert.throws(() => resolveFinalSettings([COHERE], [1]), /object/);
});
