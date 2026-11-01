import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import {
  ProviderError,
  parseSpeechInputConfig,
  parseSpeechOutputConfig,
  parseSpeechPreferences,
  parseTimingObservation,
  safeProviderError,
  sameSpeechInputConfig,
  sameSpeechOutputConfig,
} from "@speakeasy/speech-io/providers";
import type {
  SpeechConfigRegistries,
  SpeechInputConfig,
  SpeechOutputConfig,
} from "@speakeasy/speech-io/providers";

const registries: SpeechConfigRegistries = {
  input: { localPresets: [{ id: "manifest-owned-preset" }],
    apiModels: [{ kind: "cartesia", model: "researched-input" }] },
  output: [
    { kind: "local", model: "registry-owned-model", voices: [0, "0", "named-voice"], speed: { min: 0.5, max: 2 } },
    { kind: "openai", model: "researched-output", voices: ["api-voice"], speed: { min: 0.75, max: 1.5 } },
  ],
};
const localInput: SpeechInputConfig = { schema: 1, provider: { kind: "local", presetId: "manifest-owned-preset" } };
const apiInput: SpeechInputConfig = { schema: 1, provider: { kind: "cartesia", model: "researched-input" } };
const localOutput: SpeechOutputConfig = {
  schema: 1, provider: { kind: "local" }, model: "registry-owned-model", voice: 0, speed: 1,
};
const apiOutput: SpeechOutputConfig = {
  schema: 1, provider: { kind: "openai" }, model: "researched-output", voice: "api-voice", speed: 1,
};

function rejectsConfig(run: () => unknown): void {
  assert.throws(run, (error: unknown) => error instanceof ProviderError && error.code === "CONFIG_UNSUPPORTED");
}

test("direction configs and preferences round trip without runtime state", () => {
  for (const input of [localInput, apiInput]) {
    for (const output of [localOutput, apiOutput]) {
      const preference = { schema: 1, input, output };
      assert.deepEqual(parseSpeechPreferences(JSON.parse(JSON.stringify(preference)), registries), preference);
    }
  }
  const parsed = parseSpeechInputConfig(localInput, registries.input);
  assert.notEqual(parsed, localInput);
  assert.notEqual(parsed.provider, localInput.provider);
});

test("input identities come exclusively from supplied registries", () => {
  const added: SpeechInputConfig = { schema: 1, provider: { kind: "local", presetId: "new-manifest-entry" } };
  rejectsConfig(() => parseSpeechInputConfig(added, registries.input));
  assert.deepEqual(parseSpeechInputConfig(added, {
    ...registries.input, localPresets: [...registries.input.localPresets, { id: "new-manifest-entry" }],
  }), added);
  rejectsConfig(() => parseSpeechInputConfig(apiInput, { ...registries.input, apiModels: [] }));
});

test("input rejects unknown fields, direction, identity and schema", () => {
  const invalid: unknown[] = [null, [], {}, { ...localInput, schema: 2 },
    { ...localInput, endpoint: "forbidden" }, { ...localInput, apiKey: "forbidden" },
    { ...localInput, selected: true }, { ...localInput, provider: { kind: "local", presetId: "unknown" } },
    { schema: 1, provider: { kind: "local", presetId: "manifest-owned-preset", path: "forbidden" } },
    { schema: 1, provider: { kind: "local", model: "manifest-owned-preset" } },
    { schema: 1, provider: { kind: "api", model: "researched-input" } },
    { schema: 1, provider: { kind: "elevenlabs", model: "researched-input" } },
    { schema: 1, provider: { kind: "openai", model: "researched-input" } },
    { schema: 1, provider: { kind: "cartesia", model: "unresearched" } },
    { schema: 1, provider: { kind: "cartesia", model: "researched-input", options: {} } },
    { schema: 1, provider: ["local"] },
  ];
  for (const value of invalid) rejectsConfig(() => parseSpeechInputConfig(value, registries.input));
  rejectsConfig(() => parseSpeechPreferences({ schema: 1, input: localInput, output: localOutput, session: "no" }, registries));
});

test("output validates model voice pairs, numeric zero and model speed limits", () => {
  for (const voice of [0, "0", "named-voice"]) {
    for (const speed of [0.5, 1, 2]) {
      const config = { ...localOutput, voice, speed };
      assert.deepEqual(parseSpeechOutputConfig(config, registries.output), config);
    }
  }
  const invalid: unknown[] = [null, [], {}, { ...localOutput, schema: "1" },
    { ...localOutput, speed: undefined }, { ...localOutput, model: "unknown" },
    { ...localOutput, voice: "api-voice" }, { ...apiOutput, voice: "named-voice" },
    { ...localOutput, voice: -1 }, { ...localOutput, voice: 0.5 }, { ...localOutput, voice: "" },
    { ...localOutput, speed: 0.49 }, { ...localOutput, speed: 2.01 },
    ...[0, -1, NaN, Infinity, "1"].map((speed) => ({ ...localOutput, speed })),
    { ...localOutput, provider: { kind: "groq" } },
    { ...localOutput, provider: { kind: "local", presetId: "forbidden" } },
    { ...localOutput, credentials: "forbidden" },
  ];
  for (const value of invalid) rejectsConfig(() => parseSpeechOutputConfig(value, registries.output));
});

test("config identity compares fields rather than object order, preserving voice type", () => {
  const reordered = parseSpeechInputConfig({ provider: { model: "researched-input", kind: "cartesia" }, schema: 1 }, registries.input);
  assert.equal(sameSpeechInputConfig(apiInput, reordered), true);
  assert.equal(sameSpeechInputConfig(apiInput, localInput), false);
  assert.equal(sameSpeechOutputConfig(localOutput, { ...localOutput, voice: "0" }), false);
  assert.equal(sameSpeechOutputConfig(localOutput, { ...localOutput, speed: 1.1 }), false);
});

test("timing preserves measured zero, reported evidence and explicit unavailability", () => {
  for (const observation of [
    { status: "measured", ms: 0, clock: "worker", boundary: "decode" },
    { status: "measured", ms: 12, clock: "server", boundary: "request transport to full body" },
    { status: "provider-reported", ms: 4, metric: "documented-metric", boundary: "documented-boundary", evidenceRef: "vendor-research" },
    { status: "unavailable", reason: "Provider does not expose processing or network timing." },
  ]) assert.deepEqual(parseTimingObservation(observation), observation);
  const measured = { status: "measured", ms: 1, clock: "server", boundary: "request wall" };
  for (const observation of [null, 0, {}, { ...measured, ms: -1 }, { ...measured, ms: NaN },
    { ...measured, ms: Infinity }, { ...measured, unit: "seconds" }, { ...measured, clock: "mixed" },
    { status: "provider-reported", ms: 3, metric: "guessed", boundary: "processing" },
    { status: "unavailable", reason: "" }, { status: "unavailable", reason: "absent", ms: 0 },
  ]) assert.throws(() => parseTimingObservation(observation),
    (error: unknown) => error instanceof ProviderError && error.code === "TIMING_INVALID");
});

test("safe errors expose fixed bounded messages without caller or transport data", () => {
  const raw = new Error("sensitive transport body");
  const safe = safeProviderError(raw);
  assert.equal(safe.code, "PROVIDER_FAILED");
  assert.equal(JSON.stringify(safe).includes(raw.message), false);
  const mutated = new ProviderError("SELECTION_REQUIRED");
  mutated.message = raw.message;
  assert.equal(safeProviderError(mutated).message.includes(raw.message), false);
  assert.equal(JSON.stringify(mutated).includes(raw.message), false);
});

test("the providers package export imports with native loading and fetch forbidden", () => {
  const path = createRequire(import.meta.url).resolve("@speakeasy/speech-io/providers");
  const script = `
    process.dlopen = () => { throw new Error('native import forbidden'); };
    globalThis.fetch = () => { throw new Error('provider request forbidden'); };
    const contracts = await import(${JSON.stringify(pathToFileURL(path).href)});
    if (typeof contracts.parseSpeechInputConfig !== 'function') process.exit(1);
  `;
  const child = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
    cwd: process.cwd(), env: {}, encoding: "utf8",
  });
  assert.equal(child.status, 0);
});
