import assert from "node:assert/strict";
import { test } from "node:test";
import { DEFAULT_LOCAL_OUTPUT, LOCAL_OUTPUT_MODELS, LOCAL_OUTPUT_REGISTRY } from "../../speech-io/src/tts/catalogue.ts";
import { buildSpeechOutputCatalogue as build, type ApiOutputModel } from "./speech-output-catalog.ts";

const buildSpeechOutputCatalogue = (options: Omit<Parameters<typeof build>[0], "localRegistry">) =>
  build({ ...options, localRegistry: LOCAL_OUTPUT_REGISTRY });

function local(state: "ready" | "assets-missing" = "ready") {
  return { default: DEFAULT_LOCAL_OUTPUT, models: LOCAL_OUTPUT_MODELS.map((model) => ({
    ...model, readiness: { state, reason: state === "ready" ? null : "Required local assets are missing" },
  })) };
}

const api: ApiOutputModel = {
  id: "approved-api", label: "API", provider: { kind: "cartesia" },
  defaultVoice: "voice", defaultSpeed: 1, speed: { min: 1, max: 1 },
  sampleRate: 24_000, streaming: "sentence", licence: "Provider terms",
  voices: [{ id: "voice", label: "Voice", languages: ["en"] }],
  readiness: { state: "ready", reason: null },
};

test("catalogue projects only actual approved local identities and preserves readiness without preparation", () => {
  const source = local("assets-missing");
  const catalogue = buildSpeechOutputCatalogue({ local: source });
  assert.deepEqual(catalogue.models.map((model) => model.id), LOCAL_OUTPUT_MODELS.map((model) => model.id));
  assert.deepEqual(catalogue.default, DEFAULT_LOCAL_OUTPUT);
  assert.equal(catalogue.models[1].voices[0].id, "af_heart");
  assert.equal(catalogue.models[1].sampleRate, 24_000);
  assert.deepEqual(catalogue.registry, LOCAL_OUTPUT_REGISTRY);
  assert.ok(catalogue.output.every((model) => !model.selectable && model.reason === "assets-missing"));
  assert.equal(JSON.stringify(catalogue).includes("speakerId"), false);
  source.default = { ...source.default, voice: "changed" };
  assert.equal(catalogue.default.voice, "kristin");
  assert.notEqual(buildSpeechOutputCatalogue({ local: local() }).revision, catalogue.revision);
});

test("API eligibility uses injected presence only and never exposes unconfigured or deferred choices as ready", () => {
  const absent = buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: () => false });
  assert.equal(absent.models.length, 2);
  const present = buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: (kind) => kind === "cartesia" });
  assert.equal(present.models.length, 3);
  assert.equal(present.output[2].transfer, "text-to-api-audio-returned");
  assert.equal(present.output[2].selectable, true);
  const deferred = buildSpeechOutputCatalogue({ local: local(), apiModels: [{ ...api,
    readiness: { state: "adapter-deferred", reason: "Adapter deferred" },
  }], configured: () => true });
  assert.equal(deferred.output[2].selectable, false);
  assert.equal(deferred.output[2].reason, "adapter-deferred");
  assert.notEqual(absent.revision, present.revision);
  assert.equal(present.revision, buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: () => true }).revision);
});
