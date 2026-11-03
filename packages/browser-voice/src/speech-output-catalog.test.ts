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
  const locals = LOCAL_OUTPUT_MODELS.length;
  const absent = buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: () => false });
  assert.equal(absent.models.length, locals);
  const present = buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: (kind) => kind === "cartesia" });
  assert.equal(present.models.length, locals + 1);
  assert.equal(present.output[locals].transfer, "text-to-api-audio-returned");
  assert.equal(present.output[locals].selectable, true);
  const deferred = buildSpeechOutputCatalogue({ local: local(), apiModels: [{ ...api,
    readiness: { state: "adapter-deferred", reason: "Adapter deferred" },
  }], configured: () => true });
  assert.equal(deferred.output[locals].selectable, false);
  assert.equal(deferred.output[locals].reason, "adapter-deferred");
  assert.notEqual(absent.revision, present.revision);
  assert.equal(present.revision, buildSpeechOutputCatalogue({ local: local(), apiModels: [api], configured: () => true }).revision);
});

test("Qwen models publish their voice settings and voice copy voices their origin and pinned transcript", () => {
  const catalogue = buildSpeechOutputCatalogue({ local: local() });
  const model = (id: string) => catalogue.models.find((entry) => entry.id === id)!;
  assert.ok(model("qwen3-tts-1.7b").settings?.some((setting) => setting.id === "instructions"));
  assert.equal(model("kokoro-v1.0").settings, undefined);
  const voices = model("qwen3-tts-0.6b-base").voices;
  assert.deepEqual(voices.map((voice) => voice.origin), ["synthetic", "synthetic", "synthetic", "recording"]);
  assert.ok(voices[1]!.transcript?.startsWith("This is a calm and steady"));
  assert.equal(voices[3]!.transcript, undefined);
  assert.equal(JSON.stringify(catalogue).includes("references/"), false, "no payload file names");
});
