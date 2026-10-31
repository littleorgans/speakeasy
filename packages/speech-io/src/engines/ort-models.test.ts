import assert from "node:assert/strict";
import { test } from "node:test";
import { gatesSilence } from "./offline-session.ts";
import { GRANITE_MODELS } from "./ort-granite-models.ts";
import { ORT_MODELS, ORT_THREADS, ortThreads, resolveOrtModel } from "./ort-models.ts";
import { ORT_TDT_MODELS } from "./ort-tdt-models.ts";
import { ORT_WHISTLE_MODELS } from "./ort-whistle-models.ts";
import { SHERPA_MODELS } from "./sherpa-models.ts";

test("the onnxruntime view holds every family id once, apart from sherpa's", () => {
  const families = [ORT_WHISTLE_MODELS, GRANITE_MODELS, ORT_TDT_MODELS].flatMap(Object.keys);
  assert.deepEqual(Object.keys(ORT_MODELS).sort(), [...families].sort());
  assert.equal(new Set(families).size, families.length);
  for (const id of families) {
    assert.ok(!Object.hasOwn(SHERPA_MODELS, id), id);
  }
});

test("model directories are distinct under the shared root", () => {
  const names = Object.values(ORT_MODELS).map((model) => model.name);
  assert.equal(new Set(names).size, names.length);
});

test("every registered model pins its default silence gate mode and log probability floor", () => {
  const models = Object.entries({ ...ORT_MODELS, ...SHERPA_MODELS });
  const policies = Object.fromEntries(models.map(([id, model]) => {
    const policy = "silenceGate" in model ? model.silenceGate : undefined;
    return [id, {
      mode: gatesSilence(policy, "model") ? "on" : "off",
      logProbFloor: policy?.logProbFloor ?? null,
    }];
  }));
  assert.deepEqual(policies, {
    "whistle-onnx-fp32": { mode: "on", logProbFloor: null },
    "en-granite-speech-5.0-470m-turboctc-int8": { mode: "on", logProbFloor: -0.4 },
    "en-fluid-speech-mini": { mode: "off", logProbFloor: null },
    "en-phonon-2-int8": { mode: "on", logProbFloor: null },
    "en-parakeet-redux": { mode: "on", logProbFloor: -0.4 },
    "en-parakeet-tdt-0.6b-v2-int8": { mode: "off", logProbFloor: null },
    "en-moonshine-base-int8": { mode: "off", logProbFloor: null },
    "en-parakeet-ultra-int8": { mode: "off", logProbFloor: null },
    "en-canary-180m-flash-int8": { mode: "on", logProbFloor: null },
    "en-qwen3-asr-0.6b-int8-2026-03-25": { mode: "off", logProbFloor: null },
    "en-2023-06-21": { mode: "off", logProbFloor: null },
    "en-kroko-2025-08-06": { mode: "off", logProbFloor: null },
    "en-2023-06-26": { mode: "off", logProbFloor: null },
    "en-2023-02-21": { mode: "off", logProbFloor: null },
  });
});

test("onnxruntime engines default to sherpa's two threads; a descriptor overrides", () => {
  assert.equal(ORT_THREADS, 2);
  for (const model of Object.values(ORT_MODELS)) {
    assert.equal(ortThreads(model), ORT_THREADS, model.name);
    assert.equal(ortThreads({ ...model, threads: 4 }), 4, model.name);
  }
});

test("an unknown id lists the onnxruntime models", () => {
  assert.throws(() => resolveOrtModel("nope"), /en-phonon-2-int8/);
});
