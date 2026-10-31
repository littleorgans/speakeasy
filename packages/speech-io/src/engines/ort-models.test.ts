import assert from "node:assert/strict";
import { test } from "node:test";
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
