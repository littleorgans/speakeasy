import assert from "node:assert/strict";
import { test } from "node:test";
import { ORT_MODELS } from "../engines/ort-models.ts";
import { SHERPA_MODELS } from "../engines/sherpa-models.ts";
import { createEngine } from "./harness.ts";
import { BENCH_MODEL_IDS, parseBenchModelId } from "./models.ts";

test("the bench view lists every sherpa and onnxruntime id once", () => {
  assert.deepEqual(BENCH_MODEL_IDS, [...Object.keys(SHERPA_MODELS), ...Object.keys(ORT_MODELS)]);
  assert.equal(new Set(BENCH_MODEL_IDS).size, BENCH_MODEL_IDS.length);
  for (const id of BENCH_MODEL_IDS) {
    assert.equal(parseBenchModelId(id), id);
  }
  assert.throws(() => parseBenchModelId("nope"), /en-kroko-2025-08-06.*whistle-onnx-fp32/);
});

test("createEngine builds every registry id on its own runtime, without loading it", async () => {
  for (const id of BENCH_MODEL_IDS) {
    const engine = await createEngine("sherpa", id);
    const runtime = Object.hasOwn(ORT_MODELS, id) ? "onnxruntime-node-" : "sherpa-onnx-node-";
    assert.ok(engine.label?.startsWith(runtime), `${id}: ${engine.label}`);
  }
  await assert.rejects(createEngine("stub", "whistle-onnx-fp32"), /--engine sherpa/);
});
