import assert from "node:assert/strict";
import { test } from "node:test";
import { OrtGraniteEngine } from "../engines/ort-granite.ts";
import { ORT_MODELS, type OrtModel } from "../engines/ort-models.ts";
import { OrtTdtEngine } from "../engines/ort-tdt.ts";
import { OrtWhistleEngine } from "../engines/ort-whistle.ts";
import { SHERPA_MODELS } from "../engines/sherpa-models.ts";
import { createEngine } from "./harness.ts";
import { BENCH_MODEL_IDS, parseBenchModelId, type BenchModelId } from "./models.ts";

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
    // Gate off: a model's silence gate loads the shared Silero detector, a download.
    const engine = await createEngine("sherpa", id, "off");
    const runtime = Object.hasOwn(ORT_MODELS, id) ? "onnxruntime-node-" : "sherpa-onnx-node-";
    assert.ok(engine.label?.startsWith(runtime), `${id}: ${engine.label}`);
  }
  await assert.rejects(createEngine("stub", "whistle-onnx-fp32"), /--engine sherpa/);
});

test("every onnxruntime id builds the engine class its descriptor's family names", async () => {
  const classes = {
    "granite-ctc": OrtGraniteEngine,
    "parakeet-tdt": OrtTdtEngine,
    whistle: OrtWhistleEngine,
  } satisfies Record<OrtModel["family"], unknown>;
  for (const [id, model] of Object.entries(ORT_MODELS) as [BenchModelId, OrtModel][]) {
    const engine = await createEngine("sherpa", id, "off");
    assert.ok(engine instanceof classes[model.family], id);
  }
});
