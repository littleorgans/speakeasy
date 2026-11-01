import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveTtsModelPaths, LOCAL_TTS_MODELS } from "./models.ts";
import { createTtsConfig, validateSpeaker } from "./synth.ts";

test("both native families keep CPU and have no espeak or progress path", () => {
  for (const id of ["piper-kristin-medium", "kokoro-v1.0"] as const) {
    const model = LOCAL_TTS_MODELS[id];
    const config = createTtsConfig(model.family, resolveTtsModelPaths(model), 2);
    assert.equal(config.model?.provider, "cpu");
    assert.equal(config.model?.numThreads, 2);
    assert.ok(!JSON.stringify(config).includes("dataDir"));
    assert.ok(!JSON.stringify(config).includes("onProgress"));
  }
});

test("native speaker validation checks integer bounds", () => {
  validateSpeaker(0, 1);
  validateSpeaker(53, 54);
  for (const sid of [-1, 0.5, NaN, Infinity, 54]) assert.throws(() => validateSpeaker(sid, 54));
});
