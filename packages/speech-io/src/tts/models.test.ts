import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveTtsModelPaths, TTS_MODELS, type TtsModel } from "./models.ts";

const entries = Object.entries(TTS_MODELS) as [string, TtsModel][];

test("every TTS model pins its release tarball", () => {
  for (const [id, model] of entries) {
    assert.match(model.sha256, /^[0-9a-f]{64}$/, id);
    assert.ok(model.bytes > 0, id);
    const paths = resolveTtsModelPaths(model);
    assert.equal(paths.sha256, model.sha256, id);
    assert.equal(
      paths.url,
      `https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${model.name}.tar.bz2`,
      id,
    );
  }
});

test("no descriptor hands sherpa espeak-ng data", () => {
  for (const [id, model] of entries) {
    assert.ok(!("dataDir" in model), id);
    assert.ok(!("dataDir" in resolveTtsModelPaths(model)), id);
  }
});
