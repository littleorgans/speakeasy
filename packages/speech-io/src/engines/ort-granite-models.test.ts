import assert from "node:assert/strict";
import { test } from "node:test";
import {
  GRANITE_MODELS,
  isGraniteModelId,
  resolveGraniteFiles,
  type GraniteCtcModel,
} from "./ort-granite-models.ts";

const entries = Object.entries(GRANITE_MODELS) as [string, GraniteCtcModel][];
const PINNED_EXPORT =
  "https://huggingface.co/qwertz92/granite-speech-5.0-470m-turboctc-onnx/resolve/e6e3b4d6f590ac51a05f25028b53591d4b260786/";

test("every file is pinned to a commit with digest and size", () => {
  for (const [id, model] of entries) {
    for (const file of [model.graph, model.tokenizer]) {
      assert.match(file.sha256, /^[0-9a-f]{64}$/, id);
      assert.ok(file.bytes > 0, id);
      assert.ok(file.url.startsWith(PINNED_EXPORT), id);
      assert.ok(file.url.endsWith(`/${file.path}`), id);
    }
    assert.equal(model.kind, "offline", id);
    assert.ok(model.maxAudioMs > 0, id);
  }
});

test("only commercial Apache 2.0 weights are listed, never the -nc variant", () => {
  for (const [id, model] of entries) {
    assert.deepEqual(
      Object.keys(model.licence).sort(),
      ["attribution", "commercialUse", "id", "redistribution", "source", "terms"],
      id,
    );
    assert.equal(model.licence.id, "Apache-2.0", id);
    assert.equal(model.licence.commercialUse, true, id);
    assert.equal(model.licence.redistribution, true, id);
    assert.doesNotMatch(`${model.graph.url} ${model.licence.source}`, /-nc\b/, id);
  }
});

test("files resolve under the given root; prototype keys are not ids", () => {
  const model = GRANITE_MODELS["en-granite-speech-5.0-470m-turboctc-int8"];
  const spec = resolveGraniteFiles(model, "/data/speech/models");
  assert.equal(spec.dir, "/data/speech/models/granite-speech-5.0-470m-turboctc-onnx-int8");
  assert.deepEqual(spec.files, [model.graph, model.tokenizer]);
  assert.equal(isGraniteModelId("toString"), false);
});
