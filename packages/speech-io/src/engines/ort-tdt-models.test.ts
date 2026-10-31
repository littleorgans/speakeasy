import assert from "node:assert/strict";
import { test } from "node:test";
import {
  downloadBytes,
  ORT_TDT_MODELS,
  resolveOrtTdtModel,
  type OrtTdtModel,
} from "./ort-tdt-models.ts";

const REVISION_URL = /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\//;

function graphFiles(model: OrtTdtModel): string[] {
  return model.layout === "fused"
    ? [model.preprocessor, model.encoder, model.decoderJoint, model.vocab]
    : [model.frontend, model.encoder, model.decoder, model.joint, model.melFilters, model.tokenizer];
}

for (const [id, model] of Object.entries(ORT_TDT_MODELS) as [string, OrtTdtModel][]) {
  test(`${id}: every file is pinned and every role names a listed file`, () => {
    const paths = model.files.map((file) => file.name);
    assert.equal(new Set(paths).size, paths.length);
    for (const file of model.files) {
      assert.match(file.url, REVISION_URL);
      assert.ok(file.url.endsWith(`/${file.name}`));
      assert.match(file.sha256, /^[0-9a-f]{64}$/);
      assert.ok(file.bytes > 0);
    }
    for (const role of graphFiles(model)) {
      assert.ok(paths.includes(role), `${role} is not downloaded`);
    }
    assert.ok(downloadBytes(model) > 100_000_000);
  });

  test(`${id}: licence permits commercial redistribution with attribution`, () => {
    assert.equal(model.kind, "offline");
    assert.equal(model.licence.id, "CC-BY-4.0");
    assert.ok(model.licence.commercialUse && model.licence.redistribution);
    assert.ok(model.licence.attribution?.includes("CC BY 4.0"));
    assert.match(model.licence.source, /\/blob\/[0-9a-f]{40}\//);
    assert.ok(model.maxAudioMs >= 30_000 && model.maxAudioMs <= 40_000);
    assert.equal(model.decoding.vocabSize, model.decoding.blankId + 1);
  });
}

test("resolveOrtTdtModel names the valid ids on a miss", () => {
  assert.equal(resolveOrtTdtModel("en-phonon-2-int8").name, "phonon-2-onnx-int8");
  assert.throws(() => resolveOrtTdtModel("en-unknown"), /en-fluid-speech-mini/);
});
