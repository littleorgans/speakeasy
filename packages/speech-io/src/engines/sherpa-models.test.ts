import assert from "node:assert/strict";
import { test } from "node:test";
import {
  isSherpaModelId,
  resolveModelPaths,
  SHERPA_MODELS,
  type SherpaModel,
} from "./sherpa-models.ts";

const entries = Object.entries(SHERPA_MODELS) as [string, SherpaModel][];

test("every entry pins its archive", () => {
  for (const [id, model] of entries) {
    assert.match(model.archive.sha256, /^[0-9a-f]{64}$/, id);
    assert.ok(model.archive.bytes > 0, id);
    assert.equal(
      model.archive.url,
      `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${model.name}.tar.bz2`,
      id,
    );
    assert.ok(model.maxAudioMs > 0, id);
  }
});

test("every entry carries the full licence object", () => {
  for (const [id, model] of entries) {
    const { licence } = model;
    assert.deepEqual(
      Object.keys(licence).sort(),
      ["attribution", "commercialUse", "id", "redistribution", "source", "terms"],
      id,
    );
    assert.ok(licence.id && licence.terms, id);
    assert.match(licence.source, /^https:\/\//, id);
    assert.equal(typeof licence.commercialUse, "boolean", id);
    assert.equal(typeof licence.redistribution, "boolean", id);
  }
});

test("Kroko is selectable but marked noncommercial and not redistributable", () => {
  const kroko = SHERPA_MODELS["en-kroko-2025-08-06"];
  assert.equal(kroko.licence.commercialUse, false);
  assert.equal(kroko.licence.redistribution, false);
  assert.ok(isSherpaModelId("en-kroko-2025-08-06"));
  for (const [id, model] of entries) {
    assert.ok(!("use" in model), id);
  }
});

test("Parakeet names its attribution; offline entries name their family", () => {
  const parakeet = SHERPA_MODELS["en-parakeet-tdt-0.6b-v2-int8"];
  assert.match(parakeet.licence.attribution ?? "", /CC BY 4\.0/);
  assert.equal(parakeet.family, "nemo-transducer");
  assert.equal(SHERPA_MODELS["en-moonshine-base-int8"].family, "moonshine");
});

test("paths resolve under the given root; prototype keys are not ids", () => {
  const paths = resolveModelPaths(SHERPA_MODELS["en-2023-06-21"], "/data/speech/models");
  assert.equal(paths.dir, "/data/speech/models/sherpa-onnx-streaming-zipformer-en-2023-06-21");
  assert.equal(
    paths.archive,
    "/data/speech/models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
  );
  assert.equal(paths.sha256, SHERPA_MODELS["en-2023-06-21"].archive.sha256);
  assert.equal(isSherpaModelId("toString"), false);
});
