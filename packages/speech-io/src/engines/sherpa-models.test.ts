import assert from "node:assert/strict";
import { test } from "node:test";
import {
  isSherpaModelId,
  resolveModelPaths,
  SHERPA_MODELS,
  type SherpaModel,
} from "./sherpa-models.ts";

const entries = Object.entries(SHERPA_MODELS) as [string, SherpaModel][];

/** File names a descriptor hands sherpa, keyed by config field. */
const MODEL_FILE_FIELDS = [
  "encoder",
  "decoder",
  "joiner",
  "tokens",
  "preprocessor",
  "uncachedDecoder",
  "cachedDecoder",
  "convFrontend",
] as const;

test("every entry pins its download", () => {
  for (const [id, model] of entries) {
    assert.ok(model.maxAudioMs > 0, id);
    if ("archive" in model) {
      assert.match(model.archive.sha256, /^[0-9a-f]{64}$/, id);
      assert.ok(model.archive.bytes > 0, id);
      assert.equal(
        model.archive.url,
        `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${model.name}.tar.bz2`,
        id,
      );
      continue;
    }
    for (const file of model.files) {
      assert.match(file.sha256, /^[0-9a-f]{64}$/, `${id} ${file.name}`);
      assert.ok(file.bytes > 0, `${id} ${file.name}`);
      assert.match(file.url, /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\//, id);
      assert.ok(file.url.endsWith(`/${file.name}`), `${id} ${file.name}`);
    }
  }
});

test("a file set entry pins every file its config names", () => {
  for (const [id, model] of entries) {
    if (!("files" in model)) continue;
    const pinned = new Set(model.files.map((file) => file.name));
    for (const field of MODEL_FILE_FIELDS) {
      const name = (model as Record<string, unknown>)[field];
      if (typeof name === "string") assert.ok(pinned.has(name), `${id} ${field}`);
    }
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
  assert.equal(SHERPA_MODELS["en-parakeet-ultra-int8"].family, "nemo-transducer");
  assert.equal(SHERPA_MODELS["en-canary-180m-flash-int8"].family, "canary");
  assert.equal(SHERPA_MODELS["en-qwen3-asr-0.6b-int8-2026-03-25"].family, "qwen3-asr");
});

test("every CC BY 4.0 entry carries a credit line", () => {
  const ccBy = entries.filter(([, model]) => model.licence.id === "CC-BY-4.0");
  assert.deepEqual(ccBy.map(([id]) => id).sort(), [
    "en-canary-180m-flash-int8",
    "en-parakeet-tdt-0.6b-v2-int8",
    "en-parakeet-ultra-int8",
  ]);
  for (const [id, model] of ccBy) {
    assert.match(model.licence.attribution ?? "", /CC BY 4\.0/, id);
  }
});

test("paths resolve under the given root; prototype keys are not ids", () => {
  const paths = resolveModelPaths(SHERPA_MODELS["en-2023-06-21"], "/data/speech/models");
  assert.ok("archive" in paths);
  assert.equal(paths.dir, "/data/speech/models/sherpa-onnx-streaming-zipformer-en-2023-06-21");
  assert.equal(
    paths.archive,
    "/data/speech/models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
  );
  assert.equal(paths.sha256, SHERPA_MODELS["en-2023-06-21"].archive.sha256);
  assert.equal(isSherpaModelId("toString"), false);
});

test("a file set entry resolves to its pinned files under the root", () => {
  const model = SHERPA_MODELS["en-parakeet-ultra-int8"];
  const paths = resolveModelPaths(model, "/data/speech/models");
  assert.deepEqual(paths, {
    dir: "/data/speech/models/parakeet-ultra-sherpa-onnx-int8",
    files: model.files,
  });
});
