import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, rm, truncate, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  DEFAULT_LOCAL_OUTPUT, getLocalOutputCatalogue, getLocalOutputReadiness,
  LOCAL_OUTPUT_MODELS, resolveLocalOutput,
} from "./catalogue.ts";
import { resolveTtsModelPaths, LOCAL_TTS_MODELS } from "./models.ts";

test("approved catalogue binds exact browser choices and keeps legacy experiments out", () => {
  assert.deepEqual(LOCAL_OUTPUT_MODELS.map((entry) => entry.id), ["piper-kristin-medium", "kokoro-v1.0"]);
  assert.equal(resolveLocalOutput().voice.speakerId, 0);
  for (const [voice, sid] of [["af_heart", 3], ["af_bella", 2], ["am_michael", 16]] as const) {
    assert.equal(resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice }).voice.speakerId, sid);
    assert.equal(resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: sid }).voice.id, voice);
  }
  for (const model of ["piper-amy", "kokoro-v0.19", "kokoro-int8", "toString"]) {
    assert.throws(() => resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, model }));
  }
  assert.throws(() => resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, voice: "af_heart" }));
  assert.throws(() => resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, model: "kokoro-v1.0", voice: 0 }));
  for (const speed of [NaN, Infinity, 0, 0.49, 2.01]) assert.throws(() => resolveLocalOutput({ ...DEFAULT_LOCAL_OUTPUT, speed }));
});

test("catalogue readiness reads without creating assets and publishes no source paths", async () => {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-catalogue-"));
  try {
    const catalogue = await getLocalOutputCatalogue(root);
    assert.equal(catalogue.default.model, "piper-kristin-medium");
    assert.ok(catalogue.models.every((model) => model.readiness.state === "assets-missing"));
    assert.deepEqual(await readdir(root), []);
    assert.ok(!JSON.stringify(catalogue).includes(root));
    assert.ok(!JSON.stringify(catalogue).includes("https:"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("readiness rehashes payloads despite a completion marker", async () => {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-catalogue-corrupt-"));
  try {
    const model = LOCAL_TTS_MODELS["piper-kristin-medium"];
    const paths = resolveTtsModelPaths(model, root);
    await mkdir(paths.dir);
    await writeFile(join(paths.dir, ".complete"), model.files[0].sha256);
    await writeFile(paths.model, "");
    await truncate(paths.model, model.files[0].bytes);
    const readiness = await getLocalOutputReadiness("piper-kristin-medium", root);
    assert.equal(readiness.state, "failed");
    assert.equal(readiness.reason, "Local asset verification failed");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
