import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, rm, truncate, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  DEFAULT_LOCAL_OUTPUT, getLocalOutputCatalogue, getLocalOutputReadiness,
  LOCAL_OUTPUT_MODELS, localOutputSettings, resolveLocalOutput,
} from "./catalogue.ts";
import { resolveSettings } from "../settings.ts";
import { RECORDING_VOICE } from "./qwen-voices.ts";
import { payloadDir, resolveTtsModelPaths, LOCAL_TTS_MODELS, TTS_CACHE_ROOT } from "./models.ts";

test("approved catalogue binds exact browser choices and keeps legacy experiments out", () => {
  assert.deepEqual(LOCAL_OUTPUT_MODELS.map((entry) => entry.id), ["piper-kristin-medium", "kokoro-v1.0", "kokoro-mlx-bf16", "pocket-18m",
    "piper-kristin-split", "kokoro-coreml", "phonon-31m",
    "qwen3-tts-0.6b", "qwen3-tts-1.7b", "qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base", "kokoro-ane-heart", "kokoro-ane-folded-heart"]);
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
    assert.deepEqual(catalogue.models.map((model) => model.readiness.state),
      ["assets-missing", "assets-missing", ...Array(11).fill("runtime-missing")], "a worker voice needs its runtime first");
    const configured = await getLocalOutputCatalogue(root, (model) => model === "kokoro-mlx-bf16");
    assert.deepEqual(configured.models.map((model) => model.readiness.state).slice(2, 4), ["assets-missing", "runtime-missing"],
      "then its pinned payload");
    assert.deepEqual(await readdir(root), []);
    assert.ok(!JSON.stringify(catalogue).includes(root));
    assert.ok(!JSON.stringify(catalogue).includes("https:"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a hub voice's payload lives under the cache root whatever the output root; others under the output root", () => {
  assert.equal(payloadDir(LOCAL_TTS_MODELS["phonon-31m"], "/any/output"), join(TTS_CACHE_ROOT, "phonon-31m-en-8c64b55e"));
  assert.equal(payloadDir(LOCAL_TTS_MODELS["pocket-18m"], "/any/output"), join("/any/output", LOCAL_TTS_MODELS["pocket-18m"].name));
  assert.ok(!TTS_CACHE_ROOT.startsWith(process.cwd()), "never inside the repo");
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

test("Qwen models declare offered defaults with notes; Instruction only on 1.7B, Copy only on voice copy", () => {
  const qwen = ["qwen3-tts-0.6b", "qwen3-tts-1.7b", "qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base"] as const;
  for (const id of qwen) {
    const settings = localOutputSettings(id);
    assert.ok(settings.length > 0);
    for (const setting of settings) {
      assert.ok(setting.note.trim(), `${id} ${setting.id} has a note`);
      if (setting.kind === "choice") assert.ok(setting.options.some((option) => option.value === setting.default));
      if (setting.disabledBy) assert.ok(settings.some((other) => other.id === setting.disabledBy));
    }
    assert.doesNotThrow(() => resolveSettings(settings), "every default is offered");
    assert.equal(settings.some((setting) => setting.id === "instructions"), id === "qwen3-tts-1.7b");
    assert.equal(settings.some((setting) => setting.id === "copyMode"), id.endsWith("-base"));
    assert.equal(resolveSettings(settings).firstClause, false, "D1: whole first sentence for Qwen");
    assert.equal(resolveSettings(settings).maxSeconds, id.endsWith("-base") ? 20 : 0, "D8: voice copy caps parts at 20 s");
  }
  assert.deepEqual(localOutputSettings("kokoro-v1.0"), [], "other voices declare none");
  for (const id of ["qwen3-tts-0.6b", "qwen3-tts-1.7b"] as const) {
    const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === id)!;
    assert.deepEqual(model.voices.map((voice) => voice.id),
      ["ryan", "aiden", "dylan", "eric", "ono_anna", "serena", "sohee", "uncle_fu", "vivian"]);
    assert.deepEqual(model.voices.map((voice) => voice.speakerId), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert.equal(LOCAL_TTS_MODELS[id].numSpeakers, 9);
  }
  for (const id of ["qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base"] as const) {
    const voices = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === id)!.voices;
    assert.deepEqual(voices.map((voice) => voice.id), ["ref-3s", "ref-10s", "ref-20s", RECORDING_VOICE]);
    const files = new Set<string>(LOCAL_TTS_MODELS[id].files.map((file) => file.name));
    for (const voice of voices) {
      if ("file" in voice) assert.ok(files.has(voice.file) && voice.transcript.trim(), `${voice.id} is pinned with its transcript`);
    }
    assert.ok(!files.has(".gitattributes"));
  }
});
