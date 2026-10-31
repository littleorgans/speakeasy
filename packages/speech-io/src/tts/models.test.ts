import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { COMPLETE_MARKER } from "../engines/assets.ts";
import { ensureTtsModel, resolveTtsModelPaths, TTS_MODELS, type TtsModel } from "./models.ts";

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

test("a TTS install extracts the files synthesis reads and the model's attribution", async () => {
  const work = await mkdtemp(join(tmpdir(), "speak-easy-tts-models-"));
  // Each fixture carries the extra files its real archive does.
  const fixtures = {
    "piper-amy": { attribution: ["MODEL_CARD"], other: ["en_US-amy-low.onnx.json"] },
    "kokoro-int8": { attribution: ["LICENSE", "README.md"], other: [] },
  } as const;
  try {
    for (const id of ["piper-amy", "kokoro-int8"] as const) {
      const { attribution, other } = fixtures[id];
      const descriptor: TtsModel = TTS_MODELS[id];
      const source = join(work, `source-${id}`, descriptor.name);
      await mkdir(join(source, "espeak-ng-data"), { recursive: true });
      await writeFile(join(source, "espeak-ng-data", "en_dict"), "dict");
      const read = [descriptor.model, descriptor.tokens, descriptor.voices].filter((name) => name !== undefined);
      for (const name of [...read, ...attribution, ...other]) await writeFile(join(source, name), name);
      const bytes = execFileSync("tar", ["-cjf", "-", "-C", join(source, ".."), descriptor.name]);
      const sha256 = createHash("sha256").update(bytes).digest("hex");
      const paths = resolveTtsModelPaths({ ...descriptor, sha256 }, join(work, `root-${id}`));
      await mkdir(join(paths.archive, ".."), { recursive: true });
      await writeFile(paths.archive, bytes);
      await ensureTtsModel(paths);
      assert.deepEqual((await readdir(paths.dir)).sort(), [COMPLETE_MARKER, ...read, ...attribution].sort(), id);
      assert.ok(!existsSync(join(paths.dir, "espeak-ng-data")), id);
      for (const path of [paths.model, paths.tokens, paths.voices]) {
        if (path !== undefined) assert.ok(existsSync(path), `${id} ${path}`);
      }
    }
  } finally {
    await rm(work, { recursive: true, force: true });
  }
});
