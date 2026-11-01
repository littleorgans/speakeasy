import assert from "node:assert/strict";
import childProcess from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import https from "node:https";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { SherpaEngine } from "./sherpa.ts";
import { SherpaOfflineEngine } from "./sherpa-offline.ts";
import type { OfflineRecognizerConfig, OnlineRecognizerConfig } from "./sherpa-api.ts";
import { SHERPA_MODELS, type SherpaModelId } from "./sherpa-models.ts";
import { bindingWith } from "./test-support.ts";

// Independent inventory of descriptor fields consumed by the recognizer configs.
const FILE_FIELDS = [
  "encoder", "decoder", "joiner", "tokens", "bpe", "preprocessor",
  "uncachedDecoder", "cachedDecoder", "convFrontend",
] as const;

async function fixture(context: TestContext, id: SherpaModelId, readOnly = false) {
  const root = await fsp.mkdtemp(join(tmpdir(), "speak-easy-prepared-"));
  const directory = join(root, "arbitrary prepared basename");
  const model = SHERPA_MODELS[id];
  const files = FILE_FIELDS.flatMap((field) =>
    field in model ? [String(model[field as keyof typeof model])] : [],
  );
  const directories = "tokenizer" in model ? [model.tokenizer] : [];
  await fsp.mkdir(directory);
  context.after(async () => {
    for (const path of [root, directory, ...directories.map((name) => join(directory, name))]) {
      await fsp.chmod(path, 0o755);
    }
    await fsp.rm(root, { recursive: true, force: true });
  });
  for (const name of directories) {
    await fsp.mkdir(join(directory, name));
    await fsp.writeFile(join(directory, name, "fixture.json"), "{}");
  }
  for (const name of files) await fsp.writeFile(join(directory, name), "fixture bytes");
  if (readOnly) {
    for (const name of files) await fsp.chmod(join(directory, name), 0o444);
    for (const name of directories) {
      await fsp.chmod(join(directory, name, "fixture.json"), 0o444);
      await fsp.chmod(join(directory, name), 0o555);
    }
    await fsp.chmod(directory, 0o555);
    await fsp.chmod(root, 0o555);
  }
  return { root, directory, files, directories };
}

function recordingEngine(id: SherpaModelId, directory: string) {
  const configs: (OnlineRecognizerConfig | OfflineRecognizerConfig)[] = [];
  class Recognizer {
    constructor(config: OnlineRecognizerConfig | OfflineRecognizerConfig) { configs.push(config); }
    createStream() { return { acceptWaveform() {}, inputFinished() {} }; }
    isReady() { return false; }
    isEndpoint() { return false; }
    decode() {}
    reset() {}
    getResult() { return {}; }
    async decodeAsync() { return {}; }
  }
  const Engine = SHERPA_MODELS[id].kind === "streaming" ? SherpaEngine : SherpaOfflineEngine;
  return {
    configs,
    engine: new Engine({
      model: id,
      preparedModelDirectory: directory,
      binding: bindingWith({ OnlineRecognizer: Recognizer, OfflineRecognizer: Recognizer }),
    }),
  };
}

async function snapshot(directory: string): Promise<unknown[]> {
  const info = await fsp.stat(directory);
  const entries: unknown[] = [{ mode: info.mode, mtime: info.mtimeMs, ctime: info.ctimeMs }];
  for (const name of (await fsp.readdir(directory)).sort()) {
    const path = join(directory, name);
    const stat = await fsp.stat(path);
    entries.push([name, stat.isDirectory() ? await snapshot(path) : {
      mode: stat.mode, mtime: stat.mtimeMs, ctime: stat.ctimeMs,
      bytes: await fsp.readFile(path, "utf8"),
    }]);
  }
  return entries;
}

async function withoutInstallation(context: TestContext, run: () => Promise<void>) {
  let attempts = 0;
  const forbidden = () => { attempts++; throw new Error("Unexpected installation or write"); };
  for (const method of ["mkdir", "mkdtemp", "writeFile", "appendFile", "rename", "rm", "unlink", "copyFile", "link", "symlink", "chmod"] as const) {
    context.mock.method(fsp, method, forbidden);
  }
  context.mock.method(fs, "createWriteStream", forbidden);
  context.mock.method(childProcess, "spawn", forbidden);
  context.mock.method(https, "get", forbidden);
  syncBuiltinESMExports();
  try {
    await run();
    assert.equal(attempts, 0, "no installer, network, or write attempts");
  } finally {
    context.mock.restoreAll();
    syncBuiltinESMExports();
  }
}

function configuredPaths(value: unknown, directory: string): string[] {
  if (typeof value === "string") return value.startsWith(directory) ? [value] : [];
  if (value && typeof value === "object") {
    return Object.values(value).flatMap((entry) => configuredPaths(entry, directory));
  }
  return [];
}

for (const id of Object.keys(SHERPA_MODELS) as SherpaModelId[]) {
  test(`${id}: prepare and open use the exact read only directory without installation`, async (context) => {
    const { root, directory, files, directories } = await fixture(context, id, true);
    const before = await snapshot(root);
    const { engine, configs } = recordingEngine(id, directory);
    const direct = recordingEngine(id, directory);
    assert.equal((await fsp.stat(directory)).mode & 0o222, 0);
    await withoutInstallation(context, async () => {
      await engine.prepare();
      assert.equal(configs.length, 0, "preflight does not construct native recognizers");
      const session = await engine.open();
      await session.end();
      await (await direct.engine.open()).end();
    });
    for (const config of [configs[0], direct.configs[0]]) {
      assert.deepEqual(configuredPaths(config, directory).sort(),
        [...files, ...directories].map((name) => join(directory, name)).sort());
    }
    assert.deepEqual(await snapshot(root), before, "contents, markers and metadata are unchanged");
  });

  test(`${id}: missing required inputs reject preflight and open despite a marker`, async (context) => {
    const { root, directory, files, directories } = await fixture(context, id);
    await fsp.writeFile(join(directory, ".complete"), "misleading marker");
    for (const name of [...files, ...directories]) {
      await fsp.rename(join(directory, name), join(root, "withheld"));
      const before = await snapshot(root);
      const { engine, configs } = recordingEngine(id, directory);
      await withoutInstallation(context, async () => {
        await assert.rejects(engine.prepare(), /Prepared model requires readable/);
        await assert.rejects(engine.open(), /Prepared model requires readable/);
      });
      assert.equal(configs.length, 0);
      assert.deepEqual(await snapshot(root), before);
      await fsp.rename(join(root, "withheld"), join(directory, name));
      // A failed offline preflight must remain retryable after correction.
      await engine.prepare();
    }
  });
}

for (const [Engine, model] of [
  [SherpaEngine, "en-2023-06-21"],
  [SherpaOfflineEngine, "en-parakeet-ultra-int8"],
] as const) {
  test(`${Engine.name}: locations are mutually exclusive and one is required`, () => {
    const binding = bindingWith({});
    const both = { model, binding, modelRoot: "/unused", preparedModelDirectory: "/prepared" };
    // @ts-expect-error A caller cannot supply both locations.
    assert.throws(() => new Engine(both), /exactly one/);
    // @ts-expect-error A caller must supply one location.
    assert.throws(() => new Engine({ model, binding }), /exactly one/);
  });

  test(`${Engine.name}: empty files and directories in place of files reject`, async (context) => {
    const { directory, files } = await fixture(context, model);
    const path = join(directory, files[0]!);
    await fsp.writeFile(path, "");
    await assert.rejects(recordingEngine(model, directory).engine.prepare(), /nonempty file/);
    await fsp.unlink(path);
    await fsp.mkdir(path);
    await assert.rejects(recordingEngine(model, directory).engine.open(), /nonempty file/);
  });

  test(`${Engine.name}: unreadable required files reject without installation`, async (context) => {
    const { directory, files } = await fixture(context, model);
    const path = join(directory, files[0]!);
    await fsp.chmod(path, 0o000);
    try {
      await withoutInstallation(context, async () => {
        await assert.rejects(recordingEngine(model, directory).engine.prepare(), /nonempty file/);
        await assert.rejects(recordingEngine(model, directory).engine.open(), /nonempty file/);
      });
    } finally {
      await fsp.chmod(path, 0o644);
    }
  });
}
