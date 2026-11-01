import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSetupRuntime, parseSetupRuntimeLocations, readBrowserVoiceOptions } from "./server-options.ts";

test("server options keep the model and use the cascade TTS voice", () => {
  assert.deepEqual(
    readBrowserVoiceOptions([], {
      SPEAKEASY_MODEL: "operator-model",
      SPEAKEASY_TTS: "sherpa",
      SPEAKEASY_TTS_MODEL: "kokoro-v0.19",
      SPEAKEASY_TTS_VOICE: "7",
      SPEAKEASY_VOICE: "marin",
    }),
    {
      port: 4317,
      runtimeConfig: {
        llmModel: "operator-model",
        ttsEngine: "sherpa",
        ttsModel: "kokoro-v0.19",
        voice: "7",
      },
    },
  );
});

test("prompt reader flags come as a pair", () => {
  assert.deepEqual(
    readBrowserVoiceOptions(["--prompts", "set-a.json", "--out", "/tmp/takes"], {})
      .promptReader,
    { promptsPath: "set-a.json", outputDir: "/tmp/takes" },
  );
  assert.equal(readBrowserVoiceOptions([], {}).promptReader, undefined);
  assert.throws(
    () => readBrowserVoiceOptions(["--prompts", "set-a.json"], {}),
    /go together/,
  );
  assert.throws(
    () => readBrowserVoiceOptions(["--out", "--prompts", "x"], {}),
    /--out needs a value/,
  );
});

test("hotwords flag selects an external list and requires its path", () => {
  assert.equal(readBrowserVoiceOptions(["--hotwords", "../speech/words.txt"], {}).hotwordsPath,
    "../speech/words.txt");
  assert.equal(readBrowserVoiceOptions([], {}).hotwordsPath, undefined);
  for (const argv of [["--hotwords"], ["--hotwords", "--port", "0"]]) {
    assert.throws(() => readBrowserVoiceOptions(argv, {}), /--hotwords needs a value/);
  }
});

test("runtime locations are explicit server configuration with portable absent defaults", () => {
  const options = readBrowserVoiceOptions(["--setup-runtime", "runtime.json", "--output-root", "/tmp/voices"], {});
  assert.equal(options.setupRuntimeFile, "runtime.json");
  assert.equal(options.outputRoot, "/tmp/voices");
  assert.equal(readBrowserVoiceOptions([], { SPEAKEASY_SETUP_RUNTIME: "/tmp/runtime.json" }).setupRuntimeFile,
    "/tmp/runtime.json");
  assert.equal(readBrowserVoiceOptions([], {}).setupRuntimeFile, undefined);
  assert.throws(() => readBrowserVoiceOptions(["--setup-runtime"], {}), /needs a value/);
  assert.throws(() => readBrowserVoiceOptions(["--output-root", "relative"], {}), /absolute/);
  const config = { python: { "qwen/context": { pythonPath: "/runtime/python", modelPath: "/models/qwen.gguf",
    libraryPath: "/runtime/libspeech.dylib", loadTimeoutMs: 120000, operationTimeoutMs: 90000 } },
  cpu: { modelRoot: "/models/sherpa" } };
  assert.deepEqual(parseSetupRuntimeLocations(config), config);
  for (const invalid of [{ python: { "qwen/context": { ...config.python["qwen/context"], libraryPath: "relative" } } },
    { python: { "sweep/score-2": config.python["qwen/context"] } },
    { python: { "qwen/context": { ...config.python["qwen/context"], apiKey: "fake" } } },
    { python: { "qwen/context": { ...config.python["qwen/context"], operationTimeoutMs: -1 } } }]) {
    assert.throws(() => parseSetupRuntimeLocations(invalid));
  }
});

test("bootstrap constructs actual lazy transport and reports missing files without spawning", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-host-paths-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const modelPath = join(root, "snapshot");
  await mkdir(modelPath);
  await writeFile(join(modelPath, "config.json"), "{}");
  await writeFile(join(modelPath, "model.safetensors"), "fake weights never loaded");
  const runtimeFile = join(root, "runtime.json");
  await writeFile(runtimeFile, JSON.stringify({ python: {
    "mlx-parakeet/v3-greedy": { pythonPath: process.execPath, modelPath },
  }, cpu: { modelRoot: join(root, "missing") } }));
  const runtime = await createSetupRuntime(runtimeFile);
  assert.equal(runtime.executor.constructor.name, "SetupWorkers");
  assert.equal(runtime.readiness().find((entry) => entry.setupId === "mlx-parakeet/v3-greedy")!.ready, true);
  assert.equal(runtime.readiness().find((entry) => entry.setupId === "sweep/score-2")!.ready, false);
  await runtime.executor.close();
});
