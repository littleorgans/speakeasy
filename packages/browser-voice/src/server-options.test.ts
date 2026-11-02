import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createSetupRuntime, DEFAULT_OUTPUT_ROOT, DEFAULT_SETUP_RUNTIME, parseSetupRuntimeLocations,
  readBrowserVoiceOptions } from "./server-options.ts";
import { SETUPS } from "./setup-registry.ts";

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
      home: "room",
      setupRuntimeFile: resolve(DEFAULT_SETUP_RUNTIME),
      setupRuntimeDefault: true,
      outputRoot: resolve(DEFAULT_OUTPUT_ROOT),
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

test("road test flag serves the road test page at the root", () => {
  assert.equal(readBrowserVoiceOptions(["--road-test", "--port", "4870"], {}).home, "road-test");
  assert.equal(readBrowserVoiceOptions(["--port", "4870"], {}).home, "room");
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
  assert.equal(readBrowserVoiceOptions(["--setup-runtime", "runtime.json"], {}).setupRuntimeDefault, false);
  assert.equal(readBrowserVoiceOptions([], { SPEAKEASY_SETUP_RUNTIME: "/tmp/runtime.json" }).setupRuntimeDefault, false);
  assert.throws(() => readBrowserVoiceOptions(["--setup-runtime"], {}), /needs a value/);
  assert.throws(() => readBrowserVoiceOptions(["--output-root", "relative"], {}), /absolute/);
  const config = { python: { "qwen/context": { pythonPath: "/runtime/python", modelPath: "/models/qwen.gguf",
    libraryPath: "/runtime/libspeech.dylib", loadTimeoutMs: 120000, operationTimeoutMs: 90000 } },
  cpu: { modelRoot: "/models/sherpa" } };
  assert.deepEqual(parseSetupRuntimeLocations(config), config);
  const hex = { python: { "hex/unified": { pythonPath: "/runtime/python", modelPath: "/models/hex.gguf", runnerPath: "/runtime/hex-worker" } } };
  assert.deepEqual(parseSetupRuntimeLocations(hex), hex, "Hex takes its native worker path");
  assert.throws(() => parseSetupRuntimeLocations({ python: { "mlx-parakeet/v3-greedy": hex.python["hex/unified"] } }));
  for (const invalid of [{ python: { "qwen/context": { ...config.python["qwen/context"], libraryPath: "relative" } } },
    { python: { "sweep/score-2": config.python["qwen/context"] } },
    { python: { "qwen/context": { ...config.python["qwen/context"], apiKey: "fake" } } },
    { python: { "qwen/context": { ...config.python["qwen/context"], operationTimeoutMs: -1 } } }]) {
    assert.throws(() => parseSetupRuntimeLocations(invalid));
  }
});

test("with no flag or environment value the runtime file and voices default inside the repo", () => {
  const options = readBrowserVoiceOptions(["--road-test", "--port", "4870"], {});
  assert.equal(options.setupRuntimeFile, resolve("models/road-test-runtime.json"));
  assert.equal(options.setupRuntimeDefault, true);
  assert.equal(options.outputRoot, resolve("models/tts"));
  assert.equal(readBrowserVoiceOptions([], { SPEAKEASY_OUTPUT_ROOT: "/voices" }).outputRoot, "/voices");
  assert.equal(readBrowserVoiceOptions(["--output-root", "/flag"], { SPEAKEASY_OUTPUT_ROOT: "/voices" }).outputRoot, "/flag");
  assert.equal(readBrowserVoiceOptions(["--setup-runtime", "/flag.json"], { SPEAKEASY_SETUP_RUNTIME: "/env.json" }).setupRuntimeFile,
    "/flag.json");
});

test("a missing default runtime file still serves and names the file; a missing explicit one fails", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-runtime-default-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const missing = join(root, "road-test-runtime.json");
  const runtime = await createSetupRuntime(missing, { optional: true });
  t.after(() => runtime.executor.close());
  assert.equal(runtime.missingFile, missing);
  const v3 = runtime.readiness().find((entry) => entry.setupId === "mlx-parakeet/v3-greedy")!;
  assert.equal(v3.ready, false);
  assert.equal(v3.reason, `setup runtime file missing: ${missing}`);
  await assert.rejects(createSetupRuntime(missing), { code: "ENOENT" });
});

test("the committed example runtime file parses and names every Python setup", async () => {
  const example = JSON.parse(await readFile(new URL("../road-test-runtime.example.json", import.meta.url), "utf8"));
  const parsed = parseSetupRuntimeLocations(example);
  assert.deepEqual(Object.keys(parsed.python!).sort(),
    SETUPS.map((setup) => setup.id).filter((id) => id !== "sweep/score-2").sort());
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
  const v3 = runtime.readiness().find((entry) => entry.setupId === "mlx-parakeet/v3-greedy")!;
  assert.equal(v3.ready, true);
  assert.equal(v3.sizeBytes, "{}".length + "fake weights never loaded".length, "card size is measured from the model files");
  assert.equal(runtime.readiness().find((entry) => entry.setupId === "sweep/score-2")!.ready, false);
  await runtime.executor.close();
});
