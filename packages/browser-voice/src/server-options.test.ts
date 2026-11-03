import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createSetupRuntime, DEFAULT_OUTPUT_ROOT, DEFAULT_SETUP_RUNTIME, parseSetupRuntimeLocations,
  readBrowserVoiceOptions } from "./server-options.ts";
import { SETUPS } from "./setup-registry.ts";
import { CPU_SETUP_IDS, isCpuSetupId } from "./transcriber.ts";
import { isWorkerOutput, LOCAL_OUTPUT_MODELS } from "../../speech-io/src/tts/catalogue.ts";

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

test("a system managed setup takes no model path, and is ready with its runner but no measured size", async () => {
  const root = await mkdtemp(join(tmpdir(), "system-managed-"));
  try {
    assert.throws(() => parseSetupRuntimeLocations({ python: { "apple/speech": { pythonPath: process.execPath,
      modelPath: "/m", runnerPath: process.execPath } } }), /unexpected|field/i);
    const runtimeFile = join(root, "runtime.json");
    await writeFile(runtimeFile, JSON.stringify({ python: { "apple/speech": { pythonPath: process.execPath,
      runnerPath: process.execPath } } }));
    const runtime = await createSetupRuntime(runtimeFile);
    const apple = runtime.readiness().find((entry) => entry.setupId === "apple/speech")!;
    assert.equal(apple.ready, true);
    assert.equal(apple.sizeBytes, undefined, "the card says system managed instead");
    await runtime.executor.close();
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("the committed example runtime file parses and names every Python setup and worker voice", async () => {
  const example = JSON.parse(await readFile(new URL("../road-test-runtime.example.json", import.meta.url), "utf8"));
  const parsed = parseSetupRuntimeLocations(example);
  assert.deepEqual(Object.keys(parsed.python!).sort(),
    SETUPS.map((setup) => setup.id).filter((id) => !isCpuSetupId(id)).sort());
  assert.deepEqual(Object.keys(parsed.voices!).sort(),
    LOCAL_OUTPUT_MODELS.map((model) => model.id).filter(isWorkerOutput).sort());
});

test("worker voices take an interpreter from the runtime file; only an existing one makes them configured", async (t) => {
  const python = { pythonPath: process.execPath };
  assert.throws(() => parseSetupRuntimeLocations({ voices: { "kokoro-v1.0": python } }), /unknown worker voice/,
    "a sherpa voice has no worker runtime");
  assert.throws(() => parseSetupRuntimeLocations({ voices: { "kokoro-mlx-bf16": { ...python, modelPath: "/m" } } }));
  assert.throws(() => parseSetupRuntimeLocations({ voices: { "kokoro-mlx-bf16": { pythonPath: "relative/python" } } }));
  assert.throws(() => parseSetupRuntimeLocations({ voices: { "kokoro-coreml": { ...python, runnerPath: "relative" } } }),
    /absolute/, "a native voice needs an absolute runner");
  assert.throws(() => parseSetupRuntimeLocations({ voices: { "kokoro-mlx-bf16": { ...python, runnerPath: "/r" } } }),
    "only native voices take a runner");
  const root = await mkdtemp(join(tmpdir(), "speakeasy-voice-runtime-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const [pythonPath, configured] of [[process.execPath, true], [join(root, "missing-python"), false]] as const) {
    const file = join(root, `runtime-${configured}.json`);
    await writeFile(file, JSON.stringify({ voices: { "kokoro-mlx-bf16": { pythonPath, loadTimeoutMs: 5000 } } }));
    const runtime = await createSetupRuntime(file);
    t.after(() => runtime.executor.close());
    assert.equal(runtime.voices.configured("kokoro-mlx-bf16"), configured);
    if (!configured) await assert.rejects(runtime.voices.createSynth("kokoro-mlx-bf16", root), /not configured/);
  }
});

test("a composite setup needs its base setup and its card counts both models", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "speakeasy-composite-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const snapshot = async (name: string, weights: string) => {
    const path = join(root, name);
    await mkdir(path);
    await writeFile(join(path, "config.json"), "{}");
    await writeFile(join(path, "model.safetensors"), weights);
    return path;
  };
  const ultra = { pythonPath: process.execPath, modelPath: await snapshot("ultra", "ultra weights") };
  const cleanup = { pythonPath: process.execPath, modelPath: await snapshot("cleanup", "cleanup") };
  const ready = async (python: Record<string, unknown>) => {
    const runtimeFile = join(root, `runtime-${Object.keys(python).length}.json`);
    await writeFile(runtimeFile, JSON.stringify({ python }));
    const runtime = await createSetupRuntime(runtimeFile);
    t.after(() => runtime.executor.close());
    return runtime.readiness().find((entry) => entry.setupId === "mlx-cleanup/qwen-targeted")!;
  };
  const both = await ready({ "mlx-parakeet/ultra-greedy": ultra, "mlx-cleanup/qwen-targeted": cleanup });
  assert.equal(both.ready, true);
  assert.equal(both.sizeBytes, 2 * "{}".length + "ultra weights".length + "cleanup".length);
  const alone = await ready({ "mlx-cleanup/qwen-targeted": cleanup });
  assert.equal(alone.ready, false);
  assert.equal(alone.reason, "Parakeet Ultra MLX needs setup");
  assert.equal(alone.sizeBytes, undefined);
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
  for (const id of CPU_SETUP_IDS) assert.equal(runtime.readiness().find((entry) => entry.setupId === id)!.ready, false);
  await runtime.executor.close();
});

test("flags after pnpm road-test -- win over the script's own, which stay the default", () => {
  const script = ["--road-test", "--port", "4870"];
  assert.equal(readBrowserVoiceOptions(script, {}).port, 4870);
  // pnpm 12 passes the separator through: the server sees "--" then the override.
  const options = readBrowserVoiceOptions([...script, "--", "--port", "4873", "--setup-runtime", "/a.json",
    "--setup-runtime", "/b.json", "--output-root", "/x", "--output-root", "/y"], {});
  assert.deepEqual([options.port, options.home, options.setupRuntimeFile, options.outputRoot], [4873, "road-test", "/b.json", "/y"]);
  assert.throws(() => readBrowserVoiceOptions([...script, "--", "--port"], {}), /--port needs a value/);
  assert.throws(() => readBrowserVoiceOptions([...script, "--port", "70000"], {}), /--port must be an integer/);
});
