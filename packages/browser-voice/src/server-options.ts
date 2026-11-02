import type { RuntimeConfig, TtsEngine } from "@speakeasy/convo-engine";
import type { PromptReaderConfig } from "./prompt-reader.ts";
import { constants } from "node:fs";
import { access, readdir, readFile, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { exactFields, isRecord } from "./http-guards.ts";
import { SETUPS } from "./setup-registry.ts";
import { isSetupId, type SetupId } from "./setup-protocol.ts";
import { SetupWorkers, type SetupWorkersOptions } from "./setup-workers.ts";
import { ChildTranscriber } from "./transcriber.ts";
import type { SetupReadiness } from "./setup-routes.ts";

type PythonSetupId = Exclude<SetupId, "sweep/score-2">;
/** The one native location a Python setup needs beyond its model. */
const EXTRA_LOCATIONS: Partial<Record<PythonSetupId, "libraryPath" | "runnerPath">> = {
  "qwen/context": "libraryPath", "hex/unified": "runnerPath",
};
const DEFAULT_SETUP_TIMEOUT_MS = 120_000;
/** Gitignored, machine specific; read when no flag or environment value names another. */
export const DEFAULT_SETUP_RUNTIME = "models/road-test-runtime.json";
export const DEFAULT_OUTPUT_ROOT = "models/tts";
export type SetupRuntimeLocations = {
  python?: Partial<Record<PythonSetupId, { pythonPath: string; modelPath: string; libraryPath?: string;
    runnerPath?: string; loadTimeoutMs?: number; operationTimeoutMs?: number }>>;
  cpu?: { modelRoot: string };
};

export function readBrowserVoiceOptions(
  argv: string[],
  env: NodeJS.ProcessEnv,
): { port: number; runtimeConfig: RuntimeConfig; promptReader?: PromptReaderConfig; hotwordsPath?: string;
  setupRuntimeFile: string; setupRuntimeDefault: boolean; outputRoot: string; home: "room" | "road-test" } {
  const portFlag = argv.indexOf("--port");
  const port = portFlag === -1 ? 4317 : Number(argv[portFlag + 1]);
  if (!Number.isInteger(port) || port < 0 || port > 65_535) {
    throw new Error("--port must be an integer from 0 to 65535");
  }
  const promptsPath = readFlag(argv, "--prompts");
  const outputDir = readFlag(argv, "--out");
  const hotwordsPath = readFlag(argv, "--hotwords");
  const setupRuntimeFile = readFlag(argv, "--setup-runtime") ?? env.SPEAKEASY_SETUP_RUNTIME;
  const outputRoot = readFlag(argv, "--output-root") ?? env.SPEAKEASY_OUTPUT_ROOT;
  // Defaults resolve from the repo root, where every command runs.
  if ((promptsPath === undefined) !== (outputDir === undefined)) {
    throw new Error("--prompts <file> and --out <dir> go together");
  }
  return {
    port,
    home: argv.includes("--road-test") ? "road-test" : "room",
    setupRuntimeFile: setupRuntimeFile ?? resolve(DEFAULT_SETUP_RUNTIME),
    setupRuntimeDefault: setupRuntimeFile === undefined,
    outputRoot: outputRoot === undefined ? resolve(DEFAULT_OUTPUT_ROOT) : absoluteLocation(outputRoot),
    ...(hotwordsPath !== undefined ? { hotwordsPath } : {}),
    ...(promptsPath !== undefined && outputDir !== undefined
      ? { promptReader: { promptsPath, outputDir } }
      : {}),
    runtimeConfig: {
      ttsEngine: readChoice(env.SPEAKEASY_TTS, ["sherpa", "cartesia"]),
      llmModel: env.SPEAKEASY_MODEL,
      ttsModel: env.SPEAKEASY_TTS_MODEL,
      voice: env.SPEAKEASY_TTS_VOICE,
    },
  };
}

/** Pure server configuration: paths only, never portable provider preferences. */
export function parseSetupRuntimeLocations(value: unknown): SetupRuntimeLocations {
  if (!isRecord(value)) throw new Error("setup runtime must be an object");
  exactFields(value, ["python", "cpu"]);
  const result: SetupRuntimeLocations = {};
  if (value.python !== undefined) {
    if (!isRecord(value.python)) throw new Error("python runtimes must be an object");
    result.python = {};
    for (const [id, config] of Object.entries(value.python)) {
      if (!isSetupId(id) || id === "sweep/score-2" || !isRecord(config)) throw new Error("unknown Python setup");
      const extra = EXTRA_LOCATIONS[id];
      exactFields(config, ["pythonPath", "modelPath", ...(extra ? [extra] : []), "loadTimeoutMs", "operationTimeoutMs"]);
      const parsed = { pythonPath: absoluteLocation(config.pythonPath), modelPath: absoluteLocation(config.modelPath),
        ...(extra ? { [extra]: absoluteLocation(config[extra]) } : {}) };
      for (const key of ["loadTimeoutMs", "operationTimeoutMs"] as const) {
        if (config[key] !== undefined && (!Number.isSafeInteger(config[key]) || Number(config[key]) < 1)) {
          throw new Error("worker deadlines must be positive integer milliseconds");
        }
      }
      result.python[id] = { ...parsed,
        ...(config.loadTimeoutMs === undefined ? {} : { loadTimeoutMs: Number(config.loadTimeoutMs) }),
        ...(config.operationTimeoutMs === undefined ? {} : { operationTimeoutMs: Number(config.operationTimeoutMs) }) };
    }
  }
  if (value.cpu !== undefined) {
    if (!isRecord(value.cpu)) throw new Error("cpu runtime must be an object");
    exactFields(value.cpu, ["modelRoot"]);
    result.cpu = { modelRoot: absoluteLocation(value.cpu.modelRoot) };
  }
  return result;
}

function absoluteLocation(value: unknown): string {
  if (typeof value !== "string" || !value.trim() || !isAbsolute(value) || value.includes("\0")) {
    throw new Error("runtime locations must be absolute server paths");
  }
  return value;
}

/**
 * File checks only. Weight identity and runtime attestation remain in each worker.
 * An optional file may be absent: the host still serves, and each setup that
 * needs it reports the missing path as its readiness reason.
 */
export async function createSetupRuntime(file?: string, { optional = false } = {}) {
  let locations: SetupRuntimeLocations = {};
  let missingFile: string | undefined;
  if (file !== undefined) {
    const info = await stat(file).catch((error: NodeJS.ErrnoException) => {
      if (optional && error.code === "ENOENT") return undefined;
      throw error;
    });
    if (!info) missingFile = resolve(file);
    else {
      if (info.size > 16_384) throw new Error("setup runtime configuration exceeds limit");
      locations = parseSetupRuntimeLocations(JSON.parse(await readFile(file, "utf8")));
    }
  }
  const python: SetupWorkersOptions["python"] = {};
  for (const [id, config] of Object.entries(locations.python ?? {})) {
    python[id as PythonSetupId] = { pythonPath: config.pythonPath,
      backendOptions: { modelPath: config.modelPath, ...(config.libraryPath ? { libraryPath: config.libraryPath } : {}),
        ...(config.runnerPath ? { runnerPath: config.runnerPath } : {}) },
      expectedIdentity: { setupId: id }, loadTimeoutMs: config.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationTimeoutMs: config.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  const modelRoot = locations.cpu?.modelRoot ?? resolve("models/sherpa");
  const executor = new SetupWorkers({ python, cpu: new ChildTranscriber({ mode: "sweep/score-2", modelRoot,
    startupTimeoutMs: DEFAULT_SETUP_TIMEOUT_MS, timeoutMs: DEFAULT_SETUP_TIMEOUT_MS }) });
  const setupReadiness = await inspectSetupReadiness(locations, modelRoot, missingFile);
  return { executor, readiness: () => setupReadiness, missingFile };
}

async function inspectSetupReadiness(locations: SetupRuntimeLocations, modelRoot: string,
  missingFile?: string): Promise<SetupReadiness[]> {
  const readiness: SetupReadiness[] = [];
  for (const setup of SETUPS) {
    const config = setup.id === "sweep/score-2" ? undefined : locations.python?.[setup.id];
    let sizeBytes: number | undefined;
    try {
      if (setup.id === "sweep/score-2") {
        const root = join(modelRoot, "parakeet-ultra-sherpa-onnx-int8");
        for (const name of ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", ".complete"]) {
          if (!(await stat(join(root, name))).isFile()) throw new Error();
        }
        sizeBytes = await modelBytes(root);
      } else {
        if (!config) throw new Error();
        await access(config.pythonPath, constants.X_OK);
        if (!(await stat(config.pythonPath)).isFile()) throw new Error();
        const model = await stat(config.modelPath);
        if (setup.id === "qwen/context") {
          if (!model.isFile() || !config.libraryPath || !(await stat(config.libraryPath)).isFile()) throw new Error();
        } else if (setup.id === "hex/unified") {
          if (!model.isFile() || !config.runnerPath) throw new Error();
          await access(config.runnerPath, constants.X_OK);
        } else {
          if (!model.isDirectory()) throw new Error();
          for (const name of ["config.json", "model.safetensors"]) {
            if (!(await stat(join(config.modelPath, name))).isFile()) throw new Error();
          }
        }
        sizeBytes = await modelBytes(config.modelPath);
      }
      readiness.push({ setupId: setup.id, ready: true, sizeBytes });
    } catch {
      readiness.push({ setupId: setup.id, ready: false, reason: !config && missingFile && setup.id !== "sweep/score-2"
        ? `setup runtime file missing: ${missingFile}` : "runtime or model files need setup" });
    }
    readiness.at(-1)!.deadlines = { loadMs: config?.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationMs: config?.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  const ultra = readiness.find((entry) => entry.setupId === "mlx-parakeet/ultra-greedy")!;
  const cleanup = readiness.find((entry) => entry.setupId === "mlx-cleanup/qwen-targeted")!;
  if (!ultra.ready) { cleanup.ready = false; cleanup.reason = "Ultra dependency needs setup"; }
  return readiness;
}

/** Bytes of a model file, or of the files directly inside a model directory. */
async function modelBytes(path: string): Promise<number> {
  const info = await stat(path);
  if (info.isFile()) return info.size;
  let total = 0;
  for (const name of await readdir(path)) {
    const entry = await stat(join(path, name));
    if (entry.isFile()) total += entry.size;
  }
  return total;
}

function readFlag(argv: string[], flag: string): string | undefined {
  const index = argv.indexOf(flag);
  if (index === -1) return undefined;
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${flag} needs a value`);
  return value;
}

function readChoice<T extends TtsEngine>(
  value: string | undefined,
  choices: readonly T[],
): T | undefined {
  if (value === undefined) return undefined;
  const choice = choices.find((candidate) => candidate === value);
  if (choice !== undefined) return choice;
  throw new Error(`${value} must be one of: ${choices.join(", ")}`);
}
