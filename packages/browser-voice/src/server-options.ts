import type { RuntimeConfig, TtsEngine } from "@speakeasy/convo-engine";
import type { PromptReaderConfig } from "./prompt-reader.ts";
import { constants } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { exactFields, isRecord } from "./http-guards.ts";
import { SETUPS } from "./setup-registry.ts";
import { isSetupId, type SetupId } from "./setup-protocol.ts";
import { SetupWorkers, type SetupWorkersOptions } from "./setup-workers.ts";
import { ChildTranscriber } from "./transcriber.ts";
import type { SetupReadiness } from "./setup-routes.ts";

type PythonSetupId = Exclude<SetupId, "sweep/score-2">;
const DEFAULT_SETUP_TIMEOUT_MS = 120_000;
export type SetupRuntimeLocations = {
  python?: Partial<Record<PythonSetupId, { pythonPath: string; modelPath: string; libraryPath?: string;
    loadTimeoutMs?: number; operationTimeoutMs?: number }>>;
  cpu?: { modelRoot: string };
};

export function readBrowserVoiceOptions(
  argv: string[],
  env: NodeJS.ProcessEnv,
): { port: number; runtimeConfig: RuntimeConfig; promptReader?: PromptReaderConfig; hotwordsPath?: string;
  setupRuntimeFile?: string; outputRoot?: string } {
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
  if ((promptsPath === undefined) !== (outputDir === undefined)) {
    throw new Error("--prompts <file> and --out <dir> go together");
  }
  return {
    port,
    ...(setupRuntimeFile === undefined ? {} : { setupRuntimeFile }),
    ...(outputRoot === undefined ? {} : { outputRoot: absoluteLocation(outputRoot) }),
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
      exactFields(config, ["pythonPath", "modelPath", ...(id === "qwen/context" ? ["libraryPath"] : []),
        "loadTimeoutMs", "operationTimeoutMs"]);
      const parsed = { pythonPath: absoluteLocation(config.pythonPath), modelPath: absoluteLocation(config.modelPath),
        ...(id === "qwen/context" ? { libraryPath: absoluteLocation(config.libraryPath) } : {}) };
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

/** File checks only. Weight identity and runtime attestation remain in each worker. */
export async function createSetupRuntime(file?: string) {
  let locations: SetupRuntimeLocations = {};
  if (file !== undefined) {
    if ((await stat(file)).size > 16_384) throw new Error("setup runtime configuration exceeds limit");
    locations = parseSetupRuntimeLocations(JSON.parse(await readFile(file, "utf8")));
  }
  const python: SetupWorkersOptions["python"] = {};
  for (const [id, config] of Object.entries(locations.python ?? {})) {
    python[id as PythonSetupId] = { pythonPath: config.pythonPath,
      backendOptions: { modelPath: config.modelPath, ...(config.libraryPath ? { libraryPath: config.libraryPath } : {}) },
      expectedIdentity: { setupId: id }, loadTimeoutMs: config.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationTimeoutMs: config.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  const modelRoot = locations.cpu?.modelRoot ?? resolve("models/sherpa");
  const executor = new SetupWorkers({ python, cpu: new ChildTranscriber({ mode: "sweep/score-2", modelRoot,
    startupTimeoutMs: DEFAULT_SETUP_TIMEOUT_MS, timeoutMs: DEFAULT_SETUP_TIMEOUT_MS }) });
  const setupReadiness = await inspectSetupReadiness(locations, modelRoot);
  return { executor, readiness: () => setupReadiness };
}

async function inspectSetupReadiness(locations: SetupRuntimeLocations, modelRoot: string): Promise<SetupReadiness[]> {
  const readiness: SetupReadiness[] = [];
  for (const setup of SETUPS) {
    const config = setup.id === "sweep/score-2" ? undefined : locations.python?.[setup.id];
    try {
      if (setup.id === "sweep/score-2") {
        const root = join(modelRoot, "parakeet-ultra-sherpa-onnx-int8");
        for (const name of ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", ".complete"]) {
          if (!(await stat(join(root, name))).isFile()) throw new Error();
        }
      } else {
        if (!config) throw new Error();
        await access(config.pythonPath, constants.X_OK);
        if (!(await stat(config.pythonPath)).isFile()) throw new Error();
        const model = await stat(config.modelPath);
        if (setup.id === "qwen/context") {
          if (!model.isFile() || !config.libraryPath || !(await stat(config.libraryPath)).isFile()) throw new Error();
        } else {
          if (!model.isDirectory()) throw new Error();
          for (const name of ["config.json", "model.safetensors"]) {
            if (!(await stat(join(config.modelPath, name))).isFile()) throw new Error();
          }
        }
      }
      readiness.push({ setupId: setup.id, ready: true });
    } catch { readiness.push({ setupId: setup.id, ready: false, reason: "runtime or model files need setup" }); }
    readiness.at(-1)!.deadlines = { loadMs: config?.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationMs: config?.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  const ultra = readiness.find((entry) => entry.setupId === "mlx-parakeet/ultra-greedy")!;
  const cleanup = readiness.find((entry) => entry.setupId === "mlx-cleanup/qwen-targeted")!;
  if (!ultra.ready) { cleanup.ready = false; cleanup.reason = "Ultra dependency needs setup"; }
  return readiness;
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
