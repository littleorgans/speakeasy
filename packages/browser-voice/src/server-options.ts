import type { RuntimeConfig, TtsEngine } from "@speakeasy/convo-engine";
import type { PromptReaderConfig } from "./prompt-reader.ts";
import { constants } from "node:fs";
import { access, readdir, readFile, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { exactFields, isRecord } from "./http-guards.ts";
import { getSetup, SETUPS } from "./setup-registry.ts";
import { isSetupId, type SetupId } from "./setup-protocol.ts";
import { SetupWorkers, type PythonSetupId, type SetupWorkersOptions } from "./setup-workers.ts";
import { ChildTranscriber, CPU_SETUP_IDS, isCpuSetupId } from "./transcriber.ts";
import type { SetupReadiness } from "./setup-routes.ts";
import { isWorkerOutput, LOCAL_OUTPUT_MODELS, type LocalOutputModelId } from "../../speech-io/src/tts/catalogue.ts";
import { createWorkerVoiceSynth, type VoiceRuntimeLocation } from "./voice-workers.ts";

/** The one native location a Python setup needs beyond its model. */
const EXTRA_LOCATIONS: Partial<Record<PythonSetupId, "libraryPath" | "runnerPath">> = {
  "qwen/context": "libraryPath", "hex/unified": "runnerPath", "hex/vocabulary": "runnerPath",
  "cohere/transcribe": "runnerPath", "apple/speech": "runnerPath",
};
/** Setups whose weights the system manages: no model path, no measured size. */
const SYSTEM_MANAGED = new Set<PythonSetupId>(["apple/speech"]);
/** The one native location a worker voice needs beyond its interpreter. */
const VOICE_EXTRA_LOCATIONS: Partial<Record<LocalOutputModelId, "runnerPath">> = {
  "piper-kristin-split": "runnerPath", "kokoro-coreml": "runnerPath", "qwen3-tts-0.6b": "runnerPath",
};
const DEFAULT_SETUP_TIMEOUT_MS = 120_000;
/** Gitignored, machine specific; read when no flag or environment value names another. */
export const DEFAULT_SETUP_RUNTIME = "models/road-test-runtime.json";
export const DEFAULT_OUTPUT_ROOT = "models/tts";
export type SetupRuntimeLocations = {
  python?: Partial<Record<PythonSetupId, { pythonPath: string; modelPath?: string; libraryPath?: string;
    runnerPath?: string; loadTimeoutMs?: number; operationTimeoutMs?: number }>>;
  cpu?: { modelRoot: string };
  voices?: Partial<Record<LocalOutputModelId, VoiceRuntimeLocation>>;
};

export function readBrowserVoiceOptions(
  argv: string[],
  env: NodeJS.ProcessEnv,
): { port: number; runtimeConfig: RuntimeConfig; promptReader?: PromptReaderConfig; hotwordsPath?: string;
  setupRuntimeFile: string; setupRuntimeDefault: boolean; outputRoot: string; home: "room" | "road-test" } {
  const portFlag = readFlag(argv, "--port");
  const port = portFlag === undefined ? 4317 : Number(portFlag);
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
  exactFields(value, ["python", "cpu", "voices"]);
  const result: SetupRuntimeLocations = {};
  if (value.python !== undefined) {
    if (!isRecord(value.python)) throw new Error("python runtimes must be an object");
    result.python = {};
    for (const [id, config] of Object.entries(value.python)) {
      if (!isSetupId(id) || isCpuSetupId(id) || !isRecord(config)) throw new Error("unknown Python setup");
      const extra = EXTRA_LOCATIONS[id];
      const system = SYSTEM_MANAGED.has(id);
      exactFields(config, ["pythonPath", ...(system ? [] : ["modelPath"]), ...(extra ? [extra] : []),
        "loadTimeoutMs", "operationTimeoutMs"]);
      const parsed = { pythonPath: absoluteLocation(config.pythonPath),
        ...(system ? {} : { modelPath: absoluteLocation(config.modelPath) }),
        ...(extra ? { [extra]: absoluteLocation(config[extra]) } : {}) };
      result.python[id] = { ...parsed, ...deadlines(config) };
    }
  }
  if (value.voices !== undefined) {
    if (!isRecord(value.voices)) throw new Error("voice runtimes must be an object");
    result.voices = {};
    for (const [id, config] of Object.entries(value.voices)) {
      const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === id);
      if (!model || !isWorkerOutput(model.id) || !isRecord(config)) throw new Error("unknown worker voice");
      const extra = VOICE_EXTRA_LOCATIONS[model.id];
      exactFields(config, ["pythonPath", ...(extra ? [extra] : []), "loadTimeoutMs", "operationTimeoutMs"]);
      result.voices[model.id] = { pythonPath: absoluteLocation(config.pythonPath),
        ...(extra ? { [extra]: absoluteLocation(config[extra]) } : {}), ...deadlines(config) };
    }
  }
  if (value.cpu !== undefined) {
    if (!isRecord(value.cpu)) throw new Error("cpu runtime must be an object");
    exactFields(value.cpu, ["modelRoot"]);
    result.cpu = { modelRoot: absoluteLocation(value.cpu.modelRoot) };
  }
  return result;
}

function deadlines(config: Record<string, unknown>): { loadTimeoutMs?: number; operationTimeoutMs?: number } {
  for (const key of ["loadTimeoutMs", "operationTimeoutMs"] as const) {
    if (config[key] !== undefined && (!Number.isSafeInteger(config[key]) || Number(config[key]) < 1)) {
      throw new Error("worker deadlines must be positive integer milliseconds");
    }
  }
  return { ...(config.loadTimeoutMs === undefined ? {} : { loadTimeoutMs: Number(config.loadTimeoutMs) }),
    ...(config.operationTimeoutMs === undefined ? {} : { operationTimeoutMs: Number(config.operationTimeoutMs) }) };
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
      backendOptions: { ...(config.modelPath ? { modelPath: config.modelPath } : {}),
        ...(config.libraryPath ? { libraryPath: config.libraryPath } : {}),
        ...(config.runnerPath ? { runnerPath: config.runnerPath } : {}) },
      expectedIdentity: { setupId: id }, loadTimeoutMs: config.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationTimeoutMs: config.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  const modelRoot = locations.cpu?.modelRoot ?? resolve("models/sherpa");
  const cpu: SetupWorkersOptions["cpu"] = Object.fromEntries(CPU_SETUP_IDS.map((mode) => [mode, new ChildTranscriber({
    mode, modelRoot, startupTimeoutMs: DEFAULT_SETUP_TIMEOUT_MS, timeoutMs: DEFAULT_SETUP_TIMEOUT_MS })]));
  const executor = new SetupWorkers({ python, cpu });
  const setupReadiness = await inspectSetupReadiness(locations, modelRoot, missingFile);
  const voices = await inspectVoiceRuntimes(locations.voices ?? {});
  return { executor, readiness: () => setupReadiness, missingFile, voices: {
    /** Whether a worker voice's runtime files exist; its payload is checked with the catalogue. */
    configured: (id: LocalOutputModelId) => voices.has(id),
    createSynth: (id: LocalOutputModelId, outputRoot: string) => {
      const location = locations.voices?.[id];
      if (!location || !voices.has(id)) return Promise.reject(new Error("Voice runtime is not configured"));
      return createWorkerVoiceSynth(id, location, outputRoot);
    },
  } };
}

/** Worker voices whose interpreter, and native runner where one is needed, exist and run. */
async function inspectVoiceRuntimes(voices: NonNullable<SetupRuntimeLocations["voices"]>): Promise<Set<LocalOutputModelId>> {
  const ready = new Set<LocalOutputModelId>();
  for (const [id, location] of Object.entries(voices) as [LocalOutputModelId, VoiceRuntimeLocation][]) {
    try {
      for (const path of [location.pythonPath, ...(location.runnerPath ? [location.runnerPath] : [])]) {
        await access(path, constants.X_OK);
        if (!(await stat(path)).isFile()) throw new Error();
      }
      ready.add(id);
    } catch { /* Unavailable: the catalogue reports the runtime as missing. */ }
  }
  return ready;
}

async function inspectSetupReadiness(locations: SetupRuntimeLocations, modelRoot: string,
  missingFile?: string): Promise<SetupReadiness[]> {
  const readiness: SetupReadiness[] = [];
  for (const setup of SETUPS) {
    const config = isCpuSetupId(setup.id) ? undefined : locations.python?.[setup.id];
    let sizeBytes: number | undefined;
    try {
      if (isCpuSetupId(setup.id)) {
        const root = join(modelRoot, "parakeet-ultra-sherpa-onnx-int8");
        for (const name of ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", ".complete"]) {
          if (!(await stat(join(root, name))).isFile()) throw new Error();
        }
        sizeBytes = await modelBytes(root);
      } else {
        if (!config) throw new Error();
        await access(config.pythonPath, constants.X_OK);
        if (!(await stat(config.pythonPath)).isFile()) throw new Error();
        if (!config.modelPath) {
          // System managed weights: the worker reserves and installs them at load; no size to measure.
          if (!config.runnerPath) throw new Error();
          await access(config.runnerPath, constants.X_OK);
        } else {
          const model = await stat(config.modelPath);
          if (setup.id === "qwen/context") {
            if (!model.isFile() || !config.libraryPath || !(await stat(config.libraryPath)).isFile()) throw new Error();
          } else if (EXTRA_LOCATIONS[setup.id] === "runnerPath") {
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
      }
      readiness.push({ setupId: setup.id, ready: true, ...(sizeBytes === undefined ? {} : { sizeBytes }) });
    } catch {
      readiness.push({ setupId: setup.id, ready: false, reason: !config && missingFile && !isCpuSetupId(setup.id)
        ? `setup runtime file missing: ${missingFile}` : "runtime or model files need setup" });
    }
    readiness.at(-1)!.deadlines = { loadMs: config?.loadTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS,
      operationMs: config?.operationTimeoutMs ?? DEFAULT_SETUP_TIMEOUT_MS };
  }
  // A composite setup loads its base setup too: it needs the base ready, and its card counts both.
  for (const setup of SETUPS) {
    const entry = readiness.find((candidate) => candidate.setupId === setup.id)!;
    if (!setup.baseSetup || !entry.ready) continue;
    const base = readiness.find((candidate) => candidate.setupId === setup.baseSetup)!;
    if (!base.ready) {
      entry.ready = false;
      entry.reason = `${getSetup(setup.baseSetup).label} needs setup`;
      delete entry.sizeBytes;
    } else if (entry.sizeBytes !== undefined && base.sizeBytes !== undefined) entry.sizeBytes += base.sizeBytes;
  }
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

/** The last occurrence wins, so flags after `pnpm road-test --` override the script's own. */
function readFlag(argv: string[], flag: string): string | undefined {
  const index = argv.lastIndexOf(flag);
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
