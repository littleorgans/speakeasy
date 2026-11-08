import { createLocalSherpaEngine, loadSherpaBinding, SHERPA_ROOT } from "@speakeasy/speech-io";
import {
  createSherpaEngine, isAssetComplete, resolveModelPaths, resolveSherpaModel, type Hotwords,
} from "@speakeasy/speech-io/stt";
import {
  TakeTranscription, instrumentCpuBinding, preflightCpuModel, validateCpuRuntime, type CpuMeasurements,
} from "./take-transcription.ts";
import {
  CPU_PRESETS, cpuNames, isCpuSetupId, TRANSCRIBE_MODEL_ID, TranscriptionError, validateCpuSamples,
  type CpuReady, type PrepareRequest, type TranscribeReply, type TranscribeRequest,
} from "./transcriber.ts";

const presetId = process.argv[2]?.startsWith("--cpu-preset=") ? process.argv[2].slice("--cpu-preset=".length) : undefined;
if (presetId !== undefined && !isCpuSetupId(presetId)) throw new Error("Unknown CPU preset");
const preset = presetId === undefined ? undefined : CPU_PRESETS[presetId];
const fixed = preset !== undefined;
const modelRoot = fixed ? process.argv[3] ?? SHERPA_ROOT : SHERPA_ROOT;
let measurements: CpuMeasurements = { modelConstructionMs: null, processingMs: null };
// Once per child: a names change rebuilds the recognizer without hashing the model again.
let assets: Promise<Record<string, string>> | undefined;
const transcriber = new TakeTranscription({
  mode: preset?.mode ?? "legacy",
  createEngine: loadEngine,
  measurements: () => measurements,
});

// The parent going away ends the worker too, including during model startup.
process.on("disconnect", () => process.exit(0));
process.on("message", (request: unknown) => { void answer(request); });

async function answer(request: unknown): Promise<void> {
  const parsed = parseRequest(request);
  if (!parsed) {
    process.disconnect?.();
    return;
  }
  const { id, names, samples } = parsed;
  let reply: TranscribeReply;
  try {
    if (!samples) {
      const load = await transcriber.prepare(names);
      reply = { id, type: "ready", version: 1, preset: preset!, load, assets: await assets! } satisfies CpuReady & { id: number };
    } else {
      if (fixed) validateCpuSamples(samples);
      reply = { id, ...(await transcriber.transcribe(samples, names)) };
    }
  } catch (error) {
    reply = { id, error: message(error),
      ...(error instanceof TranscriptionError && error.hotwords ? { hotwords: error.hotwords } : {}) };
  }
  send(reply);
}

/** A prepare (fixed mode only) or a take, each with its names; anything else is a parent bug. */
function parseRequest(value: unknown): { id: number; names: readonly string[]; samples?: Float32Array } | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const { id, type, names, samples } = value as Partial<TranscribeRequest & PrepareRequest>;
  if (!Number.isSafeInteger(id) || id! <= 0) return undefined;
  if (type === "prepare" ? !fixed : !(samples instanceof Float32Array)) return undefined;
  try {
    return { id: id!, names: cpuNames(names), ...(type === "prepare" ? {} : { samples }) };
  } catch { return undefined; }
}

async function loadEngine(hotwords: Hotwords | undefined) {
  if (preset) {
    assets ??= preflightCpuModel(modelRoot);
    await assets;
    const native = loadSherpaBinding();
    validateCpuRuntime(native);
    const instrumented = instrumentCpuBinding(native);
    measurements = instrumented.measurements;
    return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot, binding: instrumented.binding, hotwords });
  }
  const model = resolveSherpaModel(TRANSCRIBE_MODEL_ID);
  if (!(await isAssetComplete(resolveModelPaths(model, SHERPA_ROOT)))) {
    throw new Error(`${model.name} is not installed under ${SHERPA_ROOT}`);
  }
  return createLocalSherpaEngine(TRANSCRIBE_MODEL_ID, "off", { hotwords });
}

function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function send(reply: unknown): void {
  if (process.connected) process.send?.(reply, () => {});
}
