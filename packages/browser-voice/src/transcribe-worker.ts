import { createLocalSherpaEngine, loadSherpaBinding, SHERPA_ROOT } from "@speakeasy/speech-io";
import {
  createSherpaEngine, isAssetComplete, resolveModelPaths, resolveSherpaModel, type Hotwords,
} from "@speakeasy/speech-io/stt";
import {
  TakeTranscription, instrumentCpuBinding, preflightCpuModel, validateCpuRuntime, type CpuMeasurements,
} from "./take-transcription.ts";
import {
  CPU_PRESET, TRANSCRIBE_MODEL_ID, TranscriptionError, validateCpuSamples,
  type CpuReady, type TranscribeReply, type TranscribeRequest,
} from "./transcriber.ts";

const fixed = process.argv[2] === "--cpu-preset=sweep/score-2";
const modelRoot = fixed ? process.argv[3] ?? SHERPA_ROOT : SHERPA_ROOT;
let measurements: CpuMeasurements = { modelConstructionMs: null, processingMs: null };
let assets: Record<string, string> = {};
const transcriber = new TakeTranscription({
  mode: fixed ? CPU_PRESET.mode : "legacy",
  hotwordsPath: fixed ? undefined : process.argv[2],
  createEngine: loadEngine,
  measurements: () => measurements,
});

// The parent going away ends the worker too, including during model startup.
process.on("disconnect", () => process.exit(0));
void start();

async function start(): Promise<void> {
  if (fixed) {
    try {
      const load = await transcriber.prepare();
      send({ type: "ready", version: 1, preset: CPU_PRESET, load, assets } satisfies CpuReady);
    } catch (error) {
      send({ type: "startup-error", error: message(error) });
      process.disconnect();
      return;
    }
  }
  process.on("message", (request: unknown) => { void answer(request); });
}

async function answer(request: unknown): Promise<void> {
  if (typeof request !== "object" || request === null ||
      !Number.isSafeInteger((request as TranscribeRequest).id) || (request as TranscribeRequest).id <= 0 ||
      !((request as TranscribeRequest).samples instanceof Float32Array)) {
    process.disconnect();
    return;
  }
  const { id, samples } = request as TranscribeRequest;
  let reply: TranscribeReply;
  try {
    if (fixed) validateCpuSamples(samples);
    reply = { id, ...(await transcriber.transcribe(samples)) };
  } catch (error) {
    reply = { id, error: message(error),
      ...(error instanceof TranscriptionError && error.hotwords ? { hotwords: error.hotwords } : {}) };
  }
  send(reply);
}

async function loadEngine(hotwords: Hotwords | undefined) {
  if (fixed) {
    assets = await preflightCpuModel(modelRoot);
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
