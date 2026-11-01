import { createLocalSherpaEngine, SHERPA_ROOT } from "@speakeasy/speech-io";
import {
  isAssetComplete,
  resolveModelPaths,
  resolveSherpaModel,
  type Hotwords,
} from "@speakeasy/speech-io/stt";
import { TakeTranscription } from "./take-transcription.ts";
import { TRANSCRIBE_MODEL_ID, TranscriptionError, type TranscribeReply, type TranscribeRequest } from "./transcriber.ts";

/**
 * Parakeet Ultra without a silence gate, matching TM. Each request snapshots
 * the selected hotword file immediately; unchanged content reuses the warm recognizer.
 * The model must already sit under models/sherpa; no downloads.
 */

const transcriber = new TakeTranscription({
  hotwordsPath: process.argv[2],
  createEngine: loadEngine,
});

process.on("message", (request: TranscribeRequest) => {
  void answer(request);
});
// The parent going away ends the worker too.
process.on("disconnect", () => process.exit(0));

async function answer({ id, samples }: TranscribeRequest): Promise<void> {
  let reply: TranscribeReply;
  try {
    reply = { id, ...(await transcriber.transcribe(samples)) };
  } catch (error) {
    reply = { id, error: error instanceof Error ? error.message : String(error),
      ...(error instanceof TranscriptionError && error.hotwords ? { hotwords: error.hotwords } : {}) };
  }
  process.send?.(reply);
}

async function loadEngine(hotwords: Hotwords | undefined) {
  const model = resolveSherpaModel(TRANSCRIBE_MODEL_ID);
  if (!(await isAssetComplete(resolveModelPaths(model, SHERPA_ROOT)))) {
    throw new Error(`${model.name} is not installed under ${SHERPA_ROOT}`);
  }
  return createLocalSherpaEngine(TRANSCRIBE_MODEL_ID, "off", { hotwords });
}
