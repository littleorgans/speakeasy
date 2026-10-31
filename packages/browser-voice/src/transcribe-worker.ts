import { existsSync } from "node:fs";
import { join } from "node:path";
import { createLocalSherpaEngine, SHERPA_ROOT } from "@speakeasy/speech-io";
import {
  COMPLETE_MARKER,
  isAssetComplete,
  resolveModelPaths,
  resolveSherpaModel,
  type AnySherpaEngine,
  type FinalEvent,
} from "@speakeasy/speech-io/stt";
import { PROMPT_SAMPLE_RATE } from "./prompt-reader.ts";
import type { TranscribeReply, TranscribeRequest } from "./transcriber.ts";

/**
 * Child process behind ChildTranscriber: Canary 180M Flash in English with
 * the registry's silence gate, loaded on the first request and kept warm.
 * Requests decode one at a time. The model and Silero must already sit
 * under models/sherpa; this worker never downloads.
 */

const MODEL = "en-canary-180m-flash-int8";

let engine: Promise<AnySherpaEngine> | undefined;
let queue: Promise<void> = Promise.resolve();

process.on("message", (request: TranscribeRequest) => {
  queue = queue.then(() => answer(request));
});
// The parent going away ends the worker too.
process.on("disconnect", () => process.exit(0));

async function answer({ id, samples }: TranscribeRequest): Promise<void> {
  let reply: TranscribeReply;
  try {
    reply = { id, ...(await transcribe(samples)) };
  } catch (error) {
    reply = { id, error: error instanceof Error ? error.message : String(error) };
  }
  process.send?.(reply);
}

async function transcribe(samples: Float32Array): Promise<{ text: string; decodeMs: number }> {
  engine ??= loadEngine().catch((error: unknown) => {
    engine = undefined;
    throw error;
  });
  const session = await (await engine).open({ sampleRate: PROMPT_SAMPLE_RATE });
  try {
    return await new Promise((resolve, reject) => {
      session.once("final", ({ text, decodeMs }: FinalEvent) => resolve({ text, decodeMs: decodeMs ?? 0 }));
      session.once("empty", ({ decodeMs }: { decodeMs?: number }) => resolve({ text: "", decodeMs: decodeMs ?? 0 }));
      session.once("error", ({ err }: { err: unknown }) => reject(err));
      session.pushAudio(samples);
      session.flush();
    });
  } finally {
    await session.end();
  }
}

async function loadEngine(): Promise<AnySherpaEngine> {
  const model = resolveSherpaModel(MODEL);
  if (!(await isAssetComplete(resolveModelPaths(model, SHERPA_ROOT)))) {
    throw new Error(`${model.name} is not installed under ${SHERPA_ROOT}`);
  }
  if (!existsSync(join(SHERPA_ROOT, "silero-vad", COMPLETE_MARKER))) {
    throw new Error(`silero-vad is not installed under ${SHERPA_ROOT}`);
  }
  const loaded = await createLocalSherpaEngine(MODEL);
  await loaded.prepare();
  return loaded;
}
