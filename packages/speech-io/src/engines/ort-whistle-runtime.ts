import { readFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import {
  OrtWhistleEngine,
  type WhistleFiles,
  type WhistleRecognizer,
  type WhistleRuntime,
} from "./ort-whistle.ts";
import type { WhistleModelId } from "./ort-whistle-models.ts";
import { OrtBackend } from "./vendor/whistle/backend.js";
import { parsePack } from "./vendor/whistle/pack.js";
import { Engram, Tokenizer, transcribe } from "./vendor/whistle/whistle.js";

/**
 * speak-easy's own Whistle wiring for the bench and CLI paths: onnxruntime-node
 * from this package's node_modules on the CPU provider with its default
 * threads, and models under the working directory. Loading follows the
 * reference `js/example.mjs` at the vendored revision.
 */

/** On-disk root for downloaded onnxruntime-node models. */
export const ORT_ROOT = join(process.cwd(), "models", "ort");

/** Mel filterbank shape packed in whistle.pack: 257 FFT bins x 80 mels. */
const MEL_FILTERBANK_FLOATS = 257 * 80;

export function loadWhistleRuntime(): WhistleRuntime {
  return {
    version: ort.env.versions.node ?? "unknown",
    createRecognizer: createWhistleRecognizer,
  };
}

export function createLocalWhistleEngine(model: WhistleModelId): OrtWhistleEngine {
  return new OrtWhistleEngine({
    model,
    modelRoot: ORT_ROOT,
    runtime: loadWhistleRuntime(),
  });
}

async function createWhistleRecognizer(files: WhistleFiles): Promise<WhistleRecognizer> {
  const bytes = await readFile(files.pack);
  const pack = parsePack(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
  );
  const filterbankStart = pack.base + pack.man.melfb.src;
  const context = {
    tok: new Tokenizer(pack.man.tok),
    fb: new Float32Array(
      pack.buf.slice(filterbankStart, filterbankStart + MEL_FILTERBANK_FLOATS * 4),
    ),
    engram: new Engram(pack),
  };
  const options = { executionProviders: ["cpu"] };
  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(files.encoder, options),
    ort.InferenceSession.create(files.decoder, options),
  ]);
  const backend = new OrtBackend(ort, encoder, decoder);
  return {
    transcribe: (pcm, language) =>
      transcribe(backend, context, pcm, { timestamps: true, language }),
  };
}
