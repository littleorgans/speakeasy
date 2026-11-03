import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { join } from "node:path";
import { LOCAL_OUTPUT_MODELS } from "../../speech-io/src/tts/catalogue.ts";
import { LOCAL_TTS_MODELS, payloadDir, TTS_ROOT, type WorkerTtsModel } from "../../speech-io/src/tts/models.ts";
import { HttpError, readTakeSamples, requireLoopback, sendHttpError, sendJson } from "./http-guards.ts";
import { pcmDigest } from "./setup-workers.ts";

export const REFERENCE_ROUTE = "/speech-output/references";
export const PICK_ROUTE = "/speech-output/picks/";
/** Recordings the host holds at once; the oldest is forgotten first. */
export const MAX_REFERENCES = 4;
export const REFERENCE_SECONDS = { min: 1, max: 30 } as const;
export const REFERENCE_RATES = { min: 16_000, max: 96_000 } as const;

export type StoredReference = { sha256: string; samples: Float32Array; sampleRate: number };

/**
 * The owner's voice copy recordings, in this server's memory only: never
 * written to disk, gone on Forget or restart. Keyed by the SHA256 of the PCM
 * the worker receives, so the same recording uploaded again is one entry.
 */
export class VoiceReferenceStore {
  readonly #entries = new Map<string, StoredReference>();

  add(samples: Float32Array, sampleRate: number): { id: string; seconds: number } {
    if (!Number.isSafeInteger(sampleRate) || sampleRate < REFERENCE_RATES.min || sampleRate > REFERENCE_RATES.max) {
      throw new HttpError(400, `sampleRate must be an integer from ${REFERENCE_RATES.min} to ${REFERENCE_RATES.max}`);
    }
    const seconds = samples.length / sampleRate;
    if (seconds < REFERENCE_SECONDS.min || seconds > REFERENCE_SECONDS.max) {
      throw new HttpError(400, `a reference lasts ${REFERENCE_SECONDS.min} to ${REFERENCE_SECONDS.max} seconds`);
    }
    const sha256 = pcmDigest(samples);
    this.#entries.delete(sha256);
    this.#entries.set(sha256, { sha256, samples: samples.slice(), sampleRate });
    while (this.#entries.size > MAX_REFERENCES) this.#entries.delete(this.#entries.keys().next().value!);
    return { id: sha256, seconds };
  }

  get(id: string): StoredReference | undefined {
    return this.#entries.get(id);
  }

  delete(id: string): boolean {
    return this.#entries.delete(id);
  }
}

/** Upload (`POST ?sampleRate=`, Float32 mono) and forget (`DELETE /<id>`), same origin loopback pages only. */
export async function serveVoiceReferences(request: IncomingMessage, response: ServerResponse,
  store: VoiceReferenceStore): Promise<void> {
  try {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (url.pathname === REFERENCE_ROUTE && request.method === "POST") {
      const rate = url.searchParams.get("sampleRate") ?? "";
      if (!/^\d{5}$/.test(rate)) throw new HttpError(400, "sampleRate must be whole hertz");
      const samples = await readTakeSamples(request);
      sendJson(response, 201, store.add(samples, Number(rate)));
    } else if (url.pathname.startsWith(`${REFERENCE_ROUTE}/`) && request.method === "DELETE") {
      requireLoopback(request, true);
      const id = url.pathname.slice(REFERENCE_ROUTE.length + 1);
      if (!/^[0-9a-f]{64}$/.test(id) || !store.delete(id)) sendJson(response, 404, { error: "no such reference" });
      else sendJson(response, 200, { forgotten: id });
    } else sendJson(response, 404, { error: "Not found" });
  } catch (error) { sendHttpError(response, error); }
}

/** A voice copy pick's pinned WAV, refused unless it matches its pin. */
export async function readPinnedPick(payload: WorkerTtsModel, root: string, file: string): Promise<Buffer> {
  const pin = payload.files.find((entry) => entry.name === file);
  const bytes = await readFile(join(payloadDir(payload, root), file));
  if (!pin || createHash("sha256").update(bytes).digest("hex") !== pin.sha256) {
    throw new Error("Voice copy reference differs from its pin");
  }
  return bytes;
}

/** `GET /speech-output/picks/<model>/<voice>`: a pick's WAV for Listen. Picks live under the cache root. */
export async function servePick(request: IncomingMessage, response: ServerResponse): Promise<void> {
  try {
    requireLoopback(request);
    const [modelId, voiceId, extra] = new URL(request.url ?? "/", "http://localhost").pathname
      .slice(PICK_ROUTE.length).split("/");
    const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === modelId);
    const voice = model?.voices.find((entry) => entry.id === voiceId);
    if (request.method !== "GET" || extra !== undefined || !model || !voice || !("file" in voice)) {
      sendJson(response, 404, { error: "no such pick" });
      return;
    }
    const bytes = await readPinnedPick(LOCAL_TTS_MODELS[model.id] as WorkerTtsModel, TTS_ROOT, voice.file);
    response.writeHead(200, { "content-type": "audio/wav", "content-length": bytes.length,
      "cache-control": "no-store", "x-content-type-options": "nosniff" });
    response.end(bytes);
  } catch (error) { sendHttpError(response, error); }
}
