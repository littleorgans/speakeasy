import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { ProviderError, safeProviderError } from "@speakeasy/speech-io/providers";
import type { SpeechInputConfig, SpeechTiming } from "@speakeasy/speech-io/providers";

export const CARTESIA_STT_MODEL = "ink-whisper";
export const CARTESIA_STT_VERSION = "2026-08-14";
export const CARTESIA_STT_SAMPLE_RATE = 16_000;
/** Matches the host's 120 second completed take bound. */
export const CARTESIA_STT_MAX_SAMPLES = 120 * CARTESIA_STT_SAMPLE_RATE;
export const CARTESIA_STT_MAX_RESPONSE_BYTES = 1_048_576;
const REQUEST_URL = "https://api.cartesia.ai/stt?encoding=pcm_f32le&sample_rate=16000";
const DEFAULT_TIMEOUT_MS = 60_000;

export type CartesiaSttFetch = (url: string, init: RequestInit) => Promise<Response>;
export type CartesiaSttOptions = {
  /** Caller supplies server credentials and may recheck selection here. */
  getKey: () => string | undefined;
  fetch?: CartesiaSttFetch;
  /** Monotonic milliseconds on the server clock. */
  clock?: () => number;
  timeoutMs?: number;
};

export type CartesiaSttMetadata = {
  config: SpeechInputConfig;
  apiVersion: string;
  encoding: "pcm_f32le";
  sampleRate: number;
  channels: 1;
  maxSamples: number;
};

export type CartesiaSttResult = {
  config: SpeechInputConfig;
  /** Complete provider text, preserving whitespace. Empty is a valid result. */
  text: string;
  sampleRate: number;
  channels: 1;
  sampleCount: number;
  pcmBytes: number;
  /** SHA256 of the exact original little endian Float32 upload bytes. */
  pcmSha256: string;
  observations: {
    requestId?: string;
    language?: string;
    /** Provider reported input audio duration, never processing time. */
    audioDurationSeconds?: number;
  };
  timings: Pick<SpeechTiming,
    "providerRequestWallMs" | "providerProcessingMs" | "providerNetworkMs">;
};

/**
 * One completed mono 16 kHz canonical take per HTTP request. The calling
 * service must authorize source and explicit provider selection before calling
 * transcribe. Abort its signal on revocation. This adapter has no source policy,
 * context prompt, realtime feeding, fallback or retry path.
 * Contract: https://docs.cartesia.ai/api-reference/stt/transcribe
 */
export class CartesiaBatchTranscriber {
  readonly #fetch: CartesiaSttFetch;
  readonly #getKey: CartesiaSttOptions["getKey"];
  readonly #clock: () => number;
  readonly #timeoutMs: number;

  constructor(options: CartesiaSttOptions) {
    this.#fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.#getKey = options.getKey;
    this.#clock = options.clock ?? (() => performance.now());
    this.#timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (!Number.isFinite(this.#timeoutMs) || this.#timeoutMs <= 0 || this.#timeoutMs > 2_147_483_647) {
      throw new RangeError("Cartesia request timeout must be a finite positive timer duration.");
    }
  }

  /** Pure catalogue data. No credentials, transport or clock are consulted. */
  metadata(): CartesiaSttMetadata {
    return {
      config: { schema: 1, provider: { kind: "cartesia", model: CARTESIA_STT_MODEL } },
      apiVersion: CARTESIA_STT_VERSION, encoding: "pcm_f32le",
      sampleRate: CARTESIA_STT_SAMPLE_RATE, channels: 1, maxSamples: CARTESIA_STT_MAX_SAMPLES,
    };
  }

  async transcribe(
    samples: Float32Array,
    options: { signal?: AbortSignal } = {},
  ): Promise<CartesiaSttResult> {
    if (options.signal?.aborted) throw cancelled();
    const bytes = encodeCompletedTake(samples);
    const pcmSha256 = createHash("sha256").update(bytes).digest("hex");
    const form = new FormData();
    form.append("file", new Blob([bytes], { type: "application/octet-stream" }), "take.f32le");
    form.append("model", CARTESIA_STT_MODEL);
    form.append("language", "en");

    const controller = new AbortController();
    const onAbort = () => controller.abort(cancelled());
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const timeout = setTimeout(() => controller.abort(
      new DOMException("Cartesia transcription timed out.", "TimeoutError"),
    ), this.#timeoutMs);
    try {
      const key = this.#getKey();
      if (typeof key !== "string" || !key.trim()) throw new ProviderError("PROVIDER_UNAVAILABLE");
      controller.signal.throwIfAborted();
      const start = this.#clock();
      if (!Number.isFinite(start)) throw new ProviderError("TIMING_INVALID");
      controller.signal.throwIfAborted();
      const response = await abortable(this.#fetch(REQUEST_URL, {
        method: "POST", body: form, signal: controller.signal, redirect: "error",
        headers: { authorization: `Bearer ${key}`, "cartesia-version": CARTESIA_STT_VERSION },
      }).then((response) => {
        // A test transport can ignore abort and resolve after the caller leaves.
        if (controller.signal.aborted) {
          void response.body?.cancel().catch(() => {});
          controller.signal.throwIfAborted();
        }
        return response;
      }), controller.signal);
      if (!response.ok) {
        void response.body?.cancel().catch(() => {});
        throw new ProviderError("PROVIDER_FAILED");
      }
      const chunks = await readBoundedBody(response, controller.signal);
      const requestWallMs = this.#clock() - start;
      if (!Number.isFinite(requestWallMs) || requestWallMs < 0) throw new ProviderError("TIMING_INVALID");
      const body = new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks));
      const { text, observations } = parseTranscript(JSON.parse(body));
      controller.signal.throwIfAborted();
      return {
        config: this.metadata().config, text,
        sampleRate: CARTESIA_STT_SAMPLE_RATE, channels: 1,
        sampleCount: bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
        pcmBytes: bytes.byteLength, pcmSha256, observations,
        timings: {
          providerRequestWallMs: { status: "measured", ms: requestWallMs, clock: "server",
            boundary: "HTTP dispatch through complete response body receipt, before JSON parsing" },
          providerProcessingMs: { status: "unavailable", reason: "Batch STT documents no processing metric." },
          providerNetworkMs: { status: "unavailable", reason: "Batch STT documents no independent network boundary." },
        },
      };
    } catch (error) {
      if (controller.signal.aborted) throw controller.signal.reason;
      controller.abort();
      throw safeProviderError(error);
    } finally {
      clearTimeout(timeout);
      options.signal?.removeEventListener("abort", onAbort);
    }
  }
}

function cancelled(): DOMException {
  return new DOMException("Cartesia transcription cancelled.", "AbortError");
}

/** Own one bounded snapshot, including offset views and signed zero. */
function encodeCompletedTake(samples: Float32Array): Uint8Array<ArrayBuffer> {
  if (!(samples instanceof Float32Array) || !samples.length || samples.length > CARTESIA_STT_MAX_SAMPLES) {
    throw new RangeError("Cartesia expects a bounded completed Float32 take.");
  }
  const bytes = new Uint8Array(samples.byteLength);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index]!;
    if (!Number.isFinite(sample)) throw new RangeError("Cartesia PCM samples must be finite.");
    view.setFloat32(index * Float32Array.BYTES_PER_ELEMENT, sample, true);
  }
  return bytes;
}

/** Settle promptly even when a fake transport ignores the request's signal. */
function abortable<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener("abort", onAbort, { once: true });
    pending.then((value) => {
      signal.removeEventListener("abort", onAbort);
      resolve(value);
    }, (error: unknown) => {
      signal.removeEventListener("abort", onAbort);
      reject(error);
    });
  });
}

async function readBoundedBody(response: Response, signal: AbortSignal): Promise<Uint8Array[]> {
  if (!response.body) throw new ProviderError("PROVIDER_FAILED");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  let complete = false;
  try {
    while (true) {
      const chunk = await abortable(reader.read(), signal);
      if (chunk.done) { complete = true; break; }
      length += chunk.value.byteLength;
      if (length > CARTESIA_STT_MAX_RESPONSE_BYTES) throw new ProviderError("PROVIDER_FAILED");
      chunks.push(chunk.value);
    }
    return chunks;
  } finally {
    if (!complete) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function parseTranscript(value: unknown): Pick<CartesiaSttResult, "text" | "observations"> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ProviderError("PROVIDER_FAILED");
  }
  const payload = value as Record<string, unknown>;
  if (payload.type !== "transcript" || typeof payload.text !== "string") {
    throw new ProviderError("PROVIDER_FAILED");
  }
  const observations: CartesiaSttResult["observations"] = {};
  for (const [field, target] of [["request_id", "requestId"], ["language", "language"]] as const) {
    const entry = payload[field];
    if (entry !== undefined) {
      if (typeof entry !== "string" || entry.length > 256 || /[\u0000-\u001f\u007f]/u.test(entry)) {
        throw new ProviderError("PROVIDER_FAILED");
      }
      observations[target] = entry;
    }
  }
  if (payload.duration !== undefined) {
    if (typeof payload.duration !== "number" || !Number.isFinite(payload.duration) || payload.duration < 0) {
      throw new ProviderError("PROVIDER_FAILED");
    }
    observations.audioDurationSeconds = payload.duration;
  }
  return { text: payload.text, observations };
}
