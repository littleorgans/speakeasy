import { performance } from "node:perf_hooks";
import {
  ProviderError,
  safeProviderError,
  type SpeechTiming,
} from "@speakeasy/speech-io/providers";
import type {
  AudioSegment,
  TextToSpeech,
  TTSConfig,
  TTSSession,
} from "./contract.ts";
import {
  fromArray,
  planSegments,
  planSegmentsStream,
  synthPipeline,
  type SegmentSynth,
} from "./stream.ts";
import type { SynthRequest, SynthResult } from "./synth.ts";

/**
 * Cartesia Sonic adapter: a hosted, natural-voice TTS behind the same
 * TextToSpeech contract as the local sherpa engine. It owns no pipeline of its
 * own. A whole string is planned up front (planSegments), a token stream is cut
 * on the fly (planSegmentsStream), and both feed the shared synthPipeline. The
 * only Cartesia-specific piece is CartesiaSynth: one /tts/bytes request per
 * sentence returning raw float32 PCM.
 *
 * No new npm dependency (global fetch). The API key is read from
 * process.env.CARTESIA_API_KEY at call time only, never stored, logged, or
 * echoed in an error. fetch and the key getter are injectable for testing.
 */

export const CARTESIA_BASE_URL = "https://api.cartesia.ai";
export const CARTESIA_VERSION = "2026-08-14";
// Preserve existing conversation defaults; the browser factory chooses its model.
export const DEFAULT_CARTESIA_MODEL = "sonic-3.5";
export const CARTESIA_BROWSER_MODEL = "sonic-3.6-2026-08-27";
export const CARTESIA_SPEED = Object.freeze({ min: 0.6, max: 1.5 });
/** Cartesia raw pcm_f32le output rate. AudioSegment carries this downstream. */
export const CARTESIA_SAMPLE_RATE = 44_100;

/**
 * Named English voices (ids from GET /voices, 2026-07-07). --voice accepts one
 * of these names or a raw voice UUID; anything else is rejected loudly.
 */
export const CARTESIA_VOICES: Record<string, string> = {
  katie: "f786b574-daa5-4673-aa0c-cbe3e8534c02",
  daniel: "47c38ca4-5f35-497b-b1a3-415245fb35e1",
  skylar: "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
  ronald: "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
  gemma: "62ae83ad-4f6a-430b-af41-a9bede9286ca",
};
export const DEFAULT_CARTESIA_VOICE = "katie";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const LEGACY_VERSION = "2025-04-16";
const SUPPORTED_MODELS = new Set([
  "sonic-3", "sonic-3-2026-01-12",
  "sonic-3.5", "sonic-3.5-2026-05-04",
  "sonic-3.6", CARTESIA_BROWSER_MODEL,
]);

/** Map a voice name to its id, pass a UUID through, reject anything else. */
export function resolveCartesiaVoice(voice: string): string {
  if (typeof voice !== "string") throw new ProviderError("CONFIG_UNSUPPORTED");
  const name = voice.toLowerCase();
  if (Object.hasOwn(CARTESIA_VOICES, name)) {
    return CARTESIA_VOICES[name]!;
  }
  if (UUID_RE.test(voice)) {
    return voice;
  }
  throw new ProviderError("CONFIG_UNSUPPORTED");
}

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

export type CartesiaOptions = {
  fetch?: FetchLike;
  apiKey?: () => string | undefined;
  baseUrl?: string;
  apiVersion?: typeof CARTESIA_VERSION | typeof LEGACY_VERSION;
  /** Trusted service revocation signal, independent of portable preferences. */
  signal?: AbortSignal;
  /** Recheck selection/source authorization immediately before every request. */
  beforeRequest?: () => void | Promise<void>;
  /** Monotonic milliseconds; injectable for deterministic timing checks. */
  now?: () => number;
  /** Bytes has no returned speech transcript or processing/network metric. */
  onRequestComplete?: (result: {
    model: string;
    voice: string;
    text: string;
    timing: Pick<SpeechTiming,
      "providerRequestWallMs" | "providerProcessingMs" | "providerNetworkMs">;
  }) => void;
};

export class CartesiaTextToSpeech implements TextToSpeech {
  readonly #options: CartesiaOptions;

  constructor(options: CartesiaOptions = {}) {
    if (options.apiVersion !== undefined && options.apiVersion !== CARTESIA_VERSION &&
        options.apiVersion !== LEGACY_VERSION) throw new ProviderError("CONFIG_UNSUPPORTED");
    this.#options = { ...options };
  }

  async open(config: TTSConfig = {}): Promise<TTSSession> {
    const model = config.model ?? DEFAULT_CARTESIA_MODEL;
    const speed = config.speed ?? 1;
    if (!SUPPORTED_MODELS.has(model) || typeof speed !== "number" ||
        !Number.isFinite(speed) || speed < CARTESIA_SPEED.min || speed > CARTESIA_SPEED.max ||
        (config.voice !== undefined && typeof config.voice !== "string")) {
      throw new ProviderError("CONFIG_UNSUPPORTED");
    }
    const voiceId = resolveCartesiaVoice(config.voice ?? DEFAULT_CARTESIA_VOICE);
    return new CartesiaSession(new CartesiaSynth(this.#options, model, voiceId), speed,
      this.#options.signal, this.#options.now);
  }
}

class CartesiaSession implements TTSSession {
  readonly #synth: CartesiaSynth;
  readonly #speed: number;
  readonly #controller = new AbortController();
  readonly #signal: AbortSignal;
  readonly #now: () => number;

  constructor(synth: CartesiaSynth, speed: number, signal?: AbortSignal,
    now: () => number = () => performance.now()) {
    this.#synth = synth;
    this.#speed = speed;
    this.#now = now;
    this.#signal = signal
      ? AbortSignal.any([this.#controller.signal, signal]) : this.#controller.signal;
  }

  speak(text: AsyncIterable<string> | string): AsyncIterableIterator<AudioSegment> {
    const segments =
      typeof text === "string"
        ? fromArray(planSegments(text))
        : planSegmentsStream(text);
    const controller = new AbortController();
    const signal = AbortSignal.any([this.#signal, controller.signal]);
    let demand = 0;
    let finished = false;
    let resume: (() => void) | undefined;
    const synth: SegmentSynth = {
      synth: (request) => {
        const pending = (async () => {
          // The shared pipeline prefetches. Hold its next request until next()
          // expresses consumer demand so break/close can stop transmission.
          if (demand === 0) {
            await abortable(() => new Promise<void>((resolve) => { resume = resolve; }), signal);
          }
          demand -= 1;
          return this.#synth.synth(request, signal);
        })();
        // A prefetched request may be cancelled while the pipeline is at yield.
        void pending.catch(() => {});
        return pending;
      },
    };
    const pipeline = synthPipeline(segments, synth, this.#speed, this.#now);
    signal.addEventListener("abort", () => {
      // Neither a stalled token source nor a transport can delay close().
      void pipeline.return(undefined).catch(() => {});
      void segments.return(undefined).catch(() => {});
    }, { once: true });
    return {
      [Symbol.asyncIterator]() { return this; },
      async next() {
        if (finished) return { done: true, value: undefined };
        try {
          if (signal.aborted) throw abortError();
          demand += 1;
          resume?.();
          resume = undefined;
          const result = await abortable(() => pipeline.next(), signal);
          if (result.done) {
            finished = true;
            controller.abort();
          }
          return result;
        } catch (error) {
          finished = true;
          controller.abort();
          throw error;
        }
      },
      async return() {
        finished = true;
        controller.abort();
        return { done: true, value: undefined };
      },
    };
  }

  async close(): Promise<void> {
    this.#controller.abort();
  }
}

class CartesiaSynth {
  readonly #options: CartesiaOptions;
  readonly #model: string;
  readonly #voiceId: string;

  constructor(
    options: CartesiaOptions,
    model: string,
    voiceId: string,
  ) {
    this.#options = options;
    this.#model = model;
    this.#voiceId = voiceId;
  }

  async synth(request: SynthRequest, signal: AbortSignal): Promise<SynthResult> {
    try {
      return await this.#request(request, signal);
    } catch (error) {
      if (signal.aborted) throw abortError();
      throw safeProviderError(error);
    }
  }

  async #request(request: SynthRequest, signal: AbortSignal): Promise<SynthResult> {
    await abortable(async () => { await this.#options.beforeRequest?.(); }, signal);
    if (signal.aborted) throw abortError();
    const key = (this.#options.apiKey ?? (() => process.env.CARTESIA_API_KEY))();
    if (!key) throw new ProviderError("PROVIDER_UNAVAILABLE");
    const version = this.#options.apiVersion ?? CARTESIA_VERSION;
    const init: RequestInit = {
      method: "POST",
      signal,
      headers: {
        authorization: `Bearer ${key}`,
        "cartesia-version": version,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model_id: this.#model,
        transcript: request.text,
        voice: version === LEGACY_VERSION
          ? { mode: "id", id: this.#voiceId } : { id: this.#voiceId },
        output_format: {
          container: "raw",
          encoding: "pcm_f32le",
          sample_rate: CARTESIA_SAMPLE_RATE,
        },
        language: "en",
        generation_config: { speed: request.speed ?? 1 },
      }),
    };
    const now = this.#options.now ?? (() => performance.now());
    const fetchImpl = this.#options.fetch ?? ((url: string, init: RequestInit) => fetch(url, init));
    const start = now();
    const response = await abortable(
      () => fetchImpl(`${this.#options.baseUrl ?? CARTESIA_BASE_URL}/tts/bytes`, init), signal);
    if (!response.ok) {
      void response.body?.cancel().catch(() => {});
      throw new ProviderError("PROVIDER_FAILED");
    }
    const bytes = await abortable(() => response.arrayBuffer(), signal);
    if (signal.aborted) throw abortError();
    // Request wall ends at complete body receipt, before decoding/callbacks.
    const requestWallMs = now() - start;
    if (!Number.isFinite(requestWallMs) || requestWallMs < 0 ||
        bytes.byteLength === 0 || bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new ProviderError("PROVIDER_FAILED");
    }
    // Raw pcm_f32le: a direct Float32Array view is correct on little-endian
    // hardware (Apple Silicon / x86), the only targets here.
    const samples = new Float32Array(bytes);
    if (!samples.every(Number.isFinite)) throw new ProviderError("PROVIDER_FAILED");
    this.#options.onRequestComplete?.({
      model: this.#model, voice: this.#voiceId, text: request.text,
      timing: {
        providerRequestWallMs: { status: "measured", ms: requestWallMs,
          clock: "server", boundary: "HTTP dispatch to complete response body receipt" },
        providerProcessingMs: { status: "unavailable", reason: "Cartesia bytes exposes no processing metric" },
        providerNetworkMs: { status: "unavailable", reason: "Cartesia bytes exposes no network boundary" },
      },
    });
    const audioDurationMs = (samples.length / CARTESIA_SAMPLE_RATE) * 1_000;
    return {
      samples,
      sampleRate: CARTESIA_SAMPLE_RATE,
      firstAudioMs: undefined,
      // Legacy pipeline compatibility: synthMs is request wall, not compute.
      totalSynthMs: requestWallMs,
      audioDurationMs,
      rtf: requestWallMs / (audioDurationMs || 1),
    };
  }
}

function abortError(): DOMException {
  return new DOMException("Cartesia speech cancelled", "AbortError");
}

/** Bound cancellation even when an injected transport ignores AbortSignal. */
async function abortable<T>(operation: () => Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) throw abortError();
  let onAbort!: () => void;
  try {
    return await new Promise<T>((resolve, reject) => {
      onAbort = () => reject(abortError());
      signal.addEventListener("abort", onAbort, { once: true });
      operation().then(resolve, reject);
    });
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}
