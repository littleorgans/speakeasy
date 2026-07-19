import { performance } from "node:perf_hooks";
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
 * own — a whole string is planned up front (planSegments), a token stream is cut
 * on the fly (planSegmentsStream), and both feed the shared synthPipeline. The
 * only Cartesia-specific piece is CartesiaSynth: one /tts/bytes request per
 * sentence returning raw float32 PCM.
 *
 * No new npm dependency (global fetch). The API key is read from
 * process.env.CARTESIA_API_KEY at call time only, never stored, logged, or
 * echoed in an error. fetch and the key getter are injectable for testing.
 */

export const CARTESIA_BASE_URL = "https://api.cartesia.ai";
export const CARTESIA_VERSION = "2025-04-16";
export const DEFAULT_CARTESIA_MODEL = "sonic-3.5";
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

/** Map a voice name to its id, pass a UUID through, reject anything else. */
export function resolveCartesiaVoice(voice: string): string {
  const named = CARTESIA_VOICES[voice.toLowerCase()];
  if (named) {
    return named;
  }
  if (UUID_RE.test(voice)) {
    return voice;
  }
  throw new Error(
    `unknown Cartesia voice "${voice}"; use a voice UUID or one of: ${Object.keys(CARTESIA_VOICES).join(", ")}`,
  );
}

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

export type CartesiaOptions = {
  fetch?: FetchLike;
  apiKey?: () => string | undefined;
  baseUrl?: string;
};

export class CartesiaTextToSpeech implements TextToSpeech {
  readonly #fetch: FetchLike;
  readonly #apiKey: () => string | undefined;
  readonly #baseUrl: string;

  constructor(options: CartesiaOptions = {}) {
    this.#fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.#apiKey = options.apiKey ?? (() => process.env.CARTESIA_API_KEY);
    this.#baseUrl = options.baseUrl ?? CARTESIA_BASE_URL;
  }

  async open(config: TTSConfig = {}): Promise<TTSSession> {
    const model = config.model ?? DEFAULT_CARTESIA_MODEL;
    const voiceId = resolveCartesiaVoice(
      typeof config.voice === "string" ? config.voice : DEFAULT_CARTESIA_VOICE,
    );
    const synth = new CartesiaSynth(
      this.#fetch,
      this.#apiKey,
      this.#baseUrl,
      model,
      voiceId,
    );
    return new CartesiaSession(synth, config.speed ?? 1);
  }
}

class CartesiaSession implements TTSSession {
  readonly #synth: SegmentSynth;
  readonly #speed: number;

  constructor(synth: SegmentSynth, speed: number) {
    this.#synth = synth;
    this.#speed = speed;
  }

  speak(text: AsyncIterable<string> | string): AsyncIterable<AudioSegment> {
    const segments =
      typeof text === "string"
        ? fromArray(planSegments(text))
        : planSegmentsStream(text);
    return synthPipeline(segments, this.#synth, this.#speed);
  }

  async close(): Promise<void> {
    // Stateless HTTP per sentence; nothing to tear down.
  }
}

class CartesiaSynth implements SegmentSynth {
  readonly #fetch: FetchLike;
  readonly #apiKey: () => string | undefined;
  readonly #baseUrl: string;
  readonly #model: string;
  readonly #voiceId: string;

  constructor(
    fetchImpl: FetchLike,
    apiKey: () => string | undefined,
    baseUrl: string,
    model: string,
    voiceId: string,
  ) {
    this.#fetch = fetchImpl;
    this.#apiKey = apiKey;
    this.#baseUrl = baseUrl;
    this.#model = model;
    this.#voiceId = voiceId;
  }

  async synth(request: SynthRequest): Promise<SynthResult> {
    const key = this.#apiKey();
    if (!key) {
      throw new Error("CARTESIA_API_KEY is not set");
    }
    const start = performance.now();
    const response = await this.#fetch(`${this.#baseUrl}/tts/bytes`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "cartesia-version": CARTESIA_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model_id: this.#model,
        transcript: request.text,
        voice: { mode: "id", id: this.#voiceId },
        output_format: {
          container: "raw",
          encoding: "pcm_f32le",
          sample_rate: CARTESIA_SAMPLE_RATE,
        },
        language: "en",
      }),
    });
    if (!response.ok) {
      throw new Error(
        `Cartesia request failed: HTTP ${response.status}${await errorDetail(response)}`,
      );
    }
    // Raw pcm_f32le: a direct Float32Array view is correct on little-endian
    // hardware (Apple Silicon / x86), the only targets here.
    const samples = new Float32Array(await response.arrayBuffer());
    const totalSynthMs = performance.now() - start;
    const audioDurationMs = (samples.length / CARTESIA_SAMPLE_RATE) * 1_000;
    return {
      samples,
      sampleRate: CARTESIA_SAMPLE_RATE,
      firstAudioMs: undefined,
      totalSynthMs,
      audioDurationMs,
      rtf: totalSynthMs / (audioDurationMs || 1),
    };
  }
}

/** Best-effort server error text; never includes the key. */
async function errorDetail(response: Response): Promise<string> {
  try {
    const text = (await response.text()).trim();
    return text ? ` - ${text.slice(0, 300)}` : "";
  } catch {
    return "";
  }
}
