import { performance } from "node:perf_hooks";
import { ProviderError, type SpeechTiming, type TimingObservation } from "@speakeasy/speech-io/providers";
import { encodeWavFloat32 } from "../bench/wav.ts";

export const OPENAI_FILE_MODEL = "gpt-transcribe";
export const OPENAI_FILE_SAMPLE_RATE = 16_000;
export const OPENAI_FILE_LIMITS = Object.freeze({
  fileBytes: 25_000_000,
  responseBytes: 65_536,
  textBytes: 32_768,
  timeoutMs: 60_000,
});

const TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions";
const PROCESSING_METRIC = "openai-processing-ms";

export type OpenAIFileOptions = {
  getKey: () => string | undefined;
  fetch?: typeof globalThis.fetch;
  clock?: () => number;
  timeoutMs?: number;
};

export type OpenAIFileResult = {
  model: typeof OPENAI_FILE_MODEL;
  text: string;
  languages?: { code: string }[];
  timings: Pick<SpeechTiming,
    "providerRequestWallMs" | "providerProcessingMs" | "providerNetworkMs">;
};

/** Completed canonical 16 kHz mono takes only. The caller owns dispatch consent. */
export class OpenAIFileTranscriber {
  readonly #fetch: typeof globalThis.fetch;
  readonly #getKey: OpenAIFileOptions["getKey"];
  readonly #clock: () => number;
  readonly #timeoutMs: number;

  constructor(options: OpenAIFileOptions) {
    this.#fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
    this.#getKey = options.getKey;
    this.#clock = options.clock ?? (() => performance.now());
    this.#timeoutMs = options.timeoutMs ?? OPENAI_FILE_LIMITS.timeoutMs;
    if (!Number.isFinite(this.#timeoutMs) || this.#timeoutMs <= 0 ||
        this.#timeoutMs > OPENAI_FILE_LIMITS.timeoutMs) {
      throw new ProviderError("CONFIG_UNSUPPORTED");
    }
  }

  async transcribe(
    samples: Float32Array,
    options: { signal?: AbortSignal } = {},
  ): Promise<OpenAIFileResult> {
    if (options.signal?.aborted) throw cancelled();
    let lifetime: ReturnType<typeof requestLifetime> | undefined;
    try {
      const wav = encodeTake(samples);
      const key = this.#getKey();
      if (!key?.trim()) throw new ProviderError("PROVIDER_UNAVAILABLE");
      const body = new FormData();
      body.set("file", new Blob([wav], { type: "audio/wav" }), "take.wav");
      body.set("model", OPENAI_FILE_MODEL);
      body.set("stream", "false");

      lifetime = requestLifetime(options.signal, this.#timeoutMs);
      lifetime.check();
      const started = this.#clock();
      const signal = lifetime.signal;
      const response = await lifetime.race(this.#fetch(TRANSCRIPTIONS_URL, {
        method: "POST",
        headers: { authorization: `Bearer ${key}` },
        body,
        signal,
        redirect: "error",
      }).then((response) => {
        // A transport that ignores abort may deliver its response after cancellation.
        if (signal.aborted) discardBody(response);
        return response;
      }));
      if (!response.ok || response.headers.get("content-type")?.split(";")[0]?.trim() !==
          "application/json") {
        discardBody(response);
        throw new ProviderError("PROVIDER_FAILED");
      }
      const bytes = await readBoundedBody(response, lifetime);
      lifetime.check();
      const wallMs = this.#clock() - started;
      if (!Number.isFinite(wallMs) || wallMs < 0) throw new ProviderError("PROVIDER_FAILED");
      const transcript = parseTranscript(bytes);
      lifetime.check();
      return {
        model: OPENAI_FILE_MODEL,
        ...transcript,
        timings: {
          providerRequestWallMs: { status: "measured", ms: wallMs, clock: "server",
            boundary: "Fetch dispatch through complete response body receipt; excludes WAV encoding and JSON parsing." },
          providerProcessingMs: processingObservation(response.headers),
          providerNetworkMs: { status: "unavailable",
            reason: "No independently supported network timing boundary." },
        },
      };
    } catch (error) {
      lifetime?.check();
      if (options.signal?.aborted) throw cancelled();
      // Fresh errors carry no provider body, transport cause, credential or caller value.
      throw new ProviderError(error instanceof ProviderError ? error.code : "PROVIDER_FAILED");
    } finally {
      lifetime?.dispose();
    }
  }
}

function encodeTake(samples: Float32Array): Uint8Array<ArrayBuffer> {
  if (!(samples instanceof Float32Array) || !samples.length ||
      samples.byteLength > OPENAI_FILE_LIMITS.fileBytes ||
      !samples.every(Number.isFinite)) throw new ProviderError("PROVIDER_FAILED");
  const wav = encodeWavFloat32(samples, OPENAI_FILE_SAMPLE_RATE);
  if (wav.byteLength > OPENAI_FILE_LIMITS.fileBytes) throw new ProviderError("PROVIDER_FAILED");
  // The encoder owns an ArrayBuffer. Copying into Blob preserves its exact bytes.
  return new Uint8Array(wav);
}

function requestLifetime(parent: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let failure: DOMException | undefined;
  let rejectStopped!: (error: DOMException) => void;
  const stopped = new Promise<never>((_resolve, reject) => { rejectStopped = reject; });
  const stop = (error: DOMException) => {
    if (failure) return;
    failure = error;
    controller.abort();
    rejectStopped(error);
  };
  const onAbort = () => stop(cancelled());
  parent?.addEventListener("abort", onAbort, { once: true });
  // Attach before checking an already aborted parent so rejection is always handled.
  void stopped.catch(() => {});
  if (parent?.aborted) onAbort();
  const timer = setTimeout(() => stop(new DOMException(
    "OpenAI transcription request timed out.", "TimeoutError",
  )), timeoutMs);
  return {
    signal: controller.signal,
    check() { if (failure) throw failure; },
    race<T>(operation: Promise<T>): Promise<T> { return Promise.race([operation, stopped]); },
    dispose() { clearTimeout(timer); parent?.removeEventListener("abort", onAbort); },
  };
}

function cancelled(): DOMException {
  return new DOMException("OpenAI transcription cancelled.", "AbortError");
}

function discardBody(response: Response): void {
  void response.body?.cancel().catch(() => {});
}

async function readBoundedBody(
  response: Response,
  lifetime: ReturnType<typeof requestLifetime>,
): Promise<Uint8Array> {
  const declared = response.headers.get("content-length");
  if (!response.body || (declared !== null &&
      (!/^\d+$/.test(declared) || Number(declared) > OPENAI_FILE_LIMITS.responseBytes))) {
    discardBody(response);
    throw new ProviderError("PROVIDER_FAILED");
  }
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  lifetime.signal.addEventListener("abort", cancel, { once: true });
  const bytes = new Uint8Array(OPENAI_FILE_LIMITS.responseBytes);
  let size = 0;
  let complete = false;
  try {
    for (;;) {
      const { done, value } = await lifetime.race(reader.read());
      lifetime.check();
      if (done) break;
      if (size + value.byteLength > bytes.byteLength) throw new ProviderError("PROVIDER_FAILED");
      bytes.set(value, size);
      size += value.byteLength;
    }
    complete = true;
    return bytes.subarray(0, size);
  } finally {
    lifetime.signal.removeEventListener("abort", cancel);
    if (!complete) cancel();
    reader.releaseLock();
  }
}

function parseTranscript(bytes: Uint8Array): Pick<OpenAIFileResult, "text" | "languages"> {
  const value: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      !("text" in value) || typeof value.text !== "string" ||
      Buffer.byteLength(value.text, "utf8") > OPENAI_FILE_LIMITS.textBytes) {
    throw new ProviderError("PROVIDER_FAILED");
  }
  if (!("languages" in value)) return { text: value.text };
  if (!Array.isArray(value.languages) || value.languages.length > 32) {
    throw new ProviderError("PROVIDER_FAILED");
  }
  const languages = value.languages.map((language: unknown) => {
    if (typeof language !== "object" || language === null || !("code" in language) ||
        typeof language.code !== "string" || !language.code.trim() || language.code.length > 64) {
      throw new ProviderError("PROVIDER_FAILED");
    }
    return { code: language.code };
  });
  return { text: value.text, languages };
}

function processingObservation(headers: Headers): TimingObservation {
  const raw = headers.get(PROCESSING_METRIC)?.trim();
  const ms = raw && /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw) ? Number(raw) : NaN;
  return Number.isFinite(ms) && ms >= 0
    ? { status: "provider-reported", ms, metric: PROCESSING_METRIC,
      boundary: "Provider reported HTTP request processing; includes more than isolated model compute.",
      evidenceRef: "https://developers.openai.com/api/reference/overview#debugging-requests" }
    : { status: "unavailable", reason: "No finite nonnegative openai-processing-ms response header." };
}
