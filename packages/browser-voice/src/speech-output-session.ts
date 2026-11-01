import { createHash } from "node:crypto";
import { ProviderError, parseTimingObservation } from "@speakeasy/speech-io/providers";
import type { SpeechOutputConfig, SpeechTiming, TimingObservation } from "@speakeasy/speech-io/providers";
import type { AudioSegment, TextToSpeech, TTSSession } from "../../speech-io/src/tts/contract.ts";
import type { ProviderPolicy } from "./provider-policy.ts";
import type { SpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { encodeAudioPacket } from "./protocol.ts";
import {
  parseSpeechOutputCommand, safeSpeechOutputError, SpeechOutputError,
  type SampleSegment, type SampleProviderObservation, type SpeechOutputCommand, type SpeechOutputEvent,
} from "./speech-output-protocol.ts";

export type SpeechOutputContext = {
  signal: AbortSignal;
  /** Check before every deferred sentence/request, including adapter lookahead. */
  recheck: () => void;
  apiKey: (() => string) | undefined;
  /** Attach the adapter's per request observations, without inferred metrics. */
  observe: (observation: SampleProviderObservation) => void;
};

export type SpeechOutputSessionOptions = {
  sessionId: string;
  catalogue: () => SpeechOutputCatalogue;
  policy: Pick<ProviderPolicy, "select" | "dispatch" | "revoke">;
  createOutput: (config: SpeechOutputConfig, context: SpeechOutputContext) => TextToSpeech | Promise<TextToSpeech>;
  /** Wait for queued/running native work, even when close only suppressed audio. */
  quiesce: (config: SpeechOutputConfig) => Promise<void>;
  acquire: () => (() => void) | undefined;
  sendEvent: (event: SpeechOutputEvent) => void;
  sendAudio: (packet: Uint8Array) => void;
  now?: () => number;
  timeoutMs?: number;
};

type Trial = {
  command: Extract<SpeechOutputCommand, { type: "speak-sample" }>;
  playbackId: number;
  generation: number;
  admittedAt: number;
  textId: string;
  controller: AbortController;
  release: () => void;
  timer?: ReturnType<typeof setTimeout>;
  session?: TTSSession;
  closing?: Promise<void>;
  iterator?: AsyncIterator<AudioSegment>;
  sampleRate?: number;
  produced: boolean;
  timing: SpeechTiming;
  segments: SampleSegment[];
  requests: SampleProviderObservation[];
  operationSettled: Promise<void>;
};

/** Typed voice trials share TTS, policy and host admission, without STT or an LLM. */
export class SpeechOutputSession {
  readonly #options: SpeechOutputSessionOptions;
  readonly #now: () => number;
  readonly #timeoutMs: number;
  readonly #requests = new Set<string>();
  readonly #retirements = new Set<Promise<void>>();
  #active?: Trial;
  #generation = 0;
  #playbackId = 0;
  #closed = false;

  constructor(options: SpeechOutputSessionOptions) {
    if (!options.sessionId.trim()) throw new SpeechOutputError("COMMAND_INVALID");
    this.#timeoutMs = options.timeoutMs ?? 120_000;
    if (!Number.isSafeInteger(this.#timeoutMs) || this.#timeoutMs < 1 || this.#timeoutMs > 600_000) {
      throw new SpeechOutputError("COMMAND_INVALID");
    }
    this.#options = { ...options };
    this.#now = options.now ?? (() => performance.now());
  }

  /** Applies a command immediately. Synthesis never blocks cancellation or drain. */
  handle(raw: string): void {
    if (this.#closed) return;
    let requestId: string | null = null;
    try {
      const catalogue = this.#options.catalogue();
      const command = parseSpeechOutputCommand(raw, catalogue.registry);
      if (command.type === "playback-drained") { this.#drained(command.playbackId); return; }
      requestId = command.requestId;
      if (command.type === "cancel-sample") {
        if (this.#active?.command.requestId === command.requestId) this.#cancel(this.#active);
        return;
      }
      if (command.catalogueRevision !== catalogue.revision) throw new ProviderError("SELECTION_STALE");
      if (command.type === "select-output") {
        const selected = this.#options.policy.select({ sessionId: this.#options.sessionId,
          direction: "output", catalogueRevision: command.catalogueRevision, config: command.selection });
        this.#emit({ type: "output-selected", requestId, catalogueRevision: selected.catalogueRevision,
          selectionRevision: selected.selectionRevision, selection: structuredClone(command.selection) });
        return;
      }
      this.#start(command, catalogue);
    } catch (error) {
      this.#emit({ type: "sample-error", requestId, ...safeSpeechOutputError(error) });
    }
  }

  /** Abort before closing. A late open is closed on resolution and cannot send. */
  async close(): Promise<void> {
    if (!this.#closed) {
      this.#closed = true;
      if (this.#active) this.#retire(this.#active);
      this.#options.policy.revoke(this.#options.sessionId, "output");
    }
    try { await Promise.all(this.#retirements); }
    catch { throw new ProviderError("PROVIDER_FAILED"); }
  }

  #start(command: Trial["command"], catalogue: SpeechOutputCatalogue): void {
    const admittedAt = this.#now();
    if (this.#active) throw new SpeechOutputError("BUSY");
    if (this.#requests.has(command.requestId)) throw new SpeechOutputError("REQUEST_REUSED");
    const entry = catalogue.output.find(({ config }) => config.provider.kind === command.selection.provider.kind &&
      config.model === command.selection.model);
    if (!entry?.selectable) throw new ProviderError("PROVIDER_UNAVAILABLE");
    if (this.#playbackId === 0xffff_ffff) throw new SpeechOutputError("SESSION_CLOSED");
    const release = this.#options.acquire();
    if (!release) throw new SpeechOutputError("BUSY");
    let settleOperation!: () => void;
    const operationSettled = new Promise<void>((resolve) => { settleOperation = resolve; });
    const job: Trial = {
      command, playbackId: ++this.#playbackId, generation: ++this.#generation, admittedAt,
      textId: createHash("sha256").update(command.text).digest("hex"),
      controller: new AbortController(), release, produced: false, timing: unavailableTiming(), segments: [], requests: [], operationSettled,
    };
    this.#requests.add(command.requestId);
    this.#active = job;
    job.timer = setTimeout(() => this.#fail(job, new SpeechOutputError("TIMEOUT")), this.#timeoutMs);
    job.timer.unref();
    this.#emit({ type: "sample-accepted", requestId: command.requestId, playbackId: job.playbackId,
      generation: job.generation, selection: structuredClone(command.selection), text: command.text, textId: job.textId });
    void this.#produce(job).finally(settleOperation);
  }

  async #produce(job: Trial): Promise<void> {
    const { command } = job;
    try {
      await this.#options.policy.dispatch({
        sessionId: this.#options.sessionId, direction: "output", config: command.selection,
        catalogueRevision: command.catalogueRevision, selectionRevision: command.selectionRevision,
        source: { kind: "typed-text", sessionId: this.#options.sessionId },
      }, async (authorization) => {
        try {
          const recheck = () => { this.#assertCurrent(job); authorization.recheck(); };
          recheck();
          job.timing.queueMs = measured(this.#now() - job.admittedAt, "Admission to TTS session opening; no inference queue.");
          this.#phase(job, "opening");
          const output = await this.#options.createOutput(structuredClone(command.selection), {
            signal: job.controller.signal, recheck,
            apiKey: authorization.apiKey ? () => { recheck(); return authorization.apiKey!(); } : undefined,
            observe: (observation) => {
              if (this.#current(job) && !job.produced) job.requests.push(providerObservation(observation));
            },
          });
          recheck();
          job.session = await output.open();
          recheck();
          this.#phase(job, "producing");
          job.iterator = job.session.speak(command.text)[Symbol.asyncIterator]();
          while (true) {
            recheck();
            const result = await job.iterator.next();
            recheck();
            if (result.done) break;
            this.#audio(job, result.value, recheck);
          }
          await this.#closeTts(job);
          recheck();
          job.produced = true;
          job.timing.rowWallMs = measured(this.#now() - job.admittedAt, "Admission to completed TTS production; excludes browser drain.");
          if (job.sampleRate !== undefined) {
            this.#emit({ type: "playback", requestId: command.requestId, action: "end", playbackId: job.playbackId });
          }
          if (!this.#current(job)) return;
          this.#emit(this.#completion(job, "produced"));
          if (job.sampleRate === undefined) this.#retire(job);
        } catch (error) {
          // Keep service validation categories outside policy's adapter error
          // sanitizer. Every public failure is still mapped by #fail.
          this.#fail(job, error);
        }
      });
    } catch (error) {
      this.#fail(job, error);
    } finally {
      await this.#closeTts(job);
    }
  }

  #audio(job: Trial, segment: AudioSegment, recheck: () => void): void {
    if (!(segment.samples instanceof Float32Array) || !Number.isSafeInteger(segment.sampleRate) ||
        segment.sampleRate <= 0 || segment.sampleRate > 384_000 ||
        !Number.isSafeInteger(segment.index) || segment.index < 0 || typeof segment.sentence !== "string" ||
        !segment.samples.every(Number.isFinite) ||
        (job.sampleRate !== undefined && job.sampleRate !== segment.sampleRate)) {
      throw new SpeechOutputError("AUDIO_INVALID");
    }
    if (segment.samples.length === 0) return;
    const timing = segmentTiming(segment);
    // OpenAI exposes its observations on each actual shared pipeline segment.
    // Cartesia supplies the same request scopes through context.observe.
    if ("provider" in segment && segment.provider === "openai" &&
        "model" in segment && typeof segment.model === "string" &&
        "voice" in segment && typeof segment.voice === "string" &&
        "requestedText" in segment && typeof segment.requestedText === "string") {
      const observed = segment as AudioSegment & Record<string, unknown>;
      job.requests.push(providerObservation({ model: segment.model, voice: segment.voice, text: segment.requestedText,
        returnedTranscript: typeof observed.returnedTranscript === "string" ? observed.returnedTranscript : null,
        timing, ...(observed.sessionSetupWallMs === undefined ? {} : {
          sessionSetupWallMs: parseTimingObservation(observed.sessionSetupWallMs),
        }), ...(observed.providerFirstAudioMs === undefined ? {} : {
          providerFirstAudioMs: parseTimingObservation(observed.providerFirstAudioMs),
        }),
      }));
    }
    recheck();
    if (job.timing.serverFirstAudioReadyMs.status === "unavailable" && segment.samples.some((sample) => sample !== 0)) {
      job.timing.serverFirstAudioReadyMs = measured(this.#now() - job.admittedAt,
        "Admission to first nonempty, nonzero PCM available for host delivery; includes TTS open.");
    }
    job.segments.push({ index: segment.index, sentence: segment.sentence, sampleRate: segment.sampleRate,
      audioDurationMs: segment.samples.length / segment.sampleRate * 1_000, timing });
    if (job.sampleRate === undefined) {
      job.sampleRate = segment.sampleRate;
      this.#emit({ type: "playback", requestId: job.command.requestId, action: "start",
        playbackId: job.playbackId, sampleRate: segment.sampleRate });
    }
    recheck();
    this.#options.sendAudio(encodeAudioPacket(job.playbackId, segment));
  }

  #drained(playbackId: number): void {
    const job = this.#active;
    if (!job || job.playbackId !== playbackId || !job.produced || job.sampleRate === undefined) return;
    this.#retire(job);
    this.#emit(this.#completion(job, "drained"));
  }

  #cancel(job: Trial): void {
    this.#retire(job);
    this.#clear(job);
    this.#emit(this.#completion(job, "cancelled"));
  }

  #fail(job: Trial, error: unknown): void {
    if (!this.#current(job)) return;
    this.#retire(job);
    this.#clear(job);
    this.#emit({ type: "sample-error", requestId: job.command.requestId, playbackId: job.playbackId,
      ...safeSpeechOutputError(error) });
  }

  #retire(job: Trial): void {
    if (this.#active !== job) return;
    this.#active = undefined;
    ++this.#generation;
    job.controller.abort();
    clearTimeout(job.timer);
    // Native work can finish later. The factory serializes it; never await a
    // blocked iterator return to invalidate this connection's generation.
    void Promise.resolve().then(() => job.iterator?.return?.()).catch(() => {});
    void this.#closeTts(job);
    const retirement = (async () => {
      await job.operationSettled;
      await this.#closeTts(job);
      await this.#options.quiesce(structuredClone(job.command.selection));
      job.release();
    })();
    this.#retirements.add(retirement);
    // Failed quiescence keeps the reservation and remains observable by close.
    void retirement.then(() => this.#retirements.delete(retirement), () => {});
  }

  #closeTts(job: Trial): Promise<void> {
    if (!job.session) return Promise.resolve();
    job.closing ??= Promise.resolve().then(() => job.session!.close()).catch(() => {});
    return job.closing;
  }

  #current(job: Trial): boolean {
    return !this.#closed && this.#active === job && this.#generation === job.generation && !job.controller.signal.aborted;
  }

  #assertCurrent(job: Trial): void {
    if (!this.#current(job) || job.produced) throw new SpeechOutputError("SESSION_CLOSED");
  }

  #phase(job: Trial, phase: "opening" | "producing"): void {
    this.#emit({ type: "sample-phase", requestId: job.command.requestId, playbackId: job.playbackId, phase });
  }

  #clear(job: Trial): void {
    if (job.sampleRate !== undefined) this.#emit({ type: "playback", requestId: job.command.requestId,
      action: "clear", playbackId: job.playbackId });
  }

  #completion(job: Trial, outcome: "produced" | "drained" | "cancelled"): SpeechOutputEvent {
    return { type: "sample-complete", requestId: job.command.requestId, playbackId: job.playbackId,
      outcome, selection: structuredClone(job.command.selection), textId: job.textId,
      timing: structuredClone(job.timing), segments: structuredClone(job.segments), requests: structuredClone(job.requests) };
  }

  #emit(event: SpeechOutputEvent): void {
    if (this.#closed) return;
    try { this.#options.sendEvent(event); } catch { void this.close().catch(() => {}); }
  }
}

function measured(ms: number, boundary: string): TimingObservation {
  return parseTimingObservation({ status: "measured", ms, clock: "server", boundary });
}

function unavailableTiming(): SpeechTiming {
  const absent = (reason: string): TimingObservation => ({ status: "unavailable", reason });
  return {
    clientWallMs: absent("Observed only in the browser."), rowWallMs: absent("Production has not completed."),
    queueMs: absent("Work has not started."), loadMs: absent("No separate adapter load observation."),
    firstUseMs: absent("No explicit warmup observation."), localProcessingMs: absent("No adapter processing observation."),
    providerRequestWallMs: absent("Provider request scope is retained per segment when supplied."),
    providerProcessingMs: absent("No aggregate documented provider processing metric."),
    providerNetworkMs: absent("No independently measured network interval."),
    serverFirstAudioReadyMs: absent("No nonzero PCM delivered."), browserFirstAudioMs: absent("Observed only by the browser renderer."),
  };
}

function segmentTiming(segment: AudioSegment): SpeechTiming {
  const timing = unavailableTiming();
  // Shared adapters may supply tagged observations. Legacy synthMs/readyAtMs
  // have narrower compatibility scopes and cannot become provider compute.
  const observations = "timing" in segment && segment.timing && typeof segment.timing === "object"
    ? segment.timing as Record<string, unknown> : segment as AudioSegment & Record<string, unknown>;
  for (const key of ["localProcessingMs", "providerRequestWallMs", "providerProcessingMs", "providerNetworkMs"] as const) {
    if (observations[key] !== undefined) timing[key] = parseTimingObservation(observations[key]);
  }
  return timing;
}

function providerObservation(observation: SampleProviderObservation): SampleProviderObservation {
  return {
    model: observation.model, voice: observation.voice, text: observation.text,
    ...(observation.returnedTranscript === undefined ? {} : { returnedTranscript: observation.returnedTranscript }),
    timing: {
      providerRequestWallMs: parseTimingObservation(observation.timing.providerRequestWallMs),
      providerProcessingMs: parseTimingObservation(observation.timing.providerProcessingMs),
      providerNetworkMs: parseTimingObservation(observation.timing.providerNetworkMs),
    },
    ...(observation.sessionSetupWallMs === undefined ? {} : { sessionSetupWallMs: parseTimingObservation(observation.sessionSetupWallMs) }),
    ...(observation.providerFirstAudioMs === undefined ? {} : { providerFirstAudioMs: parseTimingObservation(observation.providerFirstAudioMs) }),
  };
}
