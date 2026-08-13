import type {
  AudioSegment,
  STTConfig,
  STTSession,
  VoiceToText,
} from "@speakeasy/speech-io";
import type { ResponderSession, VoiceResponder } from "./responder/contract.ts";
import { ChatHistory } from "./history.ts";
import type { ConversationEvent, ConversationObserver } from "./events.ts";
import {
  buildTurnMetrics,
  formatTurnLine,
  type TurnMetrics,
} from "./metrics.ts";
import { assertTransition, type ConvoState } from "./state.ts";
import { EnergyVad } from "./vad.ts";

/**
 * Half-duplex speech-to-speech loop over the contracts only. It owns no engine
 * specifics: STT, the VoiceResponder (cascade or fused), the microphone, and
 * the audio sink are all injected, so the whole cycle is exercised in tests
 * with fakes and driven with real engines by the demo.
 *
 * listening: mic frames -> STT (eager endpointing closes the turn).
 * thinking:  final transcript -> ResponderSession.respond, which streams the
 *            reply as interleaved token and audio events (a cascade starts
 *            audio on the first sentence; a fused model starts immediately).
 * speaking:  audio segments -> continuous sink; mic gated (frames discarded).
 * Any stage failure logs one redacted line and returns to listening; the loop
 * never dies mid-conversation.
 */

/** Microphone seam: pushes frames until stopped. */
export interface AudioSource {
  start(handlers: {
    onFrame: (frame: Float32Array) => void;
    onError: (error: Error) => void;
  }): void | Promise<void>;
  stop(): void | Promise<void>;
}

/** Continuous playback sink (createSegmentPlayer satisfies this). */
export interface AudioSink {
  open(): void;
  write(segment: AudioSegment): void;
  /** Stop playback immediately, dropping buffered audio (barge-in). */
  /** Stop now and report how many milliseconds reached the audio device. */
  interrupt(): Promise<number | undefined>;
  end(): Promise<void>;
}

export type ConvoDeps = {
  stt: VoiceToText;
  /** The spoken-reply engine: CascadeResponder (LLM + TTS) or a fused model. */
  responder: VoiceResponder;
  mic: AudioSource;
  /** Built per turn once the first segment's sample rate is known. */
  createSink: (sampleRate: number) => AudioSink;
};

export type ConvoOptions = {
  systemPrompt?: string;
  historyLimit?: number;
  sttConfig?: STTConfig;
  /** Auto-stop after this many completed turns; unset runs until stop(). */
  maxTurns?: number;
  /** Enable voice barge-in: user speech during playback cuts the assistant off. */
  barge?: boolean;
  /** Peak-amplitude threshold for the barge-in VAD (0..1). */
  bargeThreshold?: number;
  now?: () => number;
  /** One host-neutral stream for state, transcripts, metrics, and failures. */
  onEvent?: ConversationObserver;
};

export class ConversationLoop {
  readonly #deps: ConvoDeps;
  readonly #now: () => number;
  readonly #onEvent: ConversationObserver;
  readonly #sttConfig: STTConfig;
  readonly #maxTurns: number | undefined;
  readonly #barge: boolean;
  readonly #vad: EnergyVad;
  readonly #history: ChatHistory;
  readonly #metrics: TurnMetrics[] = [];

  #state: ConvoState = "idle";
  #session: STTSession | undefined;
  #responderSession: ResponderSession | undefined;
  #endpointAt = 0;
  #turnsStarted = 0;
  #queue: Promise<void> = Promise.resolve();
  #stopping = false;
  #interrupted = false;
  #activeSink: AudioSink | undefined;
  #resolveDone: (() => void) | undefined;
  readonly done: Promise<void>;

  constructor(deps: ConvoDeps, options: ConvoOptions = {}) {
    this.#deps = deps;
    this.#now = options.now ?? (() => performance.now());
    this.#onEvent = options.onEvent ?? (() => {});
    this.#sttConfig = options.sttConfig ?? { endpoint: { mode: "eager" } };
    this.#maxTurns = options.maxTurns;
    this.#barge = options.barge ?? false;
    this.#vad = new EnergyVad({ threshold: options.bargeThreshold });
    this.#history = new ChatHistory(options.systemPrompt, options.historyLimit);
    this.done = new Promise<void>((resolve) => {
      this.#resolveDone = resolve;
    });
  }

  get state(): ConvoState {
    return this.#state;
  }

  get metrics(): readonly TurnMetrics[] {
    return this.#metrics;
  }

  async start(): Promise<void> {
    this.#session = await this.#deps.stt.open(this.#sttConfig);
    this.#responderSession = await this.#deps.responder.open();
    this.#wireSession(this.#session);
    this.#setState("listening");
    await this.#deps.mic.start({
      onFrame: (frame) => this.#onFrame(frame),
      onError: (error) => this.#notice("error", `mic error: ${redact(error)}`),
    });
  }

  async stop(): Promise<void> {
    if (this.#stopping) {
      return;
    }
    if (this.#state === "speaking" || this.#state === "thinking") {
      this.interrupt();
    }
    this.#stopping = true;
    this.#setState("idle");
    await Promise.resolve(this.#deps.mic.stop()).catch(() => {});
    await this.#queue.catch(() => {});
    await this.#responderSession?.close().catch(() => {});
    await this.#session?.end().catch(() => {});
    this.#resolveDone?.();
  }

  /** Commit the recognizer's current utterance for an explicit push-to-talk release. */
  commitInput(): void {
    if (this.#stopping || this.#state !== "listening") {
      return;
    }
    this.#endpointAt = this.#now();
    this.#session?.flush();
  }

  #wireSession(session: STTSession): void {
    session.on("partial", (event: { text: string }) => {
      if (this.#state === "listening") {
        this.#emit({
          type: "transcript",
          role: "user",
          text: event.text,
          final: false,
          mode: "replace",
        });
      }
    });
    session.on("endpoint", () => {
      this.#endpointAt = this.#now();
    });
    session.on("final", (event: { text: string }) => {
      const finalAt = this.#now();
      const transcript = event.text.trim();
      if (transcript) {
        this.#emit({
          type: "transcript",
          role: "user",
          text: transcript,
          final: true,
          mode: "replace",
        });
        this.#enqueueTurn(transcript, this.#endpointAt || finalAt, finalAt);
      }
    });
    session.on("error", (event: { err: unknown }) => {
      this.#notice("error", `stt error: ${redact(event.err)}`);
    });
  }

  /**
   * Mic routing. While listening, frames feed the recognizer. While the
   * assistant is thinking or speaking and barge-in is enabled, frames feed the
   * VAD instead; sustained speech interrupts the turn. (Barge-in wants
   * headphones — on open speakers the mic hears the assistant.)
   */
  #onFrame(frame: Float32Array): void {
    if (this.#state === "listening") {
      this.#session?.pushAudio(frame);
      return;
    }
    if (
      this.#barge &&
      (this.#state === "speaking" || this.#state === "thinking") &&
      this.#vad.accept(frame)
    ) {
      this.interrupt();
    }
  }

  /**
   * Cut the assistant off mid-turn and return to listening: stop playback now,
   * flag the in-flight turn to unwind, and reopen the mic. Fired by the barge-in
   * VAD or a host key press. No-op unless a turn is active.
   */
  interrupt(): void {
    if (this.#interrupted || (this.#state !== "speaking" && this.#state !== "thinking")) {
      return;
    }
    this.#interrupted = true;
    const playedAudioMs = this.#activeSink?.interrupt() ?? Promise.resolve(undefined);
    this.#responderSession?.interrupt(playedAudioMs.catch(() => 0));
    this.#vad.reset();
    this.#setState("listening");
    this.#emit({ type: "interrupted" });
  }

  #enqueueTurn(transcript: string, endpointAt: number, finalAt: number): void {
    if (this.#stopping) {
      return;
    }
    this.#queue = this.#queue.then(() =>
      this.#runTurn(transcript, endpointAt, finalAt),
    );
  }

  async #runTurn(
    transcript: string,
    endpointAt: number,
    finalAt: number,
  ): Promise<void> {
    if (this.#stopping || !this.#responderSession) {
      return;
    }
    const responderSession = this.#responderSession;
    const turn = (this.#turnsStarted += 1);
    this.#interrupted = false;
    this.#vad.reset();
    this.#setState("thinking");
    this.#history.addUser(transcript);

    let firstTokenAt: number | undefined;
    let firstAudioAt: number | undefined;
    let tokenCount = 0;
    let reply = "";
    let spokenMs = 0;
    let sink: AudioSink | undefined;

    try {
      for await (const event of responderSession.respond(this.#history.messages())) {
        if (this.#interrupted) {
          break; // ends the iteration; the responder cancels in its finally
        }
        if (event.type === "token") {
          firstTokenAt ??= event.at;
          tokenCount += 1;
          reply += event.text;
          this.#emit({
            type: "transcript",
            role: "assistant",
            text: event.text,
            final: false,
            mode: "append",
          });
          continue;
        }
        const segment = event.segment;
        if (firstAudioAt === undefined) {
          firstAudioAt = this.#now();
          this.#setState("speaking");
          sink = this.#deps.createSink(segment.sampleRate);
          this.#activeSink = sink;
          sink.open();
        }
        sink!.write(segment);
        spokenMs += segment.audioDurationMs;
      }
      await sink?.end();
      if (this.#interrupted) {
        this.#notice("info", `turn ${turn} | interrupted (returning to listening)`);
      } else {
        if (reply.trim()) {
          const finalReply = reply.trim();
          this.#history.addAssistant(finalReply);
          this.#emit({
            type: "transcript",
            role: "assistant",
            text: finalReply,
            final: true,
            mode: "replace",
          });
        }
        this.#recordTurn(turn, transcript, {
          endpointAt,
          finalAt,
          firstTokenAt,
          firstAudioAt,
          tokenCount,
          spokenMs,
        });
      }
    } catch (error) {
      await sink?.end().catch(() => {});
      this.#notice(
        "error",
        `turn ${turn} | error: ${redact(error)} (returning to listening)`,
      );
    } finally {
      this.#activeSink = undefined;
      this.#setState("listening");
    }

    if (this.#maxTurns !== undefined && this.#turnsStarted >= this.#maxTurns) {
      void this.stop();
    }
  }

  #recordTurn(
    turn: number,
    transcript: string,
    outcome: {
      endpointAt: number;
      finalAt: number;
      firstTokenAt: number | undefined;
      firstAudioAt: number | undefined;
      tokenCount: number;
      spokenMs: number;
    },
  ): void {
    if (outcome.firstAudioAt === undefined || outcome.firstTokenAt === undefined) {
      this.#notice("info", `turn ${turn} | no reply produced (returning to listening)`);
      return;
    }
    const metrics = buildTurnMetrics(
      turn,
      transcript,
      {
        endpointAt: outcome.endpointAt,
        finalAt: outcome.finalAt,
        firstTokenAt: outcome.firstTokenAt,
        firstAudioAt: outcome.firstAudioAt,
      },
      outcome.tokenCount,
      outcome.spokenMs,
    );
    this.#metrics.push(metrics);
    this.#emit({ type: "metrics", metrics });
    this.#notice("info", formatTurnLine(metrics));
  }

  #setState(next: ConvoState): void {
    if (this.#state === next) {
      return;
    }
    assertTransition(this.#state, next);
    this.#state = next;
    this.#emit({ type: "state", state: next });
  }

  #notice(level: "info" | "error", message: string): void {
    this.#emit({ type: "notice", level, message });
  }

  #emit(event: ConversationEvent): void {
    this.#onEvent(event);
  }
}

/** A log-safe message; adapters already redact keys, this guards the rest. */
function redact(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
