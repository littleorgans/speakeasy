import type {
  AudioSegment,
  STTConfig,
  STTSession,
  VoiceToText,
} from "@speakeasy/speech-io";
import type { ResponderSession, VoiceResponder } from "./responder/contract.ts";
import { ChatHistory } from "./history.ts";
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
  interrupt(): void;
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
  log?: (line: string) => void;
  onState?: (state: ConvoState) => void;
  onPartial?: (text: string) => void;
  onInterrupt?: () => void;
};

export class ConversationLoop {
  readonly #deps: ConvoDeps;
  readonly #now: () => number;
  readonly #log: (line: string) => void;
  readonly #onState: (state: ConvoState) => void;
  readonly #onPartial: (text: string) => void;
  readonly #onInterrupt: () => void;
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
    this.#log = options.log ?? ((line) => console.log(line));
    this.#onState = options.onState ?? (() => {});
    this.#onPartial = options.onPartial ?? (() => {});
    this.#onInterrupt = options.onInterrupt ?? (() => {});
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
      onError: (error) => this.#log(`mic error: ${redact(error)}`),
    });
  }

  async stop(): Promise<void> {
    if (this.#stopping) {
      return;
    }
    this.#stopping = true;
    this.#setState("idle");
    await Promise.resolve(this.#deps.mic.stop()).catch(() => {});
    await this.#queue.catch(() => {});
    await this.#responderSession?.close().catch(() => {});
    await this.#session?.end().catch(() => {});
    this.#resolveDone?.();
  }

  #wireSession(session: STTSession): void {
    session.on("partial", (event: { text: string }) => {
      if (this.#state === "listening") {
        this.#onPartial(event.text);
      }
    });
    session.on("endpoint", () => {
      this.#endpointAt = this.#now();
    });
    session.on("final", (event: { text: string }) => {
      const finalAt = this.#now();
      const transcript = event.text.trim();
      if (transcript) {
        this.#enqueueTurn(transcript, this.#endpointAt || finalAt, finalAt);
      }
    });
    session.on("error", (event: { err: unknown }) => {
      this.#log(`stt error: ${redact(event.err)}`);
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
    this.#activeSink?.interrupt();
    this.#vad.reset();
    this.#setState("listening");
    this.#log("interrupted");
    this.#onInterrupt();
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
        this.#log(`turn ${turn} | interrupted (returning to listening)`);
      } else {
        if (reply.trim()) {
          this.#history.addAssistant(reply.trim());
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
      this.#log(`turn ${turn} | error: ${redact(error)} (returning to listening)`);
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
      this.#log(`turn ${turn} | no reply produced (returning to listening)`);
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
    this.#log(formatTurnLine(metrics));
  }

  #setState(next: ConvoState): void {
    if (this.#state === next) {
      return;
    }
    assertTransition(this.#state, next);
    this.#state = next;
    this.#onState(next);
  }
}

/** A log-safe message; adapters already redact keys, this guards the rest. */
function redact(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
