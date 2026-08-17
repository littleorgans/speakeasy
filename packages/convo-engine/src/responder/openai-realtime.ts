import { WebSocket } from "ws";
import type { AudioSegment } from "@speakeasy/speech-io";
import type { ChatMessage } from "@speakeasy/llm";
import type { ResponderEvent, ResponderSession, VoiceResponder } from "./contract.ts";

/**
 * Fused VoiceResponder over the OpenAI Realtime API (GA WebSocket surface,
 * verified against developers.openai.com 2026-07-19).
 *
 * Text in, audio out: the local STT transcript is sent as an `input_text`
 * conversation item, the model generates and speaks the reply in one hop, and
 * audio arrives as base64 PCM16 @ 24kHz in `response.output_audio.delta`
 * events (reply text rides along in `response.output_audio_transcript.delta`).
 *
 * Conversation state lives server-side within the socket's session, so
 * respond() only forwards the newest user message; the system prompt is
 * installed once as session `instructions`. The loop's history window still
 * accumulates locally and is what any cascade fallback would see.
 *
 * The API key is read at open() time only, never stored or logged. The socket
 * factory is injectable so the adapter is unit-testable without a network.
 */

export const REALTIME_URL = "wss://api.openai.com/v1/realtime";
export const DEFAULT_REALTIME_MODEL = "gpt-realtime-2.1-mini";
export const DEFAULT_REALTIME_VOICE = "marin";
export const REALTIME_VOICES = [
  "alloy",
  "ash",
  "ballad",
  "coral",
  "echo",
  "sage",
  "shimmer",
  "verse",
  "marin",
  "cedar",
] as const;
export type RealtimeVoice = (typeof REALTIME_VOICES)[number];
const OUTPUT_SAMPLE_RATE = 24_000;
const TRUNCATE_ACK_TIMEOUT_MS = 5_000;

export function parseRealtimeVoice(value: string): RealtimeVoice {
  if (REALTIME_VOICES.includes(value as RealtimeVoice)) {
    return value as RealtimeVoice;
  }
  throw new Error(`Unsupported Realtime voice "${value}"`);
}

/** The subset of the ws client the adapter drives; fakes implement this. */
export type SocketLike = {
  on(event: "open" | "close", listener: () => void): void;
  on(event: "message", listener: (data: { toString(): string }) => void): void;
  on(event: "error", listener: (error: Error) => void): void;
  send(data: string): void;
  close(): void;
};

export type RealtimeOptions = {
  model?: string;
  /** Realtime voice name, e.g. "marin", "cedar". */
  voice?: RealtimeVoice;
  /** Key source, read on open(); defaults to the environment. */
  apiKey?: () => string | undefined;
  /** Injectable socket factory; defaults to `ws` with a bearer header. */
  createSocket?: (url: string, apiKey: string) => SocketLike;
  now?: () => number;
};

export class OpenAIRealtimeResponder implements VoiceResponder {
  readonly #model: string;
  readonly #voice: string;
  readonly #apiKey: () => string | undefined;
  readonly #createSocket: (url: string, apiKey: string) => SocketLike;
  readonly #now: () => number;

  constructor(options: RealtimeOptions = {}) {
    this.#model = options.model ?? DEFAULT_REALTIME_MODEL;
    this.#voice = parseRealtimeVoice(options.voice ?? DEFAULT_REALTIME_VOICE);
    this.#apiKey = options.apiKey ?? (() => process.env.OPENAI_API_KEY);
    this.#createSocket =
      options.createSocket ??
      ((url, apiKey) =>
        new WebSocket(url, {
          headers: { authorization: `Bearer ${apiKey}` },
        }) as SocketLike);
    this.#now = options.now ?? (() => performance.now());
  }

  async open(): Promise<ResponderSession> {
    const key = this.#apiKey();
    if (!key) {
      throw new Error("OPENAI_API_KEY is not set");
    }
    const socket = this.#createSocket(
      `${REALTIME_URL}?model=${encodeURIComponent(this.#model)}`,
      key,
    );
    const session = new RealtimeSession(socket, this.#voice, this.#now);
    await session.ready();
    return session;
  }
}

/** A server event, parsed leniently: only the routed fields are typed. */
type ServerEvent = {
  type: string;
  event_id?: string;
  response?: { id?: string };
  response_id?: string;
  item_id?: string;
  content_index?: number;
  audio_end_ms?: number;
  item?: { id?: string; type?: string };
  delta?: string;
  error?: { type?: string; code?: string; message?: string; event_id?: string };
};

type ActiveTurn = {
  responseId?: string;
  itemId?: string;
  cancelled: boolean;
  responseDone: boolean;
  cancelEventId?: string;
  deliveredAudioSamples: number;
  audioEndMs?: number;
  truncateSent: boolean;
  truncateEventId?: string;
  truncateAudioEndMs?: number;
  truncateAckTimer?: ReturnType<typeof setTimeout>;
  resolveInterruption?: () => void;
};

class RealtimeSession implements ResponderSession {
  readonly #socket: SocketLike;
  readonly #voice: string;
  readonly #now: () => number;
  readonly #queue = new EventQueue<ServerEvent>();
  readonly #cancelledTurns = new Map<string, ActiveTurn>();
  #openPromise: Promise<void>;
  #instructions: string | undefined;
  #closed = false;
  #eventSequence = 0;
  #fatalError: Error | undefined;
  #activeTurn: ActiveTurn | undefined;
  #interruptionDone: Promise<void> = Promise.resolve();

  constructor(socket: SocketLike, voice: string, now: () => number) {
    this.#socket = socket;
    this.#voice = voice;
    this.#now = now;
    this.#openPromise = new Promise((resolve, reject) => {
      socket.on("open", () => resolve());
      socket.on("error", (error) => {
        reject(error);
        this.#queue.fail(error);
      });
    });
    socket.on("message", (data) => {
      let event: ServerEvent;
      try {
        event = JSON.parse(data.toString()) as ServerEvent;
      } catch {
        return; // ignore unparseable frames
      }
      if (event.type === "error") {
        const turn = this.#activeTurn;
        const cancelEventId = event.error?.event_id;
        const cancelledTurn =
          (cancelEventId ? this.#cancelledTurns.get(cancelEventId) : undefined) ??
          (turn?.cancelled ? turn : undefined);
        if (isNoActiveResponseCancel(event)) {
          if (cancelEventId) this.#cancelledTurns.delete(cancelEventId);
          if (cancelledTurn) {
            cancelledTurn.responseDone = true;
            this.#truncateInterruptedTurn(cancelledTurn);
            if (cancelledTurn === turn) this.#queue.push(event);
          }
          return;
        }
        const error = new Error(
          `realtime error: ${event.error?.message ?? "unknown"}`,
        );
        this.#fatalError ??= error;
        if (turn) this.#resolveInterruption(turn);
        this.#queue.fail(error);
        return;
      }
      this.#observeTurn(event);
      this.#queue.push(event);
    });
    socket.on("close", () => {
      if (!this.#closed) {
        const error = new Error("realtime socket closed unexpectedly");
        this.#fatalError ??= error;
        if (this.#activeTurn) this.#resolveInterruption(this.#activeTurn);
        this.#queue.fail(error);
      }
    });
  }

  async ready(): Promise<void> {
    await this.#openPromise;
    this.#send({
      type: "session.update",
      session: {
        type: "realtime",
        output_modalities: ["audio"],
        audio: {
          output: {
            format: { type: "audio/pcm", rate: OUTPUT_SAMPLE_RATE },
            voice: this.#voice,
          },
        },
      },
    });
  }

  async *respond(messages: ChatMessage[]): AsyncGenerator<ResponderEvent> {
    await this.#interruptionDone;
    if (this.#fatalError) throw this.#fatalError;
    const start = this.#now();
    this.#syncInstructions(messages);
    const user = messages.findLast((message) => message.role === "user");
    if (!user) {
      return;
    }
    this.#send({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: user.content }],
      },
    });
    const turn: ActiveTurn = {
      cancelled: false,
      responseDone: false,
      deliveredAudioSamples: 0,
      truncateSent: false,
    };
    this.#activeTurn = turn;
    this.#send({ type: "response.create" });

    // Route by response id so a cancelled turn's stragglers never leak into
    // the next one: events before our response.created (or tagged with a
    // different id) are skipped.
    let index = 0;
    let done = false;
    try {
      for (;;) {
        const event = await this.#queue.next();
        if (turn.cancelled && turn.responseDone) {
          done = true;
          return;
        }
        if (turn.responseId === undefined) {
          if (event.type === "response.created") {
            turn.responseId = event.response?.id;
          }
          continue;
        }
        if (event.response_id !== undefined && event.response_id !== turn.responseId) {
          continue;
        }
        switch (event.type) {
          case "response.output_audio_transcript.delta":
            if (event.delta) {
              yield { type: "token", text: event.delta, at: this.#now() };
            }
            break;
          case "response.output_audio.delta":
            if (event.delta) {
              const samples = decodePcm16(event.delta);
              turn.deliveredAudioSamples += samples.length;
              yield {
                type: "audio",
                segment: {
                  index: index++,
                  sentence: "",
                  samples,
                  sampleRate: OUTPUT_SAMPLE_RATE,
                  readyAtMs: this.#now() - start,
                  synthMs: 0,
                  audioDurationMs: (samples.length / OUTPUT_SAMPLE_RATE) * 1_000,
                },
              };
            }
            break;
          case "response.done":
            done = true;
            return;
          default:
            break; // lifecycle events carry nothing the loop needs
        }
      }
    } finally {
      if (!done && !this.#closed && !turn.cancelled) {
        this.#cancel(turn, Promise.resolve(undefined));
      }
    }
  }

  interrupt(playedAudioMs: Promise<number | undefined>): void {
    const turn = this.#activeTurn;
    if (!turn || turn.cancelled || this.#closed) {
      return;
    }
    this.#cancel(turn, playedAudioMs);
  }

  async close(): Promise<void> {
    this.#closed = true;
    if (this.#activeTurn) this.#resolveInterruption(this.#activeTurn);
    this.#socket.close();
  }

  #cancel(turn: ActiveTurn, playedAudioMs: Promise<number | undefined>): void {
    turn.cancelled = true;
    if (!turn.responseDone) {
      turn.cancelEventId = `cancel_${++this.#eventSequence}`;
      this.#cancelledTurns.set(turn.cancelEventId, turn);
      this.#send({
        event_id: turn.cancelEventId,
        type: "response.cancel",
        ...(turn.responseId ? { response_id: turn.responseId } : {}),
      });
    }
    this.#interruptionDone = new Promise((resolve) => {
      turn.resolveInterruption = resolve;
    });
    void playedAudioMs.then(
      (milliseconds) => {
        turn.audioEndMs = Math.max(0, Math.floor(milliseconds ?? 0));
        this.#truncateInterruptedTurn(turn);
      },
      () => {
        turn.audioEndMs = 0;
        this.#truncateInterruptedTurn(turn);
      },
    );
  }

  #observeTurn(event: ServerEvent): void {
    const turn = this.#activeTurn;
    if (!turn) return;
    if (event.type === "response.created" && turn.responseId === undefined) {
      turn.responseId = event.response?.id;
      return;
    }
    if (event.response_id !== undefined && event.response_id !== turn.responseId) {
      return;
    }
    if (event.type === "response.output_item.added" && event.item?.type === "message") {
      turn.itemId = event.item.id;
    }
    if (event.type === "response.output_audio.delta") {
      turn.itemId ??= event.item_id;
    }
    if (event.type === "response.done") {
      turn.responseDone = true;
      if (turn.cancelEventId) this.#cancelledTurns.delete(turn.cancelEventId);
    }
    if (
      event.type === "conversation.item.truncated" &&
      event.item_id === turn.itemId &&
      event.content_index === 0 &&
      event.audio_end_ms === turn.truncateAudioEndMs
    ) {
      this.#resolveInterruption(turn);
      return;
    }
    this.#truncateInterruptedTurn(turn);
  }

  #truncateInterruptedTurn(turn: ActiveTurn): void {
    if (this.#closed) return;
    if (!turn.cancelled || turn.audioEndMs === undefined || turn.truncateSent) {
      return;
    }
    if (turn.itemId) {
      turn.truncateSent = true;
      const deliveredAudioMs = Math.floor(
        (turn.deliveredAudioSamples / OUTPUT_SAMPLE_RATE) * 1_000,
      );
      turn.truncateAudioEndMs = Math.min(turn.audioEndMs, deliveredAudioMs);
      turn.truncateEventId = `truncate_${++this.#eventSequence}`;
      this.#send({
        event_id: turn.truncateEventId,
        type: "conversation.item.truncate",
        item_id: turn.itemId,
        content_index: 0,
        audio_end_ms: turn.truncateAudioEndMs,
      });
      turn.truncateAckTimer = setTimeout(() => {
        const error = new Error("realtime truncate acknowledgement timed out");
        this.#fatalError ??= error;
        this.#resolveInterruption(turn);
        this.#queue.fail(error);
      }, TRUNCATE_ACK_TIMEOUT_MS);
      return;
    }
    if (turn.responseDone) {
      this.#resolveInterruption(turn);
    }
  }

  #resolveInterruption(turn: ActiveTurn): void {
    if (turn.truncateAckTimer) {
      clearTimeout(turn.truncateAckTimer);
      turn.truncateAckTimer = undefined;
    }
    turn.resolveInterruption?.();
    turn.resolveInterruption = undefined;
  }

  /** Install/refresh the system prompt as session instructions when it changes. */
  #syncInstructions(messages: ChatMessage[]): void {
    const system = messages.find((message) => message.role === "system");
    if (system && system.content !== this.#instructions) {
      this.#instructions = system.content;
      this.#send({
        type: "session.update",
        session: { type: "realtime", instructions: system.content },
      });
    }
  }

  #send(payload: unknown): void {
    this.#socket.send(JSON.stringify(payload));
  }
}

function isNoActiveResponseCancel(event: ServerEvent): boolean {
  return (
    event.error?.code === "response_cancel_not_active" ||
    /^Cancellation failed: no active response found\.?$/i.test(event.error?.message ?? "")
  );
}

/** Base64 PCM16 mono -> Float32 samples (the AudioSegment/player format). */
export function decodePcm16(base64: string): Float32Array {
  const bytes = Buffer.from(base64, "base64");
  const samples = new Float32Array(Math.floor(bytes.length / 2));
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = bytes.readInt16LE(i * 2) / 32768;
  }
  return samples;
}

/** Unbounded push queue bridging socket callbacks to the respond() pull loop. */
class EventQueue<T> {
  readonly #items: T[] = [];
  #waiter: { resolve: (item: T) => void; reject: (error: Error) => void } | undefined;
  #error: Error | undefined;

  push(item: T): void {
    if (this.#error) {
      return;
    }
    if (this.#waiter) {
      const { resolve } = this.#waiter;
      this.#waiter = undefined;
      resolve(item);
      return;
    }
    this.#items.push(item);
  }

  fail(error: Error): void {
    this.#error ??= error;
    if (this.#waiter) {
      const { reject } = this.#waiter;
      this.#waiter = undefined;
      reject(error);
    }
  }

  next(): Promise<T> {
    const buffered = this.#items.shift();
    if (buffered !== undefined) {
      return Promise.resolve(buffered);
    }
    if (this.#error) {
      return Promise.reject(this.#error);
    }
    return new Promise((resolve, reject) => {
      this.#waiter = { resolve, reject };
    });
  }
}
