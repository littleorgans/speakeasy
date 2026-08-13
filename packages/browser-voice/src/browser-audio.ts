import type { AudioSegment, AudioSink, AudioSource } from "@speakeasy/convo-engine";
import { encodeAudioPacket, type BrowserHostEvent } from "./protocol.ts";

const PLAYBACK_TIMEOUT_MS = 60_000;
const INTERRUPTION_TIMEOUT_MS = 1_000;

export class BrowserAudioSource implements AudioSource {
  #handlers:
    | { onFrame: (frame: Float32Array) => void; onError: (error: Error) => void }
    | undefined;

  start(handlers: {
    onFrame: (frame: Float32Array) => void;
    onError: (error: Error) => void;
  }): void {
    this.#handlers = handlers;
  }

  accept(frame: Float32Array): void {
    this.#handlers?.onFrame(frame);
  }

  fail(error: Error): void {
    this.#handlers?.onError(error);
  }

  stop(): void {
    this.#handlers = undefined;
  }
}

export class BrowserAudioSink implements AudioSink {
  readonly #playbackId: number;
  readonly #sampleRate: number;
  readonly #sendEvent: (event: BrowserHostEvent) => void;
  readonly #sendAudio: (packet: Uint8Array) => void;
  #resolveDrain: ((audioEndMs: number | undefined) => void) | undefined;
  #drained: Promise<number | undefined> | undefined;
  #timeout: NodeJS.Timeout | undefined;
  #isDrained = false;

  constructor(options: {
    playbackId: number;
    sampleRate: number;
    sendEvent: (event: BrowserHostEvent) => void;
    sendAudio: (packet: Uint8Array) => void;
  }) {
    this.#playbackId = options.playbackId;
    this.#sampleRate = options.sampleRate;
    this.#sendEvent = options.sendEvent;
    this.#sendAudio = options.sendAudio;
  }

  get playbackId(): number {
    return this.#playbackId;
  }

  open(): void {
    this.#sendEvent({
      type: "playback",
      action: "start",
      playbackId: this.#playbackId,
      sampleRate: this.#sampleRate,
    });
  }

  write(segment: AudioSegment): void {
    if (segment.sampleRate !== this.#sampleRate) {
      throw new Error(
        `playback sample rate changed from ${this.#sampleRate} to ${segment.sampleRate}`,
      );
    }
    this.#sendAudio(encodeAudioPacket(this.#playbackId, segment));
  }

  interrupt(): Promise<number | undefined> {
    const drained = this.#waitForDrain(INTERRUPTION_TIMEOUT_MS, 0);
    this.#sendEvent({
      type: "playback",
      action: "clear",
      playbackId: this.#playbackId,
    });
    return drained;
  }

  async end(): Promise<void> {
    if (this.#isDrained) {
      return;
    }
    if (!this.#drained) {
      this.#waitForDrain(PLAYBACK_TIMEOUT_MS, undefined);
      this.#sendEvent({
        type: "playback",
        action: "end",
        playbackId: this.#playbackId,
      });
    }
    await this.#drained;
  }

  drained(audioEndMs?: number): void {
    if (this.#isDrained) {
      return;
    }
    this.#isDrained = true;
    if (this.#timeout) {
      clearTimeout(this.#timeout);
      this.#timeout = undefined;
    }
    this.#resolveDrain?.(audioEndMs);
    this.#resolveDrain = undefined;
  }

  #waitForDrain(
    timeoutMs: number,
    timeoutPosition: number | undefined,
  ): Promise<number | undefined> {
    if (!this.#drained) {
      this.#drained = new Promise((resolve) => {
        this.#resolveDrain = resolve;
        this.#timeout = setTimeout(
          () => this.drained(timeoutPosition),
          timeoutMs,
        );
      });
    }
    return this.#drained;
  }
}
