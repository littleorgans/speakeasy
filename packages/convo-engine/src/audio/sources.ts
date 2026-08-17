import { setTimeout as delay } from "node:timers/promises";
import {
  CAPTURE_FRAME_MS,
  pacedFrames,
  startMicCapture,
  type MicCapture,
  type WavAudio,
} from "@speakeasy/speech-io";
import type { ConversationObserver } from "../events.ts";
import type { AudioSource } from "../loop.ts";

const DEFAULT_SILENCE_TAIL_MS = 700;

type AudioSourceHandlers = Parameters<AudioSource["start"]>[0];

export type WavSourceOptions = {
  frameMs?: number;
  silenceTailMs?: number;
};

export type ScriptedWavSourceOptions = WavSourceOptions & {
  onUtteranceEnd?: () => void;
  utteranceGapMs?: number;
};

type ResolvedWavSourceOptions = {
  frameMs: number;
  silenceTailMs: number;
};

/** Live microphone through the shared ffmpeg capture helper. */
export class MicAudioSource implements AudioSource {
  readonly #device: string;
  #capture: MicCapture | undefined;

  constructor(device: string) {
    this.#device = device;
  }

  start(handlers: AudioSourceHandlers): void {
    this.#capture = startMicCapture({
      device: this.#device,
      onFrame: handlers.onFrame,
      onError: handlers.onError,
    });
  }

  async stop(): Promise<void> {
    await this.#capture?.stop();
  }
}

/** Replay one decoded utterance at real time, followed by endpointing silence. */
export class WavAudioSource implements AudioSource {
  readonly #wav: WavAudio;
  readonly #options: ResolvedWavSourceOptions;
  #feeding: Promise<void> | undefined;
  #stopped = false;

  constructor(wav: WavAudio, options: WavSourceOptions = {}) {
    this.#wav = wav;
    this.#options = resolveOptions(options);
  }

  start(handlers: AudioSourceHandlers): void {
    this.#stopped = false;
    this.#feeding = this.#feed(handlers).catch((error: unknown) => {
      handlers.onError(toError(error));
    });
  }

  async stop(): Promise<void> {
    this.#stopped = true;
    await this.#feeding;
  }

  async #feed(handlers: AudioSourceHandlers): Promise<void> {
    const frames = framesWithSilenceTail(this.#wav, this.#options);
    for await (const [, frame] of pacedFrames(
      frames,
      this.#options.frameMs,
    )) {
      if (this.#stopped) {
        return;
      }
      handlers.onFrame(frame);
    }
  }
}

/** Replay each utterance after the conversation returns to listening. */
export class ScriptedWavSource implements AudioSource {
  readonly #utterances: readonly WavAudio[];
  readonly #options: ResolvedWavSourceOptions;
  readonly #onUtteranceEnd: () => void;
  readonly #utteranceGapMs: number;
  readonly done: Promise<void>;
  #resolveDone: () => void = () => {};
  #handlers: AudioSourceHandlers | undefined;
  #feeding: Promise<void> | undefined;
  #nextIndex = 0;
  #listening = false;
  #readyForUtterance = false;
  #cancelFeeding = false;
  #stopped = false;
  #finished = false;

  constructor(
    utterances: readonly WavAudio[],
    options: ScriptedWavSourceOptions = {},
  ) {
    this.#utterances = utterances;
    this.#options = resolveOptions(options);
    this.#onUtteranceEnd = options.onUtteranceEnd ?? (() => {});
    this.#utteranceGapMs = options.utteranceGapMs ?? 0;
    if (!Number.isFinite(this.#utteranceGapMs) || this.#utteranceGapMs < 0) {
      throw new Error(
        `utteranceGapMs must be non-negative, received ${this.#utteranceGapMs}`,
      );
    }
    this.done = new Promise((resolve) => {
      this.#resolveDone = resolve;
    });
  }

  readonly onEvent: ConversationObserver = (event) => {
    if (event.type !== "state") {
      return;
    }
    this.#listening = event.state === "listening";
    if (this.#listening) {
      this.#readyForUtterance = true;
    } else {
      this.#readyForUtterance = false;
      this.#cancelFeeding = true;
    }
    this.#pump();
  };

  start(handlers: AudioSourceHandlers): void {
    this.#handlers = handlers;
    this.#stopped = false;
    this.#pump();
  }

  async stop(): Promise<void> {
    this.#stopped = true;
    await this.#feeding;
    this.#finish();
  }

  #pump(): void {
    if (this.#stopped || this.#feeding || !this.#handlers) {
      return;
    }
    const utterance = this.#utterances[this.#nextIndex];
    if (!utterance) {
      this.#finish();
      return;
    }
    if (!this.#listening || !this.#readyForUtterance) {
      return;
    }

    const waitBeforeFeed = this.#nextIndex > 0;
    this.#cancelFeeding = false;
    this.#nextIndex += 1;
    const handlers = this.#handlers;
    const feeding = this.#feed(
      utterance,
      handlers,
      waitBeforeFeed,
    ).catch((error: unknown) => {
      handlers.onError(toError(error));
    });
    this.#feeding = feeding;
    void feeding.then(() => {
      if (this.#feeding === feeding) {
        this.#feeding = undefined;
      }
      this.#pump();
    });
  }

  async #feed(
    utterance: WavAudio,
    handlers: AudioSourceHandlers,
    waitBeforeFeed: boolean,
  ): Promise<void> {
    if (waitBeforeFeed && this.#utteranceGapMs > 0) {
      await delay(this.#utteranceGapMs);
    }
    const frames = framesWithSilenceTail(utterance, this.#options);
    for await (const [, frame] of pacedFrames(
      frames,
      this.#options.frameMs,
    )) {
      if (this.#stopped || !this.#listening || this.#cancelFeeding) {
        return;
      }
      handlers.onFrame(frame);
    }
    this.#onUtteranceEnd();
  }

  #finish(): void {
    if (!this.#finished) {
      this.#finished = true;
      this.#resolveDone();
    }
  }
}

function resolveOptions(options: WavSourceOptions): ResolvedWavSourceOptions {
  const frameMs = options.frameMs ?? CAPTURE_FRAME_MS;
  const silenceTailMs = options.silenceTailMs ?? DEFAULT_SILENCE_TAIL_MS;
  if (!Number.isFinite(silenceTailMs) || silenceTailMs < 0) {
    throw new Error(
      `silenceTailMs must be non-negative, received ${silenceTailMs}`,
    );
  }
  return { frameMs, silenceTailMs };
}

function* framesWithSilenceTail(
  wav: WavAudio,
  options: ResolvedWavSourceOptions,
): Generator<Float32Array> {
  yield* wav.frames;
  const frameSamples = Math.round((wav.sampleRate * options.frameMs) / 1_000);
  const silenceFrames = Math.round(options.silenceTailMs / options.frameMs);
  for (let index = 0; index < silenceFrames; index += 1) {
    yield new Float32Array(frameSamples);
  }
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
