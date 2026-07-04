import type { EventEmitter } from "node:events";

/**
 * speak-easy contract: the engine-agnostic boundary.
 *
 * Nothing host-specific or engine-specific leaks through this interface.
 * Behind it, the engine may run as in-renderer WASM or as an IPC bridge to a
 * native sidecar in the Electron main process. Consumers (transport-matters
 * director input, littleorgans conversational loop) only ever see this.
 *
 * The core consumes audio FRAMES; it does not own the microphone. Capture
 * (getUserMedia -> AudioWorklet -> Float32 PCM) is a thin host helper shipped
 * separately, which keeps the core testable from a recorded WAV.
 */

export type STTConfig = {
  /** PCM sample rate of the frames pushed to the session. Default 16000. */
  sampleRate?: number;
  /** BCP-47 language hint, e.g. "en". Engine may ignore. */
  language?: string;
  endpoint?: EndpointConfig;
};

export type EndpointMode = "eager" | "turn-aware" | "manual";
export type EndpointConfig = {
  mode?: EndpointMode;
  minTrailingSilenceMs?: number;
  minUtteranceMs?: number;
};
export type PartialEvent = { text: string };
export type FinalEvent = { text: string };
export type EndpointEvent = Record<string, never>;
export type STTErrorEvent = { err: unknown };

/**
 * A single transcription session: push 16kHz mono Float32 PCM frames, call
 * end() at end-of-input, listen for results.
 *
 * Events:
 *   "partial"  (PartialEvent)   incremental hypothesis, may be revised
 *   "final"    (FinalEvent)     committed segment
 *   "endpoint" (EndpointEvent)  advisory end-of-speech detected
 *   "error"    (STTErrorEvent)
 *
 * The headline metric is the endpoint -> final gap: target < 200ms.
 */
export interface STTSession extends EventEmitter {
  pushAudio(frame: Float32Array): void;
  flush(): void;
  reset(): void;
  end(): Promise<void>;
}

export interface VoiceToText {
  open(config?: STTConfig): Promise<STTSession>;
}

/**
 * Post-decode rewrite configuration for the `withRewrite` decorator, which
 * wraps any VoiceToText and rewrites committed final text. Engine-agnostic:
 * a decorator concern, never consulted by an engine itself.
 */
export type RewriteRule = {
  from: string;
  to: string;
  /** True when `from` is a common word that would over-trigger elsewhere. */
  overTrigger?: boolean;
};

/** Number rendering: to digits ("ten"->"10"), to words ("10"->"ten"), or off. */
export type NumbersMode = "digits" | "words" | "off";

export type RewriteConfig = {
  rules: RewriteRule[];
  numbers: NumbersMode;
};
