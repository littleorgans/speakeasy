/**
 * speak-easy TTS contract: the engine-agnostic text-to-speech boundary.
 *
 * The mirror of src/contract.ts (the STT VoiceToText boundary). Nothing
 * engine-specific leaks through it: behind it an adapter may drive sherpa
 * kokoro/piper in-process (the default) or shell out to a Python mlx-audio
 * sidecar. Consumers (the convo-engine speaking state) only ever see this.
 *
 * `speak` consumes a stream of text and yields audio segments as each is
 * synthesized. The text stream is the LLM feed shape: token deltas arrive
 * incrementally, the adapter accumulates them, cuts speakable sentences (with an
 * aggressive short first chunk for low time-to-first-audio), and synthesizes as
 * sentences complete. A plain string is accepted as the degenerate one-shot case.
 */

export type TTSConfig = {
  /** Engine-defined model id. The adapter maps it to its own registry. */
  model?: string;
  /** Voice selector: a numeric speaker id (sherpa) or a named/UUID voice (Cartesia). */
  voice?: number | string;
  /** Speaking rate multiplier; 1 is the model's native pace. */
  speed?: number;
};

/**
 * A single synthesized chunk of speech: mono Float32 PCM plus the timing needed
 * to prove gapless playback. Fields match stream.ts's internal SpeechSegment so
 * the sherpa pipeline yields this type directly.
 */
export type AudioSegment = {
  index: number;
  /** The sentence/chunk of text this audio renders. */
  sentence: string;
  samples: Float32Array;
  sampleRate: number;
  /** Wall-clock ms from the speak() call until this segment was ready. */
  readyAtMs: number;
  /** Synth time for this segment alone. */
  synthMs: number;
  audioDurationMs: number;
};

/**
 * A speaking session bound to one loaded model. `speak` may be called for each
 * turn; it consumes incremental text (or a whole string) and streams audio.
 */
export interface TTSSession {
  speak(text: AsyncIterable<string> | string): AsyncIterable<AudioSegment>;
  close(): Promise<void>;
}

export interface TextToSpeech {
  open(config?: TTSConfig): Promise<TTSSession>;
}
