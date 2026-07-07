import type {
  AudioSegment,
  TextToSpeech,
  TTSConfig,
  TTSSession,
} from "./contract.ts";
import { parseTtsModelId } from "./models.ts";
import {
  fromArray,
  planSegments,
  planSegmentsStream,
  synthPipeline,
  type SegmentSynth,
} from "./stream.ts";
import { TtsSynth } from "./synth.ts";

/**
 * Adapter #1: the in-process sherpa TTS engine (kokoro/piper) behind the
 * TextToSpeech contract. It owns no pipeline logic of its own — open() loads a
 * TtsSynth, and speak() feeds the shared synthPipeline from stream.ts, choosing
 * the segmenter by input shape: a whole string is planned up front
 * (planSegments), an incremental token stream is cut on the fly
 * (planSegmentsStream). Both paths run the identical 1-deep synth pipeline.
 */

/** Builds the loaded synth for a model id; injectable so the adapter is testable. */
export type TtsSynthFactory = (modelId: string) => Promise<SegmentSynth>;

const defaultSynthFactory: TtsSynthFactory = (modelId) =>
  TtsSynth.create(parseTtsModelId(modelId));

export class SherpaTextToSpeech implements TextToSpeech {
  readonly #createSynth: TtsSynthFactory;

  constructor(createSynth: TtsSynthFactory = defaultSynthFactory) {
    this.#createSynth = createSynth;
  }

  async open(config: TTSConfig = {}): Promise<TTSSession> {
    const synth = await this.#createSynth(config.model ?? "piper-amy");
    return new SherpaTTSSession(synth, config.speed ?? 1);
  }
}

class SherpaTTSSession implements TTSSession {
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
    // The sherpa OfflineTts binding exposes no explicit free; the native handle
    // is released when the TtsSynth is garbage-collected. Nothing to do today,
    // but the contract carries close() so a sidecar adapter can shut a process.
  }
}
