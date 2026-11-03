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
import { TtsSynth, validateSpeaker, validateSpeed, type SynthExecution } from "./synth.ts";

/**
 * Adapter #1: the in-process sherpa TTS engine (kokoro/piper) behind the
 * TextToSpeech contract. It owns no pipeline logic of its own. open() loads a
 * TtsSynth, and speak() feeds the shared synthPipeline from stream.ts, choosing
 * the segmenter by input shape: a whole string is planned up front
 * (planSegments), an incremental token stream is cut on the fly
 * (planSegmentsStream). Both paths run the identical 1-deep synth pipeline.
 */

/** Builds the loaded synth for a model id; injectable so the adapter is testable. */
export type LoadedTtsSynth = SegmentSynth & { numSpeakers: number; sampleRate: number };
export type TtsSynthFactory = (modelId: string) => Promise<LoadedTtsSynth>;

const defaultSynthFactory: TtsSynthFactory = (modelId) =>
  TtsSynth.create(parseTtsModelId(modelId));

export class SherpaTextToSpeech implements TextToSpeech {
  readonly #createSynth: TtsSynthFactory;

  constructor(createSynth: TtsSynthFactory = defaultSynthFactory) {
    this.#createSynth = createSynth;
  }

  /** execution binds one reading's settings and reference into every segment request. */
  async open(config: TTSConfig = {}, execution?: SynthExecution): Promise<TTSSession> {
    if (typeof config.voice === "string") throw new Error("Resolve named local voices through the local output factory");
    const sid = config.voice ?? 0;
    validateSpeed(config.speed ?? 1);
    if (!Number.isInteger(sid) || sid < 0) throw new Error("Invalid TTS speaker");
    const synth = await this.#createSynth(config.model ?? "piper-amy");
    validateSpeaker(sid, synth.numSpeakers);
    return new SherpaTTSSession(synth, config.speed ?? 1, sid, execution);
  }
}

class SherpaTTSSession implements TTSSession {
  readonly #synth: SegmentSynth;
  readonly #speed: number;
  readonly #sid: number;
  readonly #execution: SynthExecution | undefined;
  #closed = false;

  constructor(synth: SegmentSynth, speed: number, sid: number, execution?: SynthExecution) {
    this.#synth = synth;
    this.#speed = speed;
    this.#sid = sid;
    this.#execution = execution;
  }

  speak(text: AsyncIterable<string> | string): AsyncIterable<AudioSegment> {
    if (this.#closed) throw new Error("TTS session is closed");
    // The first clause split is a pipeline setting: the planner consumes it, the voice never sees it.
    const { firstClause, ...settings } = this.#execution?.settings ?? {};
    const plan = { firstClause: firstClause !== false };
    const segments =
      typeof text === "string"
        ? fromArray(planSegments(text, plan))
        : planSegmentsStream(text, plan);
    const reference = this.#execution?.reference;
    // Bind the speaker id (kokoro is multi-voice) and the reading's settings into
    // the synth the shared pipeline drives. Bounds were checked at open.
    const voiced: SegmentSynth = {
      synth: (request) => {
        if (this.#closed) return Promise.reject(new Error("TTS session is closed"));
        return this.#synth.synth({ ...request, sid: this.#sid,
          ...(this.#execution ? { settings } : {}), ...(reference ? { reference } : {}) });
      },
    };
    const audio = synthPipeline(segments, voiced, this.#speed);
    const session = this;
    return (async function* () {
      try {
        for await (const segment of audio) {
          if (session.#closed) return;
          yield segment;
        }
      } catch (error) {
        if (!session.#closed) throw error;
      }
    })();
  }

  async close(): Promise<void> {
    this.#closed = true;
    // The sherpa OfflineTts binding exposes no explicit free; the native handle
    // is released when the TtsSynth is garbage collected. Closing suppresses
    // late delivery; it cannot abort a native generation already in flight.
  }
}
