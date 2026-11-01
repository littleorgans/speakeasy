import { createHash } from "node:crypto";
import type { HotwordFileSnapshot } from "@speakeasy/speech-io";
import type { AnySherpaEngine, FinalEvent, Hotwords } from "@speakeasy/speech-io/stt";
import { PROMPT_SAMPLE_RATE } from "./prompt-reader.ts";
import { TRANSCRIBE_MODEL_ID, TranscriptionError, type Transcript } from "./transcriber.ts";

/** One file snapshot per request; one active recognizer and decode at a time. */
export class TakeTranscription {
  readonly #read: () => Promise<HotwordFileSnapshot>;
  readonly #createEngine: (hotwords: Hotwords | undefined) => Promise<AnySherpaEngine>;
  #engine: AnySherpaEngine | undefined;
  #content: string | undefined;
  #queue: Promise<unknown> = Promise.resolve();

  constructor(options: {
    readSnapshot(): Promise<HotwordFileSnapshot>;
    createEngine(hotwords: Hotwords | undefined): Promise<AnySherpaEngine>;
  }) {
    this.#read = options.readSnapshot;
    this.#createEngine = options.createEngine;
  }

  transcribe(samples: Float32Array): Promise<Transcript> {
    // Read immediately, before an earlier decode can delay this take. Handle
    // read rejection now as well, so queued failures are never unhandled.
    const snapshot = this.#read().then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    );
    const result = this.#queue.then(async () => {
      const captured = await snapshot;
      if ("error" in captured) throw new TranscriptionError(String(captured.error), TRANSCRIBE_MODEL_ID);
      const { source, content, hotwords } = captured.value;
      const status = {
        path: source, count: hotwords?.phrases.length ?? 0,
        revision: createHash("sha256").update(content).digest("hex"),
      };
      try {
        if (!this.#engine || this.#content !== content) {
          // Drop the old engine before constructing its replacement. A failed
          // new snapshot can never decode through the old recognizer.
          this.#engine = undefined;
          this.#content = undefined;
          const engine = await this.#createEngine(hotwords);
          await engine.prepare();
          this.#engine = engine;
          this.#content = content;
        }
        return { ...(await decode(this.#engine, samples)), model: TRANSCRIBE_MODEL_ID, hotwords: status };
      } catch (error) {
        throw new TranscriptionError(error instanceof Error ? error.message : String(error), TRANSCRIBE_MODEL_ID, status);
      }
    });
    this.#queue = result.catch(() => {});
    return result;
  }
}

async function decode(engine: AnySherpaEngine, samples: Float32Array): Promise<{ text: string; decodeMs: number }> {
  const session = await engine.open({ sampleRate: PROMPT_SAMPLE_RATE });
  try {
    return await new Promise((resolve, reject) => {
      session.once("final", ({ text, decodeMs }: FinalEvent) => resolve({ text, decodeMs: decodeMs ?? 0 }));
      session.once("empty", ({ decodeMs }: { decodeMs?: number }) => resolve({ text: "", decodeMs: decodeMs ?? 0 }));
      session.once("error", ({ err }: { err: unknown }) => reject(err));
      session.pushAudio(samples);
      session.flush();
    });
  } finally {
    await session.end();
  }
}
