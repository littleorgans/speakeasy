import { assertSampleRate, SPEECH_SAMPLE_RATE } from "../audio/pcm.js";
import type { STTConfig, STTSession } from "../contract.ts";
import { BufferedOfflineSession, type DecodeAudio, type SilenceGate } from "./offline-session.ts";
import type { ResultDetail } from "./result.ts";

/**
 * The lifecycle every whole utterance engine shares: readiness, one load per
 * model lifetime shared by its sessions and retried after a failure, decode
 * tracking, and release. Each engine describes its family in an
 * `OfflineAdapter`; decoding stays with the family, and native code with the
 * adapter, so this module loads none.
 *
 * A model lifetime runs from the first load to release(). Every session it
 * opens decodes on its loaded value; release() waits for their decodes in
 * flight, then frees the value, and any later decode of theirs emits `error`
 * rather than reach a freed native.
 */

/**
 * A value loaded on first get() and shared by concurrent callers. A failed
 * load clears it, so the next get() retries; take() detaches it for the
 * caller to free. A load that fails after take() never clears a newer one.
 */
export class Lazy<T> {
  readonly #load: () => Promise<T>;
  #value: Promise<T> | undefined;

  constructor(load: () => Promise<T>) {
    this.#load = load;
  }

  get(): Promise<T> {
    if (this.#value === undefined) {
      const value: Promise<T> = this.#load().catch((error: unknown) => {
        if (this.#value === value) {
          this.#value = undefined;
        }
        throw error;
      });
      this.#value = value;
    }
    return this.#value;
  }

  take(): Promise<T> | undefined {
    const value = this.#value;
    this.#value = undefined;
    return value;
  }
}

/** How one engine family prepares, loads, decodes and frees its model. */
export type OfflineAdapter<Prepared, Loaded> = {
  /** The engine's name, in sample rate and released model errors. */
  name: string;
  /** Longest utterance a session buffers; later audio is dropped. */
  maxAudioMs: number;
  silenceGate: SilenceGate | undefined;
  /**
   * What prepare() completes, so what bench cold and warm timings include:
   * `preflight` installs or checks assets only; `load` also builds the
   * shared native resources.
   */
  prepare: "preflight" | "load";
  /** One decode at a time across every session, for a model holding decoder state. */
  serial?: boolean;
  /** Asset install or check and pure preparation; runs before each load attempt, unmemoized. */
  preflight(): Promise<Prepared>;
  /** Build the native resources every session of a lifetime shares. */
  load(prepared: Prepared): Promise<Loaded> | Loaded;
  /** Free them; omitted where the runtime frees by garbage collection. */
  release?(loaded: Loaded): Promise<void>;
  /** The family decode for one session. */
  decoder(loaded: Loaded, config: STTConfig | undefined): DecodeAudio;
};

/** One model lifetime: whether release() ended it, and its decodes in flight. */
type Lifetime = { ended: boolean; decodes: Set<Promise<unknown>> };

function newLifetime(): Lifetime {
  return { ended: false, decodes: new Set() };
}

export class OfflineLifecycle<Prepared, Loaded> {
  readonly #adapter: OfflineAdapter<Prepared, Loaded>;
  readonly #loaded: Lazy<Loaded>;
  #lifetime = newLifetime();
  #serial: Promise<unknown> = Promise.resolve();

  constructor(adapter: OfflineAdapter<Prepared, Loaded>) {
    this.#adapter = adapter;
    this.#loaded = new Lazy(async () => adapter.load(await adapter.preflight()));
  }

  async prepare(): Promise<void> {
    if (this.#adapter.prepare === "load") {
      await this.#loaded.get();
    } else {
      await this.#adapter.preflight();
    }
  }

  async open(config?: STTConfig): Promise<STTSession> {
    const { name, maxAudioMs, silenceGate } = this.#adapter;
    assertSampleRate(name, config);
    // Captured before the load, so a session opened during release() is stale.
    const lifetime = this.#lifetime;
    const decode = this.#adapter.decoder(await this.#loaded.get(), config);
    return new BufferedOfflineSession(
      (maxAudioMs * SPEECH_SAMPLE_RATE) / 1_000,
      (audio) => this.#track(lifetime, () => decode(audio)),
      silenceGate,
    );
  }

  /** End the model lifetime: wait for its decodes, then free what it loaded. */
  async release(): Promise<void> {
    const loaded = this.#loaded.take();
    const lifetime = this.#lifetime;
    lifetime.ended = true;
    this.#lifetime = newLifetime();
    await Promise.allSettled(lifetime.decodes);
    if (!loaded || !this.#adapter.release) {
      return;
    }
    let value: Loaded;
    try {
      value = await loaded;
    } catch {
      // The load failed, so it built nothing to free.
      return;
    }
    await this.#adapter.release(value);
  }

  #track(lifetime: Lifetime, decode: () => Promise<ResultDetail>): Promise<ResultDetail> {
    if (lifetime.ended) {
      return Promise.reject(new Error(`${this.#adapter.name}: model released`));
    }
    const result = this.#adapter.serial ? this.#serialized(decode) : decode();
    lifetime.decodes.add(result);
    const settled = () => lifetime.decodes.delete(result);
    result.then(settled, settled);
    return result;
  }

  /** Chain after the previous decode; a failure does not break the chain. */
  #serialized<T>(work: () => Promise<T>): Promise<T> {
    const result = this.#serial.then(work, work);
    this.#serial = result.catch(() => undefined);
    return result;
  }
}

/** A native resource its owner frees. */
export type Releasable = { release(): Promise<void> };

/** Free several native resources together. */
export async function releaseAll(...parts: Releasable[]): Promise<void> {
  await Promise.all(parts.map((part) => part.release()));
}
