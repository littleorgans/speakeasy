import type { NamesSnapshot } from "./names.ts";
import {
  assertLivePartial, assertSetupLoadResult, type LivePartial, type LiveSettings, type SetupExecutor, type SetupId,
} from "./setup-protocol.ts";

export type LiveTakeEvent =
  | { type: "loading" }
  /** cold: this load started the model, so its time is model start. */
  | { type: "ready"; cold: boolean }
  | ({ type: "partial" } & LivePartial)
  | { type: "error"; message: string };

export type LiveTakeOptions = {
  setupId: SetupId;
  /** Applied when the stream starts; absent means the engine's defaults. */
  settings?: LiveSettings;
  /** The take's Names list, for a setup that takes names; its worker loads with it. */
  names?: NamesSnapshot;
  executor: SetupExecutor;
  maxSamples: number;
  onEvent(event: LiveTakeEvent): void;
  /** True while this take may call the executor: no comparison runs and the host admits audio work. */
  admit(): boolean;
  /** After each executor call, so an idle release never retires a live model. */
  touch(): void;
  /** Once the take has stopped and its last call settled. */
  finished(): void;
};

const ADMIT_RETRY_MS = 50;

/**
 * Live words for one held take. Audio queues as it arrives; whenever the
 * executor is free, everything queued goes in one feed, so a slow model reads
 * larger slices and never falls behind. Loads on first use. Only changed
 * words are reported. Stopping drops queued audio: the final pass hears the
 * whole take anyway.
 */
export class LiveTake {
  readonly setupId: SetupId;
  readonly settled: Promise<void>;
  readonly #options: LiveTakeOptions;
  readonly #controller = new AbortController();
  #queued: Float32Array[] = [];
  #samples = 0;
  #stopped = false;
  #wake: (() => void) | undefined;

  constructor(options: LiveTakeOptions) {
    if (!options.executor.feed) throw new Error(`${options.setupId} gives final text only`);
    this.setupId = options.setupId;
    this.#options = options;
    this.settled = this.#run();
  }

  get stopped(): boolean { return this.#stopped; }

  push(samples: Float32Array): void {
    if (this.#stopped || this.#samples + samples.length > this.#options.maxSamples) return;
    this.#queued.push(samples);
    this.#samples += samples.length;
    this.#wake?.();
  }

  /** Stops feeding; an in flight call still settles before `settled`. */
  stop(): Promise<void> {
    this.#stopped = true;
    this.#queued = [];
    this.#wake?.();
    return this.settled;
  }

  /** Stops and aborts the in flight call, retiring its worker generation. */
  cancel(): Promise<void> {
    this.#controller.abort();
    return this.stop();
  }

  async #run(): Promise<void> {
    const { executor, setupId, names, onEvent } = this.#options;
    const signal = this.#controller.signal;
    let loaded = false;
    let start = true;
    let words = "";
    try {
      while (!this.#stopped) {
        if (!this.#queued.length || !this.#options.admit()) {
          await this.#sleep(this.#queued.length ? ADMIT_RETRY_MS : undefined);
          continue;
        }
        if (!loaded) {
          onEvent({ type: "loading" });
          const load = await executor.load(setupId, { signal, ...(names ? { names } : {}) });
          assertSetupLoadResult(load);
          loaded = true;
          this.#options.touch();
          // Sent even after a release: the page counts a cold load as model start.
          onEvent({ type: "ready", cold: load.startupMs !== 0 });
          continue;
        }
        const samples = concat(this.#queued);
        this.#queued = [];
        const partial = await executor.feed!(setupId, samples, start, { signal }, this.#options.settings);
        assertLivePartial(partial);
        start = false;
        this.#options.touch();
        const heard = `${partial.stable}\u0000${partial.tentative}`;
        if (heard !== words && !this.#stopped) onEvent({ type: "partial", ...partial });
        words = heard;
      }
    } catch (error) {
      if (!this.#stopped) onEvent({ type: "error", message: error instanceof Error ? error.message : String(error) });
      this.#stopped = true;
    } finally {
      this.#options.finished();
    }
  }

  #sleep(ms?: number): Promise<void> {
    return new Promise((resolve) => {
      const timer = ms === undefined ? undefined : setTimeout(() => wake(), ms);
      const wake = () => { clearTimeout(timer); this.#wake = undefined; resolve(); };
      this.#wake = wake;
    });
  }
}

function concat(parts: readonly Float32Array[]): Float32Array {
  const joined = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { joined.set(part, offset); offset += part.length; }
  return joined;
}
