export type AudioWorkKind = "room" | "comparison" | "tryout" | "api-input" | "legacy-input";
export type ReleaseAudioWork = () => void;

/** Comparisons share the scheduler's serial active/queued group. */
export class AudioWorkLease {
  #kind: AudioWorkKind | undefined;
  #reservations = 0;
  #closed = false;

  get active(): AudioWorkKind | undefined { return this.#kind; }

  tryAcquire(kind: AudioWorkKind): ReleaseAudioWork | undefined {
    if (this.#closed || this.#kind !== undefined &&
        !(kind === this.#kind && (kind === "comparison" || kind === "legacy-input"))) return undefined;
    this.#kind = kind;
    this.#reservations++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (--this.#reservations === 0) this.#kind = undefined;
    };
  }

  close(): void { this.#closed = true; }
}
