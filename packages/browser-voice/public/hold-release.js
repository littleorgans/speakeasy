export const HOLD_RELEASE_TAIL_MS = 200;

/** Keeps capture open briefly so browser audio already in flight reaches STT. */
export class HoldReleaseGate {
  constructor({
    onCommit,
    onChange = () => {},
    schedule = (callback, delay) => window.setTimeout(callback, delay),
    cancelSchedule = (timer) => window.clearTimeout(timer),
  }) {
    this.onCommit = onCommit;
    this.onChange = onChange;
    this.schedule = schedule;
    this.cancelSchedule = cancelSchedule;
    this.held = false;
    this.releasing = false;
    this.timer = undefined;
    this.generation = 0;
  }

  get capturing() {
    return this.held || this.releasing;
  }

  press() {
    if (this.capturing) return false;
    this.held = true;
    this.onChange();
    return true;
  }

  release() {
    if (!this.held) return false;
    this.held = false;
    this.releasing = true;
    this.onChange();
    const generation = ++this.generation;
    this.timer = this.schedule(() => {
      if (generation !== this.generation) return;
      this.timer = undefined;
      this.releasing = false;
      this.onChange();
      this.onCommit();
    }, HOLD_RELEASE_TAIL_MS);
    return true;
  }

  cancel() {
    this.generation += 1;
    if (this.timer !== undefined) this.cancelSchedule(this.timer);
    this.timer = undefined;
    const changed = this.capturing;
    this.held = false;
    this.releasing = false;
    if (changed) this.onChange();
  }
}
