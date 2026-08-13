export const HOLD_RELEASE_TAIL_MS: number;

export class HoldReleaseGate {
  constructor(options: {
    onCommit: () => void;
    onChange?: () => void;
    schedule?: (callback: () => void, delay: number) => number;
    cancelSchedule?: (timer: number) => void;
  });
  readonly held: boolean;
  readonly releasing: boolean;
  readonly capturing: boolean;
  press(): boolean;
  release(): boolean;
  cancel(): void;
}
