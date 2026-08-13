import assert from "node:assert/strict";
import { test } from "node:test";
import {
  HOLD_RELEASE_TAIL_MS,
  HoldReleaseGate,
} from "../public/hold-release.js";

test("hold release captures the browser audio tail before committing", () => {
  let scheduled: (() => void) | undefined;
  let delay = 0;
  let commits = 0;
  const gate = new HoldReleaseGate({
    onCommit: () => commits += 1,
    schedule: (callback: () => void, milliseconds: number) => {
      scheduled = callback;
      delay = milliseconds;
      return 1;
    },
    cancelSchedule: () => {},
  });

  assert.equal(gate.press(), true);
  assert.equal(gate.held, true);
  assert.equal(gate.capturing, true);
  assert.equal(gate.release(), true);
  assert.equal(gate.held, false);
  assert.equal(gate.releasing, true);
  assert.equal(gate.capturing, true);
  assert.equal(commits, 0);
  assert.equal(delay, HOLD_RELEASE_TAIL_MS);

  scheduled?.();
  assert.equal(gate.releasing, false);
  assert.equal(gate.capturing, false);
  assert.equal(commits, 1);
});

test("cancelling a pending release prevents a stale commit", () => {
  let scheduled: (() => void) | undefined;
  let cancelled = 0;
  let commits = 0;
  const gate = new HoldReleaseGate({
    onCommit: () => commits += 1,
    schedule: (callback: () => void) => {
      scheduled = callback;
      return 7;
    },
    cancelSchedule: (timer: number) => cancelled = timer,
  });

  gate.press();
  gate.release();
  gate.cancel();
  assert.equal(cancelled, 7);
  assert.equal(gate.capturing, false);
  scheduled?.();
  assert.equal(commits, 0);
});
