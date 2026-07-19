/**
 * The half-duplex conversation state machine. Every legal transition lives in
 * this one table so the loop can never drift into an ad-hoc state.
 *
 *   idle -> listening -> thinking -> speaking -> listening -> ...
 *
 * `idle` is the pre-start / stopped state, reachable from anywhere on stop().
 * thinking -> listening covers an empty or failed turn (recover, do not die).
 *
 * Barge-in is the documented v2 seam: it would add `speaking -> listening` on
 * user speech detected mid-playback. That transition belongs exactly here, at
 * the state boundary; v1 does not listen while speaking.
 */

export const CONVO_STATES = ["idle", "listening", "thinking", "speaking"] as const;

export type ConvoState = (typeof CONVO_STATES)[number];

const LEGAL_TRANSITIONS: Record<ConvoState, readonly ConvoState[]> = {
  idle: ["listening"],
  listening: ["thinking", "idle"],
  thinking: ["speaking", "listening", "idle"],
  speaking: ["listening", "idle"],
};

export function canTransition(from: ConvoState, to: ConvoState): boolean {
  return LEGAL_TRANSITIONS[from].includes(to);
}

export function assertTransition(from: ConvoState, to: ConvoState): void {
  if (!canTransition(from, to)) {
    throw new Error(`illegal convo transition: ${from} -> ${to}`);
  }
}
