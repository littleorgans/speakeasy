import { performance } from "node:perf_hooks";
import type { FinalEvent, STTSession } from "../contract.ts";

/** What flush() committed: trimmed final text ("" for empty) and its arrival. */
export type CommitObservation = { text: string; at: number };

export type FlushCommit = {
  /** Stamped after the commit listeners register, immediately before flush(). */
  releaseAt: number;
  /** Undefined when neither final nor empty arrived within the timeout. */
  observation: CommitObservation | undefined;
  /** observation.at - releaseAt; NaN on timeout. */
  releaseToCommitMs: number;
};

/**
 * The push-to-talk commit transaction: register the final/empty listeners,
 * stamp the release, flush(), await the commit. Listeners register BEFORE
 * flush() because the contract makes no synchrony guarantee: sherpa commits
 * during flush(), others commit from a later task. Listeners and the timer
 * are released on every outcome; a throwing flush() rethrows after cleanup.
 * Session lifetime (end, error events) stays with the caller.
 */
export async function flushCommit(
  session: STTSession,
  timeoutMs: number,
): Promise<FlushCommit> {
  const waiter = nextCommit(session, timeoutMs);
  const releaseAt = performance.now();
  try {
    session.flush();
  } catch (error) {
    waiter.release();
    throw error;
  }
  const observation = await waiter.committed;
  return {
    releaseAt,
    observation,
    releaseToCommitMs: observation ? observation.at - releaseAt : Number.NaN,
  };
}

function nextCommit(
  session: STTSession,
  timeoutMs: number,
): { committed: Promise<CommitObservation | undefined>; release: () => void } {
  let release = (): void => {};
  const committed = new Promise<CommitObservation | undefined>((resolve) => {
    const cleanup = (): void => {
      clearTimeout(timer);
      session.off("final", onFinal);
      session.off("empty", onEmpty);
    };
    const finish = (text: string): void => {
      cleanup();
      resolve({ text, at: performance.now() });
    };
    const onFinal = (event: FinalEvent): void => finish(event.text.trim());
    const onEmpty = (): void => finish("");
    const timer = setTimeout(() => {
      cleanup();
      resolve(undefined);
    }, timeoutMs);
    session.once("final", onFinal);
    session.once("empty", onEmpty);
    release = cleanup;
  });
  return { committed, release };
}
