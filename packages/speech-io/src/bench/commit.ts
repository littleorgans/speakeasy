import { performance } from "node:perf_hooks";
import type { FinalEvent, STTSession } from "../contract.ts";

export type CommitObservation = { text: string; at: number };

/** Register before flush(): engines can answer synchronously with final or empty. */
export function nextCommit(
  session: STTSession,
  timeoutMs: number,
): Promise<CommitObservation | undefined> {
  return new Promise((resolve) => {
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
  });
}
