import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import type { STTSession, VoiceToText } from "../contract.ts";
import { MoonshineEngine } from "../engines/moonshine.ts";
import { SherpaEngine } from "../engines/sherpa.ts";
import type { SherpaModelId } from "../engines/sherpa-models.ts";
import { StubEngine } from "../engines/stub.ts";
import { FLUSH_FINAL_TIMEOUT_MS, SESSION_TIMEOUT_MS } from "./config.ts";
import { isWordTolerantTranscript, wordErrorCount } from "./transcript.ts";
import type {
  BenchEngine,
  EngineName,
  FinalObservation,
  PttRunResult,
} from "./types.ts";
import type { WavAudio } from "./wav.ts";

/**
 * Session plumbing shared by the bench modes (sweep, ptt, corpus): engine
 * construction, event observation, timeouts, and the single-utterance
 * push-to-talk run.
 */

export type SessionObservers = {
  readonly firstPartialAt?: number;
  finals: FinalObservation[];
  finalAtIndex: (
    index: number,
    timeoutMs: number,
  ) => Promise<FinalObservation | undefined>;
  throwIfError: () => void;
};

export function createEngine(
  name: EngineName,
  model?: SherpaModelId,
): BenchEngine {
  if (model && name !== "sherpa") {
    throw new Error(`--model only applies to --engine sherpa, not ${name}`);
  }
  if (name === "stub") {
    return new StubEngine();
  }
  if (name === "moonshine") {
    return new MoonshineEngine();
  }
  if (name === "sherpa") {
    return new SherpaEngine(model);
  }
  throw new Error(`Unsupported engine ${name}`);
}

export function attachSessionObservers(session: STTSession): SessionObservers {
  const endpointEvents: number[] = [];
  const finalTexts: string[] = [];
  const finals: FinalObservation[] = [];
  const finalWaiters: Array<() => void> = [];
  const state: { firstPartialAt?: number; error?: unknown } = {};

  session.on("partial", () => {
    state.firstPartialAt ??= performance.now();
  });
  session.on("endpoint", () => {
    endpointEvents.push(performance.now());
  });
  session.on("final", ({ text }: { text: string }) => {
    const finalAt = performance.now();
    const trimmed = text.trim();
    if (trimmed) {
      finalTexts.push(trimmed);
    }
    finals.push({
      endpointAt: lastEndpointBefore(endpointEvents, finalAt) ?? finalAt,
      finalAt,
      text: finalTexts.join(" ").trim(),
    });
    for (const notify of finalWaiters.splice(0)) {
      notify();
    }
  });
  session.on("error", ({ err }: { err: unknown }) => {
    state.error = err;
  });

  return {
    get firstPartialAt() {
      return state.firstPartialAt;
    },
    finals,
    // Resolves with finals[index] as soon as it exists (immediately if the
    // engine committed synchronously, on the event if it commits async),
    // or undefined after timeoutMs.
    finalAtIndex(index, timeoutMs) {
      if (finals.length > index) {
        return Promise.resolve(finals[index]);
      }
      return new Promise((resolve) => {
        const timer = setTimeout(() => {
          resolve(undefined);
        }, timeoutMs);
        finalWaiters.push(() => {
          clearTimeout(timer);
          resolve(finals[index]);
        });
      });
    },
    throwIfError() {
      if (state.error) {
        throw state.error;
      }
    },
  };
}

export async function runPttOnce(input: {
  run: number;
  engine: VoiceToText;
  audio: WavAudio;
  releaseFrame: number;
  frameMs: number;
  expected: string;
}): Promise<PttRunResult> {
  // Push-to-talk: the consumer finalizes externally (button release), so the
  // engine's own silence endpointing is disabled and flush() is the commit
  // path. The final is awaited as an EVENT because the contract makes no
  // synchrony guarantee for flush(): sherpa commits synchronously, moonshine
  // commits from an async task.
  const session = await input.engine.open({
    sampleRate: input.audio.sampleRate,
    endpoint: { mode: "manual" },
  });
  const state = attachSessionObservers(session);
  const start = performance.now();

  for (let index = 0; index <= input.releaseFrame; index += 1) {
    session.pushAudio(input.audio.frames[index]!);
    if (index < input.releaseFrame) {
      await delay(input.frameMs);
    }
  }

  const finalsBefore = state.finals.length;
  const flushAt = performance.now();
  session.flush();
  const final = await state.finalAtIndex(finalsBefore, FLUSH_FINAL_TIMEOUT_MS);
  const flushToFinalMs = final ? final.finalAt - flushAt : Number.NaN;
  const finalText = final?.text ?? "";

  await withTimeout(session.end(), SESSION_TIMEOUT_MS);
  state.throwIfError();

  return {
    run: input.run,
    flushToFinalMs,
    firstPartialMs: state.firstPartialAt
      ? state.firstPartialAt - start
      : undefined,
    wordErrors: wordErrorCount(finalText, input.expected),
    finalText,
    textCorrect: isWordTolerantTranscript(finalText, input.expected),
  };
}

export function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Session timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function lastEndpointBefore(
  endpoints: number[],
  finalAt: number,
): number | undefined {
  return endpoints.findLast((endpointAt) => endpointAt <= finalAt);
}
