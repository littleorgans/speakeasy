import assert from "node:assert/strict";
import { once } from "node:events";
import { test } from "node:test";
import type { EmptyEvent, FinalEvent, STTErrorEvent } from "../contract.ts";
import {
  BufferedOfflineSession,
  gatesSilence,
  type SilenceGate,
} from "./offline-session.ts";
import type { ResultDetail } from "./result.ts";

const HELD = new Float32Array(1_600);

function sessionWith(detail: ResultDetail, gate?: SilenceGate) {
  const decoded: number[] = [];
  const checked: number[] = [];
  const watched = gate && {
    ...gate,
    hasSpeech: (audio: Float32Array) => {
      checked.push(audio.length);
      return gate.hasSpeech(audio);
    },
  };
  const session = new BufferedOfflineSession(
    16_000,
    async (audio) => {
      decoded.push(audio.length);
      return detail;
    },
    watched,
  );
  return { session, decoded, checked };
}

function scored(...logProbs: Array<number | null>): ResultDetail {
  return {
    text: "thank you",
    tokens: logProbs.map((logProb, index) => ({ text: ` w${index}`, startMs: index * 80, logProb })),
  };
}

/** Which event a flushed hold answers with, and its payload. */
async function flushed(session: BufferedOfflineSession): Promise<"final" | "empty"> {
  return (await answer(session))[0];
}

async function answer(
  session: BufferedOfflineSession,
): Promise<readonly ["final", FinalEvent] | readonly ["empty", EmptyEvent]> {
  const event = Promise.race([
    once(session, "final").then(([final]) => ["final", final as FinalEvent] as const),
    once(session, "empty").then(([empty]) => ["empty", empty as EmptyEvent] as const),
  ]);
  session.pushAudio(HELD);
  session.flush();
  return event;
}

test("without a gate every hold with audio decodes", async () => {
  const { session, decoded } = sessionWith(scored(-3));
  assert.equal(await flushed(session), "final");
  assert.deepEqual(decoded, [1_600]);
});

test("a hold the detector finds silent answers empty without decoding", async () => {
  const { session, decoded, checked } = sessionWith(scored(-0.01), {
    hasSpeech: () => false,
    logProbFloor: null,
  });
  const [kind, event] = await answer(session);
  assert.equal(kind, "empty");
  assert.equal(event.audioMs, 100);
  assert.deepEqual(checked, [1_600]);
  assert.deepEqual(decoded, []);
});

test("speech decodes to final through the gate", async () => {
  const { session, decoded } = sessionWith(scored(-0.01), { hasSpeech: () => true, logProbFloor: null });
  const [kind, event] = await answer(session);
  assert.equal(kind, "final");
  assert.equal((event as FinalEvent).text, "thank you");
  assert.deepEqual(decoded, [1_600]);
});

test("a final whose mean token log probability is under the floor becomes empty", async () => {
  const gate = { hasSpeech: () => true, logProbFloor: -0.3 };
  assert.equal(await flushed(sessionWith(scored(-0.47, -0.59), gate).session), "empty");
  // Mean, not minimum: one unsure word among sure ones stays.
  assert.equal(await flushed(sessionWith(scored(-0.9, -0.01, -0.01, -0.01), gate).session), "final");
  // Unscored tokens are skipped; a final with none is never floored.
  assert.equal(await flushed(sessionWith(scored(null, -0.01), gate).session), "final");
  assert.equal(await flushed(sessionWith(scored(null), gate).session), "final");
});

test("a detector failure surfaces as an error event", { timeout: 5_000 }, async () => {
  const { session, decoded } = sessionWith(scored(-0.01), {
    hasSpeech: () => {
      throw new Error("vad failed");
    },
    logProbFloor: null,
  });
  const error = once(session, "error") as Promise<[STTErrorEvent]>;
  session.pushAudio(HELD);
  session.flush();
  assert.match(String((await error)[0].err), /vad failed/);
  assert.deepEqual(decoded, []);
});

test("the descriptor decides the gate unless the mode forces it", () => {
  const policy = { logProbFloor: null };
  assert.equal(gatesSilence(policy, "model"), true);
  assert.equal(gatesSilence(undefined, "model"), false);
  assert.equal(gatesSilence(undefined, "on"), true);
  assert.equal(gatesSilence(policy, "off"), false);
});
