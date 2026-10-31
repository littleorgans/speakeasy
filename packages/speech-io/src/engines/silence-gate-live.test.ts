import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { FinalEvent, STTSession } from "../contract.ts";
import { normalizeTranscript } from "../bench/normalize.ts";
import { readWavFrames } from "../bench/wav.ts";
import { createLocalOrtEngine } from "./ort-local.ts";
import { ORT_MODELS, resolveOrtModel, type OrtModelId } from "./ort-models.ts";
import type { SilenceGateMode } from "./offline-session.ts";
import { createLocalSherpaEngine, localSilenceGate } from "./sherpa-binding.ts";
import { silenceFixtures } from "./test-support.ts";

/**
 * Live: every onnxruntime model and Canary on the evidence silence fixtures and on
 * spoken clips, with the silence gate off and on. Off records what each model
 * says to silence; on, every fixture answers empty and speech still decodes.
 * Downloads Silero VAD into ./models/sherpa (run from the repo root). Opt in
 * with SPEAKEASY_LIVE_ORT=1.
 */

const LIVE = process.env.SPEAKEASY_LIVE_ORT === "1";
const PHRASE = "Turn off the kitchen lights.";
/** Single short words: the detector's 100 ms minimum must keep them. */
const SHORT_WORDS = ["Go.", "No."];
const FRAME_SAMPLES = 320;
const CANARY = "en-canary-180m-flash-int8";
const MODELS = [
  ...Object.keys(ORT_MODELS).map((id) => ({
    id,
    floor: resolveOrtModel(id).silenceGate?.logProbFloor ?? null,
    modes: ["off", "on"] as const,
    build: (gate: SilenceGateMode) => createLocalOrtEngine(id as OrtModelId, gate),
  })),
  {
    id: CANARY,
    floor: null,
    // Exercise the shipped descriptor policy on the sherpa path.
    modes: ["off", "model"] as const,
    build: (gate: SilenceGateMode) => createLocalSherpaEngine(CANARY, gate),
  },
];

async function spoken(text: string): Promise<Float32Array> {
  const dir = await mkdtemp(join(tmpdir(), "speak-easy-gate-live-"));
  try {
    const wav = join(dir, "clip.wav");
    execFileSync("say", ["-o", wav, "--file-format=WAVE", "--data-format=LEI16@16000", text]);
    return (await readWavFrames(wav)).samples;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

type Outcome = { text: string | null; decodeMs: number; meanLogProb: number | null };

async function decode(session: STTSession, audio: Float32Array): Promise<Outcome> {
  for (let start = 0; start < audio.length; start += FRAME_SAMPLES) {
    session.pushAudio(audio.subarray(start, start + FRAME_SAMPLES));
  }
  // One shared controller drops both waits, and their error listeners, once either fires.
  const done = new AbortController();
  const result = Promise.race([
    once(session, "final", { signal: done.signal }).then(([event]) => event as FinalEvent),
    once(session, "empty", { signal: done.signal }).then(([event]) => ({ ...(event as object), text: null })),
  ]);
  session.flush();
  const event = (await result) as { text: string | null; decodeMs?: number; tokens?: FinalEvent["tokens"] };
  done.abort();
  const scores = (event.tokens ?? []).flatMap((token) => (token.logProb === null ? [] : [token.logProb]));
  return {
    text: event.text,
    decodeMs: Math.round(event.decodeMs ?? Number.NaN),
    meanLogProb: scores.length > 0 ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null,
  };
}

function describe({ text, decodeMs, meanLogProb }: Outcome): string {
  const score = meanLogProb === null ? "" : ` meanLogProb ${meanLogProb.toFixed(3)}`;
  return `${text === null ? "empty" : `final ${JSON.stringify(text)}`} ${decodeMs} ms${score}`;
}

test("the silence detector keeps Go. and No. before any decoder", {
  skip: !LIVE || process.platform !== "darwin",
}, async () => {
  const gate = await localSilenceGate(undefined, "on");
  for (const word of SHORT_WORDS) {
    assert.equal(gate!.hasSpeech(await spoken(word)), true, word);
  }
});

for (const { id, floor, modes, build } of MODELS) {
  test(
    `${id}: the silence gate empties silence and keeps speech`,
    { skip: !LIVE || process.platform !== "darwin", timeout: 1_200_000 },
    async (t) => {
      const phrase = await spoken(PHRASE);
      const words = await Promise.all(SHORT_WORDS.map(spoken));
      const ungatedWords: Array<string | null> = [];
      /** With the gate off, a floor must sit under speech and over hallucination. */
      const calibrated = (outcome: Outcome, speech: boolean, name: string) => {
        if (floor === null || outcome.text === null || outcome.meanLogProb === null) return;
        assert.equal(outcome.meanLogProb >= floor, speech, `${name} scores ${outcome.meanLogProb}`);
      };
      for (const gate of modes) {
        const engine = await build(gate);
        const session = await engine.open({ language: "en" });
        await decode(session, phrase); // warm up
        const speech = await decode(session, phrase);
        t.diagnostic(`gate ${gate} speech: ${describe(speech)}`);
        assert.equal(normalizeTranscript(speech.text ?? ""), normalizeTranscript(PHRASE));
        calibrated(speech, true, PHRASE);
        for (const [index, word] of words.entries()) {
          const outcome = await decode(session, word);
          t.diagnostic(`gate ${gate} ${SHORT_WORDS[index]}: ${describe(outcome)}`);
          if (gate === "off") ungatedWords.push(outcome.text);
          else assert.equal(outcome.text, ungatedWords[index], `${SHORT_WORDS[index]} changed with the gate`);
          calibrated(outcome, true, SHORT_WORDS[index]!);
        }
        for (const [name, audio] of silenceFixtures()) {
          const outcome = await decode(session, audio);
          t.diagnostic(`gate ${gate} ${name}: ${describe(outcome)}`);
          if (gate !== "off") assert.equal(outcome.text, null, name);
          calibrated(outcome, false, name);
        }
        await session.end();
        if ("release" in engine) await engine.release();
      }
    },
  );
}
