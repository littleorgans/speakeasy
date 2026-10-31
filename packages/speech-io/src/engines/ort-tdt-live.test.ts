import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { FinalEvent } from "../contract.ts";
import { normalizeTranscript } from "../bench/normalize.ts";
import { readWavFrames } from "../bench/wav.ts";
import { ORT_TDT_MODELS, type OrtTdtModelId } from "./ort-tdt-models.ts";
import { createLocalOrtTdtEngine } from "./ort-tdt-runtime.ts";

/**
 * Live smoke: downloads each TDT model into ./models/ort (run from the repo
 * root) and transcribes a clip spoken by macOS `say`. Opt in with
 * SPEAKEASY_LIVE_ORT=1; peak RSS is per process, so run one model per
 * process (--test-name-pattern) to read it per model.
 */

const LIVE = process.env.SPEAKEASY_LIVE_ORT === "1";
const PHRASE = "Open the settings file, then save it as version two.";
const FRAME_SAMPLES = 320;

async function spokenClip(): Promise<Float32Array> {
  const dir = await mkdtemp(join(tmpdir(), "speak-easy-ort-tdt-live-"));
  try {
    const wav = join(dir, "clip.wav");
    execFileSync("say", ["-o", wav, "--file-format=WAVE", "--data-format=LEI16@16000", PHRASE]);
    return (await readWavFrames(wav)).samples;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

for (const id of Object.keys(ORT_TDT_MODELS) as OrtTdtModelId[]) {
  test(
    `${id} transcribes a spoken clip`,
    { skip: !LIVE || process.platform !== "darwin", timeout: 1_200_000 },
    async (t) => {
      const clip = await spokenClip();
      const engine = createLocalOrtTdtEngine(id);
      const session = await engine.open();
      const decodeMs: number[] = [];
      for (let run = 0; run < 3; run += 1) {
        for (let start = 0; start < clip.length; start += FRAME_SAMPLES) {
          session.pushAudio(clip.subarray(start, start + FRAME_SAMPLES));
        }
        session.flush();
        const [final] = (await once(session, "final")) as [FinalEvent];
        assert.equal(normalizeTranscript(final.text), normalizeTranscript(PHRASE));
        assert.ok(final.tokens?.length);
        decodeMs.push(Math.round(final.decodeMs ?? NaN));
        if (run === 0) t.diagnostic(`text: ${final.text}`);
      }
      await session.end();
      await engine.release();
      t.diagnostic(`label: ${engine.label}`);
      t.diagnostic(`decodeMs (cold, warm, warm): ${decodeMs.join(", ")}`);
      t.diagnostic(`peak RSS MB: ${Math.round(process.resourceUsage().maxRSS / 1_024)}`);
    },
  );
}
