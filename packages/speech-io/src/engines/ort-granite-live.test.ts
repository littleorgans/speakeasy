import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { FinalEvent } from "../contract.ts";
import { readWavFrames } from "../bench/wav.ts";
import { GRANITE_MODELS, type GraniteModelId } from "./ort-granite-models.ts";
import { createLocalOrtEngine } from "./ort-local.ts";

/**
 * Live smoke: downloads each Granite model into ./models/ort (run from the
 * repo root) and transcribes a clip spoken by macOS `say`; same phrase and
 * frame size as the Whistle live test. Opt in with SPEAKEASY_LIVE_ORT=1.
 */

const LIVE = process.env.SPEAKEASY_LIVE_ORT === "1";
const PHRASE = "Turn off the kitchen lights.";
const FRAME_SAMPLES = 320;

async function spokenClip(): Promise<Float32Array> {
  const dir = await mkdtemp(join(tmpdir(), "speak-easy-ort-live-"));
  try {
    const wav = join(dir, "clip.wav");
    execFileSync("say", ["-o", wav, "--file-format=WAVE", "--data-format=LEI16@16000", PHRASE]);
    return (await readWavFrames(wav)).samples;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/** The English export emits lowercase text without punctuation. */
function normalized(text: string): string {
  return text.toLowerCase().replace(/[^a-z ]/g, "").trim();
}

for (const id of Object.keys(GRANITE_MODELS) as GraniteModelId[]) {
  test(
    `${id} transcribes a spoken clip`,
    { skip: !LIVE || process.platform !== "darwin", timeout: 600_000 },
    async (t) => {
      const clip = await spokenClip();
      const engine = createLocalOrtEngine(id);
      const session = await engine.open();
      const decodeMs: number[] = [];
      for (let run = 0; run < 3; run += 1) {
        for (let start = 0; start < clip.length; start += FRAME_SAMPLES) {
          session.pushAudio(clip.subarray(start, start + FRAME_SAMPLES));
        }
        session.flush();
        const [final] = (await once(session, "final")) as [FinalEvent];
        assert.equal(normalized(final.text), normalized(PHRASE));
        assert.ok(final.tokens?.length);
        decodeMs.push(Math.round(final.decodeMs ?? NaN));
        if (run === 0) t.diagnostic(`text: ${final.text} (audioMs ${final.audioMs})`);
      }
      // Recorded, not asserted: the model emits filler words on silence.
      session.pushAudio(new Float32Array(32_000));
      session.flush();
      const silence = await Promise.race([
        once(session, "final").then(([event]) => `final ${JSON.stringify((event as FinalEvent).text)}`),
        once(session, "empty").then(() => "empty"),
      ]);
      await session.end();
      t.diagnostic(`label: ${engine.label}`);
      t.diagnostic(`decodeMs (cold, warm, warm): ${decodeMs.join(", ")}`);
      t.diagnostic(`2 s of zeros: ${silence}`);
      t.diagnostic(`peak RSS MiB: ${Math.round(process.resourceUsage().maxRSS / 1024)}`);
    },
  );
}
