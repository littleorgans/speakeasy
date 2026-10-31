import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { after, test } from "node:test";
import { COMPLETE_MARKER, completeMarkerValue } from "./assets.ts";
import {
  loadSileroDetector,
  sileroVadFiles,
  type Vad,
  type VadConfig,
} from "./silero-vad.ts";

/** A sherpa `Vad` that queues one segment per window holding a sample over 0.5. */
function fakeVad() {
  const fake = { configs: [] as VadConfig[], windows: [] as number[], resets: 0 };
  class FakeVad implements Vad {
    #segments = 0;
    constructor(config: VadConfig) {
      fake.configs.push(config);
    }
    acceptWaveform(samples: Float32Array): void {
      fake.windows.push(samples.length);
      if (samples.some((sample) => sample > 0.5)) this.#segments += 1;
    }
    flush(): void {}
    isEmpty(): boolean {
      return this.#segments === 0;
    }
    pop(): void {
      this.#segments -= 1;
    }
    reset(): void {
      fake.resets += 1;
    }
  }
  return { fake, binding: { Vad: FakeVad } };
}

async function modelRoot(installed: boolean): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-silero-"));
  after(() => rm(root, { recursive: true, force: true }));
  if (installed) {
    const files = sileroVadFiles(root);
    await mkdir(files.dir, { recursive: true });
    await writeFile(join(files.dir, COMPLETE_MARKER), completeMarkerValue(files));
  }
  return root;
}

test("the pinned model fails closed on a digest mismatch, before any VAD exists", async () => {
  const { binding, fake } = fakeVad();
  const root = await modelRoot(false);
  await assert.rejects(
    loadSileroDetector(binding, root, async () => Readable.from([Buffer.from("not silero")])),
    /Digest mismatch/,
  );
  assert.equal(fake.configs.length, 0);
});

test("a hold is scanned in 512 sample windows and holds speech when a segment remains", async () => {
  const { binding, fake } = fakeVad();
  const root = await modelRoot(true);
  const hasSpeech = await loadSileroDetector(binding, root);

  const { sileroVad, sampleRate } = fake.configs[0]!;
  assert.equal(sileroVad.model, join(root, "silero-vad", "silero_vad.onnx"));
  assert.equal(sileroVad.windowSize, 512);
  assert.equal(sileroVad.minSpeechDuration, 0.1);
  assert.equal(sampleRate, 16_000);

  assert.equal(hasSpeech(new Float32Array(1_600)), false);
  assert.deepEqual(fake.windows, [512, 512, 512], "the partial last window is dropped");
  const loud = new Float32Array(1_600);
  loud[1_000] = 0.9;
  assert.equal(hasSpeech(loud), true);
  assert.equal(hasSpeech(new Float32Array(1_600)), false, "segments never carry into the next hold");
  assert.equal(fake.resets, 3);
});
