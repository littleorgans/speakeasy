import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after } from "node:test";
import { COMPLETE_MARKER, completeMarkerValue } from "./assets.ts";
import type { SherpaBinding } from "./sherpa-api.ts";
import {
  resolveModelPaths,
  SHERPA_MODELS,
  type SherpaModelId,
} from "./sherpa-models.ts";

/**
 * Test helpers for the engine tests: a model root whose entry already looks
 * installed, so engines open against fake bindings with no download. Roots
 * are removed after the test file runs.
 */
export async function seededModelRoot(model: SherpaModelId): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-models-"));
  after(() => rm(root, { recursive: true, force: true }));
  const paths = resolveModelPaths(SHERPA_MODELS[model], root);
  await mkdir(paths.dir, { recursive: true });
  await writeFile(join(paths.dir, COMPLETE_MARKER), completeMarkerValue(paths));
  return root;
}

/**
 * The silence inputs of the onnxruntime engine evidence (Whistle, Granite,
 * TDT): zeros, uniform noise by peak amplitude and by RMS, 1 to 3 s each,
 * from Granite's MINSTD generator so every run sees the same samples.
 */
export function silenceFixtures(sampleRate = 16_000): Array<[string, Float32Array]> {
  let seed = 12_345;
  const uniform = () => ((seed = (seed * 48_271) % 2_147_483_647) / 2_147_483_647) * 2 - 1;
  const noise = (seconds: number, peak: number) =>
    Float32Array.from({ length: sampleRate * seconds }, () => uniform() * peak);
  const fixtures: Array<[string, Float32Array]> = [];
  for (const seconds of [1, 2, 3]) {
    fixtures.push([`zeros ${seconds}s`, new Float32Array(sampleRate * seconds)]);
  }
  for (const peak of [0.001, 0.01]) {
    for (const seconds of [1, 2, 3]) {
      fixtures.push([`noise peak ${peak} ${seconds}s`, noise(seconds, peak)]);
    }
  }
  for (const rms of [0.001, 0.01]) {
    for (const seconds of [1, 2, 3]) {
      fixtures.push([`noise rms ${rms} ${seconds}s`, noise(seconds, rms * Math.sqrt(3))]);
    }
  }
  return fixtures;
}

export function bindingWith(
  parts: Partial<Omit<SherpaBinding, "version">>,
): SherpaBinding {
  const missing = class {
    constructor() {
      throw new Error("recognizer not faked in this test");
    }
  };
  return {
    version: "fake",
    OnlineRecognizer: parts.OnlineRecognizer ?? (missing as never),
    OfflineRecognizer: parts.OfflineRecognizer ?? (missing as never),
  };
}
