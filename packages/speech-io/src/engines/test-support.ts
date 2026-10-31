import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after } from "node:test";
import { COMPLETE_MARKER } from "./assets.ts";
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
  await writeFile(join(paths.dir, COMPLETE_MARKER), paths.sha256 ?? "");
  return root;
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
