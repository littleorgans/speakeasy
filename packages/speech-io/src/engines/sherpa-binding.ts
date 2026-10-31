import { createRequire } from "node:module";
import { join } from "node:path";
import { loadHotwords } from "./hotwords.ts";
import type { SherpaBinding } from "./sherpa-api.ts";
import { createSherpaEngine, type AnySherpaEngine } from "./sherpa-factory.ts";
import { DEFAULT_SHERPA_MODEL, type SherpaModelId } from "./sherpa-models.ts";

/**
 * speak-easy's own sherpa wiring for the CLI paths (demo, bench, convo
 * runtime): the binding from this package's node_modules, models under the
 * working directory, and ./hotwords.txt. Outside the `./stt` entry, so hosts
 * that ship the native separately never bundle the require.
 */

/** On-disk root for downloaded + extracted sherpa models. */
export const SHERPA_ROOT = join(process.cwd(), "models", "sherpa");

export function loadSherpaBinding(): SherpaBinding {
  const require = createRequire(import.meta.url);
  return require("sherpa-onnx-node") as SherpaBinding;
}

export async function createLocalSherpaEngine(
  model: SherpaModelId = DEFAULT_SHERPA_MODEL,
): Promise<AnySherpaEngine> {
  return createSherpaEngine({
    model,
    modelRoot: SHERPA_ROOT,
    binding: loadSherpaBinding(),
    hotwords: await loadHotwords(),
  });
}
