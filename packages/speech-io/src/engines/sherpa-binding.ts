import { createRequire } from "node:module";
import { join } from "node:path";
import { loadHotwords, type Hotwords } from "./hotwords.ts";
import {
  gatesSilence,
  type SilenceGate,
  type SilenceGateMode,
  type SilenceGatePolicy,
  type SpeechDetector,
} from "./offline-session.ts";
import type { SherpaBinding } from "./sherpa-api.ts";
import { createSherpaEngine, type AnySherpaEngine } from "./sherpa-factory.ts";
import {
  DEFAULT_SHERPA_MODEL,
  resolveSherpaModel,
  type SherpaModelId,
} from "./sherpa-models.ts";
import { loadSileroDetector, type VadBinding } from "./silero-vad.ts";

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
  gate: SilenceGateMode = "model",
  options?: { hotwords: Hotwords | undefined },
): Promise<AnySherpaEngine> {
  const descriptor = resolveSherpaModel(model);
  return createSherpaEngine({
    model,
    modelRoot: SHERPA_ROOT,
    binding: loadSherpaBinding(),
    hotwords: options ? options.hotwords : await loadHotwords(),
    silenceGate:
      descriptor.kind === "offline"
        ? await localSilenceGate(descriptor.silenceGate, gate)
        : undefined,
  });
}

let detector: Promise<SpeechDetector> | undefined;

/**
 * The silence gate a local engine runs under `mode`, or none. Every engine
 * shares one Silero detector, loaded on first use under SHERPA_ROOT.
 */
export async function localSilenceGate(
  policy: SilenceGatePolicy | undefined,
  mode: SilenceGateMode,
): Promise<SilenceGate | undefined> {
  if (!gatesSilence(policy, mode)) {
    return undefined;
  }
  detector ??= loadSileroDetector(
    loadSherpaBinding() as SherpaBinding & VadBinding,
    SHERPA_ROOT,
  ).catch((err: unknown) => {
    detector = undefined;
    throw err;
  });
  return { hasSpeech: await detector, logProbFloor: policy?.logProbFloor ?? null };
}
