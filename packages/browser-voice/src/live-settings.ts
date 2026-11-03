import type { LiveSettings, SetupId } from "./setup-protocol.ts";
import { getSetup } from "./setup-registry.ts";

/**
 * Live words: the engines that truly stream, the settings each exposes, and
 * each setup's default pairing. Every live model pairs with every final
 * model: each runs in its own worker, and the final pass waits for the live
 * stream to settle. Defaults are today's values; the worker applies exactly
 * these and refuses others.
 */
export type LiveOption = Readonly<{ value: number; label: string; phrase: string }>;
export type LiveSetting = Readonly<{ id: string; label: string; note: string; default: number; options: readonly LiveOption[] }>;
export type LiveStream = Readonly<{ setupId: SetupId; settings: readonly LiveSetting[] }>;
export type { LiveSettings };

const seconds = (ms: number) => ms < 1_000 ? `${ms} ms` : `${Number((ms / 1_000).toFixed(2))} s`;

// parakeet-mlx normalizes each chunk's mel on its own: chunks under 800 ms
// of near silence wiped earlier words after a pause, so none are offered.
// Its (64, 64) context, the last 5.12 s rewritable, is the only one that
// held: 2.56 s and 10.24 s both garbled settled words.
const PARAKEET: readonly LiveSetting[] = [
  { id: "updateMs", label: "Updates every", default: 800,
    note: "Longer means fewer, larger updates.",
    options: [800, 1_200, 1_600].map((ms) => ({ value: ms, label: seconds(ms),
      phrase: `updates every ${seconds(ms)}, the last 5.12 s may change` })) },
];
// transcribe.cpp's Unified training menu, as chunk plus lookahead; Hex ships 2.08 s.
const HEX: readonly LiveSetting[] = [
  { id: "lookaheadMs", label: "Lookahead", default: 2_080,
    note: "Shorter shows words sooner, slightly less accurately.",
    options: [2_080, 1_120, 480].map((ms) => ({ value: ms, label: seconds(ms),
      phrase: `${seconds(ms)} lookahead, written words never change` })) },
];

// mlx-audio's Nemotron delay menu; the research measured 560 ms, the final keeps it.
const NEMOTRON: readonly LiveSetting[] = [
  { id: "delayMs", label: "Delay", default: 560,
    note: "Shorter shows words sooner; longer hears more before each word.",
    options: [1_120, 560, 160].map((ms) => ({ value: ms, label: seconds(ms),
      phrase: `${seconds(ms)} delay, written words never change` })) },
];

// SpeechTranscriber's reporting options; the research measured both, volatile results alone first.
const APPLE: readonly LiveSetting[] = [
  { id: "fastResults", label: "Fast results", default: 0,
    note: "On shows words sooner and revises them more.",
    options: [{ value: 0, label: "Off", phrase: "volatile results, the newest words may change" },
      { value: 1, label: "On", phrase: "fast volatile results, the newest words may change" }] },
];

export const LIVE_STREAMS: readonly LiveStream[] = Object.freeze([
  { setupId: "hex/unified", settings: HEX },
  { setupId: "nemotron/streaming", settings: NEMOTRON },
  { setupId: "apple/speech", settings: APPLE },
  { setupId: "mlx-parakeet/v3-greedy", settings: PARAKEET },
  { setupId: "mlx-parakeet/ultra-greedy", settings: PARAKEET },
]);

export function liveStream(setupId: unknown): LiveStream | undefined {
  return LIVE_STREAMS.find((stream) => stream.setupId === setupId);
}

/** Today's pairing: a streaming setup hears itself, a cleanup setup its base, the rest have none. */
export function defaultLiveSetup(setupId: SetupId): SetupId | null {
  const setup = getSetup(setupId);
  if (setup.card.transcript !== "partials") return null;
  return liveStream(setup.baseSetup ?? setup.id)?.setupId ?? null;
}

/** Every setting of the stream, defaults filled in; unknown names or values are refused. */
export function resolveLiveSettings(stream: LiveStream, value: unknown = {}): LiveSettings {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("live settings must be an object");
  const given = value as Record<string, unknown>;
  for (const name of Object.keys(given)) {
    if (!stream.settings.some((setting) => setting.id === name)) throw new Error(`unknown live setting ${name}`);
  }
  return Object.freeze(Object.fromEntries(stream.settings.map((setting) => {
    const chosen = given[setting.id] ?? setting.default;
    if (!setting.options.some((option) => option.value === chosen)) throw new Error(`unsupported ${setting.id}`);
    return [setting.id, chosen as number];
  })));
}
