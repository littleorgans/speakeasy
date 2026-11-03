import type { LiveSettings, SetupId } from "./setup-protocol.ts";
import { resolveLiveSettings, type LiveSetting } from "./live-settings.ts";

/**
 * Final settings: what a setup's whole take pass exposes, in the live
 * settings shape. Defaults are today's values; the comparison sends every
 * value to the worker, which applies exactly these and refuses others.
 */
export type FinalSetup = Readonly<{ setupId: SetupId; settings: readonly LiveSetting[] }>;

// Research ct-pbias: prefix sequence bias +1.25 and +2.5 toward five names
// (runtime/hex/cohere-sequence-bias.patch). Both recovered missed names and
// both damaged other words, so Off stays the default.
const COHERE: readonly LiveSetting[] = [
  { id: "namesBias", label: "Names", default: 0,
    note: "Biases Cohere toward Grok, Claude, Codex, GPT, TM; can turn similar words into these names",
    options: [{ value: 0, label: "Off", phrase: "no names bias" },
      { value: 1.25, label: "Gentle", phrase: "names bias 1.25" },
      { value: 2.5, label: "Strong", phrase: "names bias 2.5" }] },
];

export const FINAL_SETUPS: readonly FinalSetup[] = Object.freeze([
  { setupId: "cohere/transcribe", settings: COHERE },
]);

export function finalSetup(setupId: unknown): FinalSetup | undefined {
  return FINAL_SETUPS.find((setup) => setup.setupId === setupId);
}

/** Final settings by setup id. */
export type FinalSettings = Readonly<Partial<Record<SetupId, LiveSettings>>>;

/**
 * Every selected setup that has final settings, each with every setting and
 * defaults filled in; settings for another setup, unknown names or values
 * that were not offered are refused.
 */
export function resolveFinalSettings(setupIds: readonly SetupId[], value: unknown = {}): FinalSettings {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("final settings must be an object");
  const given = value as Record<string, unknown>;
  for (const id of Object.keys(given)) {
    if (!setupIds.includes(id as SetupId) || !finalSetup(id)) throw new Error(`no final settings for ${id}`);
  }
  return Object.freeze(Object.fromEntries(setupIds.flatMap((id) => {
    const setup = finalSetup(id);
    return setup ? [[id, resolveLiveSettings(setup, given[id])]] : [];
  })));
}
