import { readFileSync } from "node:fs";
import { isSetupId, SETUP_IDS, sha256, type SetupId } from "./setup-protocol.ts";

type Json = null | boolean | number | string | readonly Json[] | { readonly [key: string]: Json };
export type SetupDefinition = Readonly<{
  id: SetupId; label: string; model: string; revision: string;
  runtime: Readonly<Record<string, Json>>; controls: Readonly<Record<string, Json>>;
  operation: "transcribe" | "cleanup"; maxSamples: number; baseSetup?: SetupId;
  [key: string]: Json | undefined;
}>;
export type SetupManifest = Readonly<{
  schema: 1; defaultSetupId: SetupId; idleReleaseMs: number; setups: readonly SetupDefinition[];
}>;

// Pins every model, runtime, control, dependency and portable artifact name.
export const SETUP_MANIFEST_SHA256 = "86f83f550b1c1f00bf15262a04497ccfcddda696bf56743be8f91da32c3276b5";

export function assertSetupManifest(value: unknown): asserts value is SetupManifest {
  if (sha256(JSON.stringify(value) ?? "") !== SETUP_MANIFEST_SHA256) {
    throw new Error("local setup manifest identity/controls mismatch");
  }
}

function freeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

const manifest: unknown = JSON.parse(readFileSync(new URL("../runtime/setups.json", import.meta.url), "utf8"));
assertSetupManifest(manifest);
export const SETUP_MANIFEST = freeze(manifest);
export const SETUPS = SETUP_MANIFEST.setups;

export function getSetup(id: unknown): SetupDefinition {
  if (!isSetupId(id)) throw new Error("unknown local setup id");
  return SETUPS.find((setup) => setup.id === id)!;
}

/** Accept only immutable ids, never client supplied model or decode overrides. */
export function parseSetupSelection(value: unknown = [SETUP_MANIFEST.defaultSetupId]): readonly SetupId[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > SETUP_IDS.length
    || value.some((id) => !isSetupId(id)) || new Set(value).size !== value.length) {
    throw new Error("select unique local setup ids");
  }
  return Object.freeze(SETUP_IDS.filter((id) => value.includes(id)));
}
