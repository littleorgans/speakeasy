import { ProviderError } from "@speakeasy/speech-io/providers";
import type { ApiProviderKind } from "@speakeasy/speech-io/providers";

/** Canonical names only. Unresearched credential aliases do not enable a provider. */
export const PROVIDER_CREDENTIAL_NAMES = {
  cartesia: "CARTESIA_API_KEY",
  groq: "GROQ_API_KEY",
  mistral: "MISTRAL_API_KEY",
  openai: "OPENAI_API_KEY",
  gemini: "GEMINI_API_KEY",
  elevenlabs: "ELEVENLABS_API_KEY",
} as const satisfies Record<ApiProviderKind, string>;

export type ProviderPresence = Record<ApiProviderKind, "configured" | "unconfigured">;
export type EffectiveEnvironment = Readonly<Record<string, string | undefined>>;

/**
 * Construct after the existing server bootstrap has loaded its eligible .env.
 * Both directions share this resolver. It never reads files or probes providers.
 */
export function createProviderCredentials(
  effectiveEnvironment: () => EffectiveEnvironment = () => process.env,
) {
  function read(kind: ApiProviderKind): string | undefined {
    try {
      const value = effectiveEnvironment()[PROVIDER_CREDENTIAL_NAMES[kind]];
      return typeof value === "string" && value.trim().length > 0 ? value : undefined;
    } catch {
      throw new ProviderError("PROVIDER_UNAVAILABLE");
    }
  }

  function configured(kind: ApiProviderKind): boolean {
    return read(kind) !== undefined;
  }

  function snapshot(): ProviderPresence {
    return Object.fromEntries(
      (Object.keys(PROVIDER_CREDENTIAL_NAMES) as ApiProviderKind[])
        .map((kind) => [kind, configured(kind) ? "configured" : "unconfigured"]),
    ) as ProviderPresence;
  }

  /** Retain the function, never its return value, in a long lived adapter. */
  function getter(kind: ApiProviderKind): () => string {
    return () => {
      const value = read(kind);
      if (value === undefined) throw new ProviderError("PROVIDER_UNAVAILABLE");
      return value;
    };
  }

  return { configured, snapshot, getter };
}

export type ProviderCredentials = ReturnType<typeof createProviderCredentials>;
