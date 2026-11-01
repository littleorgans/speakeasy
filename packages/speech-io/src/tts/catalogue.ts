import { parseSpeechOutputConfig, type SpeechOutputConfig, type SpeechOutputRegistry } from "../provider-config.ts";
import { CMU_DICT, resolveTtsFrontendAssets } from "./frontend.ts";
import { resolveTtsModelPaths, LOCAL_TTS_MODELS, TTS_ROOT, verifyTtsFiles } from "./models.ts";

/** Only owner approved browser choices. The legacy sweep registry stays separate. */
export const LOCAL_OUTPUT_MODELS = [
  {
    id: "piper-kristin-medium",
    label: "Piper Kristin medium",
    provider: { kind: "local" },
    defaultVoice: "kristin",
    sampleRate: 22_050,
    streaming: "sentence",
    licence: "Voice released as public domain. Piper Plus MIT and CMU dictionary notices retained. International and talent rights unverified.",
    voices: [{ id: "kristin", label: "Kristin", languages: ["en-GB"], speakerId: 0 }],
  },
  {
    id: "kokoro-v1.0",
    label: "Kokoro v1.0 fp32",
    provider: { kind: "local" },
    defaultVoice: "af_heart",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights. Retain licence and model notices. Per voice training provenance and generated audio rights unverified.",
    voices: [
      { id: "af_heart", label: "Heart", languages: ["en-US"], speakerId: 3 },
      { id: "af_bella", label: "Bella", languages: ["en-US"], speakerId: 2 },
      { id: "am_michael", label: "Michael", languages: ["en-US"], speakerId: 16 },
    ],
  },
] as const;

export type LocalOutputModel = typeof LOCAL_OUTPUT_MODELS[number];
export type LocalOutputModelId = LocalOutputModel["id"];
export type LocalOutputReadiness = {
  state: "ready" | "assets-missing" | "failed";
  /** Safe category only. No paths, source locations or native error text. */
  reason: string | null;
};

export const LOCAL_OUTPUT_SPEED = { min: 0.5, max: 2 } as const;

/** Projection consumed by the one portable config parser. No second schema. */
export const LOCAL_OUTPUT_REGISTRY: SpeechOutputRegistry = LOCAL_OUTPUT_MODELS.map((model) => ({
  kind: model.provider.kind,
  model: model.id,
  voices: model.voices.flatMap((voice) => [voice.id, voice.speakerId]),
  speed: LOCAL_OUTPUT_SPEED,
}));

export const DEFAULT_LOCAL_OUTPUT: SpeechOutputConfig = Object.freeze({
  schema: 1, provider: Object.freeze({ kind: "local" as const }), model: "piper-kristin-medium", voice: "kristin", speed: 1,
});

export function resolveLocalOutput(config: SpeechOutputConfig = DEFAULT_LOCAL_OUTPUT) {
  config = parseSpeechOutputConfig(config, LOCAL_OUTPUT_REGISTRY);
  const model = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === config.model)!;
  const voice = model.voices.find((entry) => entry.id === config.voice || entry.speakerId === config.voice)!;
  return { model, voice, speed: config.speed };
}

/** Reads and hashes current bytes. Does not prepare, load, download or synthesize. */
export async function getLocalOutputReadiness(
  modelId: LocalOutputModelId,
  root: string = TTS_ROOT,
): Promise<LocalOutputReadiness> {
  const paths = resolveTtsModelPaths(LOCAL_TTS_MODELS[modelId], root);
  try {
    await verifyTtsFiles(paths.dir, paths.files!, paths.attribution);
    const frontend = resolveTtsFrontendAssets(root);
    await verifyTtsFiles(frontend.dir, [{ name: "cmudict_data.json", sha256: CMU_DICT.sha256, bytes: CMU_DICT.bytes }]);
    return { state: "ready", reason: null };
  } catch (error) {
    const missing = error instanceof Error && /missing|wrong size/.test(error.message);
    return {
      state: missing ? "assets-missing" : "failed",
      reason: missing ? "Required local assets or notices are missing or incomplete" : "Local asset verification failed",
    };
  }
}

export async function getLocalOutputCatalogue(root: string = TTS_ROOT) {
  const models = await Promise.all(LOCAL_OUTPUT_MODELS.map(async (model) => ({
    ...model, readiness: await getLocalOutputReadiness(model.id, root),
  })));
  return { default: { ...DEFAULT_LOCAL_OUTPUT, provider: { ...DEFAULT_LOCAL_OUTPUT.provider } }, models };
}
