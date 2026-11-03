import { parseSpeechOutputConfig, type SpeechOutputConfig, type SpeechOutputRegistry } from "../provider-config.ts";
import type { VoiceSetting } from "../settings.ts";
import { CMU_DICT, resolveTtsFrontendAssets } from "./frontend.ts";
import { join } from "node:path";
import { LOCAL_TTS_MODELS, payloadDir, TTS_ROOT, verifyTtsFiles } from "./models.ts";
import { QWEN_COPY_SETTINGS, QWEN_COPY_VOICES, QWEN_CUSTOM_17B_SETTINGS, QWEN_CUSTOM_SETTINGS, QWEN_SPEAKERS } from "./qwen-voices.ts";

const QWEN_COPY_LICENCE = "Apache 2.0 weights and qwen-tts code (Qwen); retain the licence and notices. " +
  "Copy only your own voice or the synthetic references.";

/** Only owner approved browser choices. The legacy sweep registry stays separate. */
export const LOCAL_OUTPUT_MODELS = [
  {
    id: "piper-kristin-medium",
    label: "Piper Kristin medium",
    provider: { kind: "local" },
    runtime: "sherpa-onnx VITS, CPU",
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
    runtime: "sherpa-onnx Kokoro, CPU",
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
  {
    id: "kokoro-mlx-bf16",
    label: "Kokoro v1.0 MLX bf16",
    provider: { kind: "local" },
    runtime: "mlx-audio 0.6.3 on MLX, Metal GPU",
    defaultVoice: "af_heart",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights and code. misaki's espeak-ng fallback is GPL 3.0. Per voice training provenance and generated audio rights unverified.",
    /** One short line on the card wherever a licence is not plainly permissive. */
    terms: "misaki's espeak-ng fallback is GPL 3.0",
    voices: [
      { id: "af_heart", label: "Heart", languages: ["en-US"], speakerId: 0 },
      { id: "am_michael", label: "Michael", languages: ["en-US"], speakerId: 1 },
    ],
  },
  {
    id: "pocket-18m",
    label: "Pocket TTS 18M",
    provider: { kind: "local" },
    runtime: "pocket-tts 4.0.1 on PyTorch, CPU",
    defaultVoice: "bill_boerst",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "CC BY 4.0 weights: attribution required; the model card prohibits voice impersonation without consent. MIT code. Voice bill_boerst from Kyutai's CC0 collection.",
    terms: "CC BY 4.0, credit required; no voice impersonation without consent",
    voices: [{ id: "bill_boerst", label: "Bill", languages: ["en"], speakerId: 0 }],
  },
  {
    id: "piper-kristin-split",
    label: "Piper Kristin split decoder",
    provider: { kind: "local" },
    runtime: "libpiper (piper1-gpl 10880972), onnxruntime 1.22.0, CPU",
    defaultVoice: "kristin",
    sampleRate: 22_050,
    streaming: "sentence",
    licence: "Voice released as public domain. libpiper, and the eSpeak NG it links statically, are GPL 3.0: a runtime licence boundary sherpa does not have, pending packaging review. onnxruntime MIT.",
    terms: "GPL 3.0 runtime (libpiper, eSpeak NG)",
    voices: [{ id: "kristin", label: "Kristin", languages: ["en-GB"], speakerId: 0 }],
  },
  {
    id: "kokoro-coreml",
    label: "Kokoro v1.0 Core ML",
    provider: { kind: "local" },
    runtime: "FluidAudio e396cf78 on Core ML, CPU and Neural Engine",
    defaultVoice: "af_heart",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights and FluidAudio code; its prebuilt text normalizer is Apache 2.0 per its manifest, unverified at source. Per voice training provenance and generated audio rights unverified.",
    voices: [
      { id: "af_heart", label: "Heart", languages: ["en-US"], speakerId: 0 },
      { id: "am_michael", label: "Michael", languages: ["en-US"], speakerId: 1 },
    ],
  },
  {
    id: "phonon-31m",
    label: "Phonon 31M",
    provider: { kind: "local" },
    runtime: "ptts 0.6.0 (Rust), CPU, f32",
    defaultVoice: "default",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights, tokenizer and voice; ptts code MIT or Apache 2.0. Builds on Kyutai's Pocket TTS and Mimi (CC BY 4.0); codec lineage and Kyutai credit unverified. Generated audio rights unverified.",
    voices: [{ id: "default", label: "Default", languages: ["en"], speakerId: 0 }],
  },
  {
    id: "qwen3-tts-0.6b",
    label: "Qwen3 TTS 0.6B",
    provider: { kind: "local" },
    runtime: "speech.cpp 0.10.0, Metal GPU, Q8_0",
    defaultVoice: "ryan",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights (Qwen); speech.cpp MIT. Retain the licence and notices. Speaker consent and generated audio rights unverified.",
    voices: QWEN_SPEAKERS,
    settings: QWEN_CUSTOM_SETTINGS,
  },
  {
    id: "qwen3-tts-1.7b",
    label: "Qwen3 TTS 1.7B",
    provider: { kind: "local" },
    runtime: "speech.cpp 0.10.0, Metal GPU, Q8_0",
    defaultVoice: "ryan",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights (Qwen); speech.cpp MIT. Retain the licence and notices. Speaker consent and generated audio rights unverified.",
    voices: QWEN_SPEAKERS,
    settings: QWEN_CUSTOM_17B_SETTINGS,
  },
  {
    id: "qwen3-tts-0.6b-base",
    label: "Qwen3 TTS 0.6B voice copy",
    provider: { kind: "local" },
    runtime: "qwen-tts 0.1.1 on PyTorch, Metal GPU (MPS), float32",
    defaultVoice: "ref-10s",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: QWEN_COPY_LICENCE,
    voices: QWEN_COPY_VOICES,
    settings: QWEN_COPY_SETTINGS,
  },
  {
    id: "qwen3-tts-1.7b-base",
    label: "Qwen3 TTS 1.7B voice copy",
    provider: { kind: "local" },
    runtime: "qwen-tts 0.1.1 on PyTorch, Metal GPU (MPS), float32",
    defaultVoice: "ref-10s",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: QWEN_COPY_LICENCE,
    voices: QWEN_COPY_VOICES,
    settings: QWEN_COPY_SETTINGS,
  },
  {
    id: "kokoro-ane-heart",
    label: "Kokoro ANE hybrid",
    provider: { kind: "local" },
    runtime: "kokoro-coreml b23caa8 on Core ML, Neural Engine and GPU",
    defaultVoice: "af_heart",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights (hexgrad/Kokoro-82M), kokoro-coreml SDK and MisakiSwift. The hosted bundle's own card licence and the full transitive review are unchecked. Generated audio rights unverified.",
    voices: [{ id: "af_heart", label: "Heart", languages: ["en-US"], speakerId: 0 }],
  },
  {
    id: "kokoro-ane-folded-heart",
    label: "Kokoro ANE folded",
    provider: { kind: "local" },
    runtime: "kokoro-coreml b23caa8 on Core ML, Neural Engine only",
    defaultVoice: "af_heart",
    sampleRate: 24_000,
    streaming: "sentence",
    licence: "Apache 2.0 weights (hexgrad/Kokoro-82M), kokoro-coreml SDK and MisakiSwift. The hosted bundle's own card licence and the full transitive review are unchecked. Generated audio rights unverified.",
    voices: [{ id: "af_heart", label: "Heart", languages: ["en-US"], speakerId: 0 }],
  },
] as const;

export type LocalOutputModel = typeof LOCAL_OUTPUT_MODELS[number];
export type LocalOutputModelId = LocalOutputModel["id"];
export type LocalOutputReadiness = {
  state: "ready" | "assets-missing" | "runtime-missing" | "failed";
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

/** Pinned payload bytes, for the plain text model card. */
export function localOutputSizeBytes(modelId: LocalOutputModelId): number {
  return LOCAL_TTS_MODELS[modelId].files.reduce((sum, file) => sum + file.bytes, 0);
}

/** The voice settings a model declares; none for a model without per reading controls. */
export function localOutputSettings(modelId: LocalOutputModelId): readonly VoiceSetting[] {
  const model: LocalOutputModel = LOCAL_OUTPUT_MODELS.find((entry) => entry.id === modelId)!;
  return "settings" in model ? model.settings : [];
}

/** True for a voice the host runs in its own worker runtime, never sherpa. */
export function isWorkerOutput(modelId: LocalOutputModelId): boolean {
  return LOCAL_TTS_MODELS[modelId].family === "worker";
}

/**
 * Reads and hashes current bytes. Does not prepare, load, download or synthesize.
 * A worker voice is ready only when the host has configured its runtime.
 */
export async function getLocalOutputReadiness(
  modelId: LocalOutputModelId,
  root: string = TTS_ROOT,
  workerConfigured: (modelId: LocalOutputModelId) => boolean = () => false,
): Promise<LocalOutputReadiness> {
  const model = LOCAL_TTS_MODELS[modelId];
  if (model.family === "worker" && !workerConfigured(modelId)) {
    return { state: "runtime-missing", reason: "Voice runtime is not configured" };
  }
  try {
    await verifyTtsFiles(payloadDir(model, root), model.files, model.attribution);
    // A phoneme runtime also needs the sherpa voice whose frontend the host borrows.
    const frontendVoice = model.family === "worker" && "phonemesFrom" in model ? LOCAL_TTS_MODELS[model.phonemesFrom] : undefined;
    if (frontendVoice) await verifyTtsFiles(join(root, frontendVoice.name), frontendVoice.files, frontendVoice.attribution);
    if (model.family === "worker" && !frontendVoice) return { state: "ready", reason: null };
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

export async function getLocalOutputCatalogue(
  root: string = TTS_ROOT,
  workerConfigured?: (modelId: LocalOutputModelId) => boolean,
) {
  const models = await Promise.all(LOCAL_OUTPUT_MODELS.map(async (model) => ({
    ...model, readiness: await getLocalOutputReadiness(model.id, root, workerConfigured),
  })));
  return { default: { ...DEFAULT_LOCAL_OUTPUT, provider: { ...DEFAULT_LOCAL_OUTPUT.provider } }, models };
}
