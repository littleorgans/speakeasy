import { join } from "node:path";
import { ensureAsset } from "../engines/assets.ts";

/**
 * Registry of offline TTS models for the head-to-head sweep.
 *
 * Mirrors src/engines/sherpa-models.ts: each descriptor names the release
 * tarball plus the per-family file layout inside it. Same release host as the
 * STT models, different release tag (tts-models). Swapping models is
 * config-only: add a descriptor here, select it with `--model <id>`.
 */

const TTS_MODELS_BASE =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

/** On-disk root for downloaded + extracted TTS models. */
export const TTS_ROOT = join(process.cwd(), "models", "tts");

export type TtsModelFamily = "vits" | "kokoro";

export type TtsModel = {
  /** Release tarball basename; doubles as the extracted directory name. */
  name: string;
  /** Human note: architecture / voice count / size. */
  note: string;
  family: TtsModelFamily;
  /** Acoustic model onnx filename inside the extracted directory. */
  model: string;
  tokens: string;
  /** espeak-ng phoneme data directory shipped inside the tarball. */
  dataDir: string;
  /** Speaker embedding bank; kokoro only. */
  voices?: string;
};

export const TTS_MODELS = {
  "piper-amy": {
    name: "vits-piper-en_US-amy-low",
    note: "Piper VITS en_US amy low, single voice, 16kHz, 64MB tarball",
    family: "vits",
    model: "en_US-amy-low.onnx",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
  "piper-ryan-high": {
    name: "vits-piper-en_US-ryan-high",
    note: "Piper VITS en_US ryan high, single voice, 22kHz, 110MB tarball",
    family: "vits",
    model: "en_US-ryan-high.onnx",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
  "piper-lessac-high": {
    name: "vits-piper-en_US-lessac-high",
    note: "Piper VITS en_US lessac high, single voice, 22kHz, 110MB tarball",
    family: "vits",
    model: "en_US-lessac-high.onnx",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
  "piper-libritts": {
    name: "vits-piper-en_US-libritts_r-medium",
    note: "Piper VITS en_US libritts_r medium, 904 voices, 22kHz, 78MB tarball",
    family: "vits",
    model: "en_US-libritts_r-medium.onnx",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
  "kokoro-v0.19": {
    name: "kokoro-en-v0_19",
    note: "Kokoro 82M StyleTTS2-based, multi-voice, 24kHz, 305MB tarball",
    family: "kokoro",
    model: "model.onnx",
    voices: "voices.bin",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
  "kokoro-int8": {
    name: "kokoro-int8-en-v0_19",
    note: "Kokoro v0.19 int8: same 11 voices as kokoro-v0.19, 24kHz, 98MB tarball",
    family: "kokoro",
    model: "model.int8.onnx",
    voices: "voices.bin",
    tokens: "tokens.txt",
    dataDir: "espeak-ng-data",
  },
} as const satisfies Record<string, TtsModel>;

export type TtsModelId = keyof typeof TTS_MODELS;

export type TtsModelPaths = {
  dir: string;
  archive: string;
  url: string;
  model: string;
  tokens: string;
  dataDir: string;
  voices?: string;
};

/** Resolve a descriptor to absolute on-disk paths and the download URL. */
export function resolveTtsModelPaths(model: TtsModel): TtsModelPaths {
  const dir = join(TTS_ROOT, model.name);
  return {
    dir,
    archive: join(TTS_ROOT, `${model.name}.tar.bz2`),
    url: `${TTS_MODELS_BASE}/${model.name}.tar.bz2`,
    model: join(dir, model.model),
    tokens: join(dir, model.tokens),
    dataDir: join(dir, model.dataDir),
    voices: model.voices ? join(dir, model.voices) : undefined,
  };
}

/** Download + extract the model tarball unless already present. */
export async function ensureTtsModel(paths: TtsModelPaths): Promise<void> {
  await ensureAsset({
    url: paths.url,
    archive: paths.archive,
    extractTo: TTS_ROOT,
    sentinel: paths.model,
  });
}

export function isTtsModelId(value: string): value is TtsModelId {
  return value in TTS_MODELS;
}

/** Parse a CLI `--model` value into a typed id, or throw with the valid set. */
export function parseTtsModelId(value: string): TtsModelId {
  if (isTtsModelId(value)) {
    return value;
  }
  throw new Error(
    `--model must be one of ${Object.keys(TTS_MODELS).join(", ")}; received ${value}`,
  );
}
