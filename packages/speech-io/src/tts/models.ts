import { join } from "node:path";
import { ensureAsset } from "../engines/assets.ts";

/**
 * Registry of offline TTS models for the head-to-head sweep.
 *
 * Mirrors src/engines/sherpa-models.ts: each descriptor names the release
 * tarball plus the per-family file layout inside it. Same release host as the
 * STT models, different release tag (tts-models). Swapping models is
 * config-only: add a descriptor here, select it with `--model <id>`.
 *
 * Digests are the release assets' published GitHub sha256, read 2026-11-01.
 * Releases are mutable, so an upstream change fails closed in `ensureAsset`.
 * Text to phonemes lives in frontend.ts, shared by every descriptor.
 */

const TTS_MODELS_BASE =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

/** On-disk root for downloaded + extracted TTS models. */
export const TTS_ROOT = join(process.cwd(), "models", "tts");

export type TtsModelFamily = "vits" | "kokoro";

export type TtsModel = {
  /** Release tarball basename; doubles as the extracted directory name. */
  name: string;
  /** Release tarball sha256 (hex). */
  sha256: string;
  bytes: number;
  /** Human note: architecture / voice count / size. */
  note: string;
  family: TtsModelFamily;
  /** Acoustic model onnx filename inside the extracted directory. */
  model: string;
  tokens: string;
  /** Speaker embedding bank; kokoro only. */
  voices?: string;
};

export const TTS_MODELS = {
  "piper-amy": {
    name: "vits-piper-en_US-amy-low",
    sha256: "c70f5284a09a7fd4ed203b39b2ff51cac1432b422b852eb647b481dade3cf639", // GitHub digest
    bytes: 67_095_344,
    note: "Piper VITS en_US amy low, single voice, 16kHz, 64MB tarball",
    family: "vits",
    model: "en_US-amy-low.onnx",
    tokens: "tokens.txt",
  },
  "piper-ryan-high": {
    name: "vits-piper-en_US-ryan-high",
    sha256: "6a71edf4d308b9cb2eaeadc8d1f3c6bf96120ecb7fe52c29a2b6e139c59760ed", // GitHub digest
    bytes: 115_630_708,
    note: "Piper VITS en_US ryan high, single voice, 22kHz, 110MB tarball",
    family: "vits",
    model: "en_US-ryan-high.onnx",
    tokens: "tokens.txt",
  },
  "piper-lessac-high": {
    name: "vits-piper-en_US-lessac-high",
    sha256: "8619d204c7005866fe4f420181dfa79622af6a6222389f0b0818d2af31e0db0e", // GitHub digest
    bytes: 115_545_841,
    note: "Piper VITS en_US lessac high, single voice, 22kHz, 110MB tarball",
    family: "vits",
    model: "en_US-lessac-high.onnx",
    tokens: "tokens.txt",
  },
  "piper-libritts": {
    name: "vits-piper-en_US-libritts_r-medium",
    sha256: "10dc268f3e371696d721486123e2705a9fc1faa113491979fde4d88dba1f1b1c", // GitHub digest
    bytes: 82_038_311,
    note: "Piper VITS en_US libritts_r medium, 904 voices, 22kHz, 78MB tarball",
    family: "vits",
    model: "en_US-libritts_r-medium.onnx",
    tokens: "tokens.txt",
  },
  "kokoro-v0.19": {
    name: "kokoro-en-v0_19",
    sha256: "912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7", // GitHub digest
    bytes: 319_625_534,
    note: "Kokoro 82M StyleTTS2-based, multi-voice, 24kHz, 305MB tarball",
    family: "kokoro",
    model: "model.onnx",
    voices: "voices.bin",
    tokens: "tokens.txt",
  },
  "kokoro-int8": {
    name: "kokoro-int8-en-v0_19",
    sha256: "c9f0dd393615805b0bab050c340834d5e684e732aec91c0e860cd30e982c08bd", // GitHub digest
    bytes: 103_248_205,
    note: "Kokoro v0.19 int8: same 11 voices as kokoro-v0.19, 24kHz, 98MB tarball",
    family: "kokoro",
    model: "model.int8.onnx",
    voices: "voices.bin",
    tokens: "tokens.txt",
  },
} as const satisfies Record<string, TtsModel>;

export type TtsModelId = keyof typeof TTS_MODELS;

export type TtsModelPaths = {
  dir: string;
  archive: string;
  url: string;
  sha256: string;
  model: string;
  tokens: string;
  voices?: string;
  /** Archive files synthesis reads; install skips the rest, espeak-ng-data included. */
  members: readonly string[];
};

/** Resolve a descriptor to absolute on-disk paths and the download URL. */
export function resolveTtsModelPaths(
  model: TtsModel,
  root: string = TTS_ROOT,
): TtsModelPaths {
  const dir = join(root, model.name);
  return {
    dir,
    archive: join(root, `${model.name}.tar.bz2`),
    url: `${TTS_MODELS_BASE}/${model.name}.tar.bz2`,
    sha256: model.sha256,
    model: join(dir, model.model),
    tokens: join(dir, model.tokens),
    voices: model.voices ? join(dir, model.voices) : undefined,
    members: [model.model, model.tokens, ...(model.voices ? [model.voices] : [])],
  };
}

/**
 * Download the model tarball and extract its members unless already present.
 * An install made before members were listed keeps its files: the completion
 * marker is the archive digest either way.
 */
export async function ensureTtsModel(paths: TtsModelPaths): Promise<void> {
  await ensureAsset({
    url: paths.url,
    sha256: paths.sha256,
    archive: paths.archive,
    dir: paths.dir,
    members: paths.members,
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
