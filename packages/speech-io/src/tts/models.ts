import { stat } from "node:fs/promises";
import { join } from "node:path";
import { ensureAsset, sha256File } from "../engines/assets.ts";

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
 * Approved browser payloads are separate below, preserving CLI sweep defaults.
 * Text to phonemes lives in frontend.ts, shared by every descriptor.
 */

const TTS_MODELS_BASE =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

/** On-disk root for downloaded + extracted TTS models. */
export const TTS_ROOT = join(process.cwd(), "models", "tts");

export type TtsModelFamily = "vits" | "kokoro";

export type TtsModelFile = { name: string; sha256: string; bytes: number };

export type TtsModel = {
  /** Release tarball basename; doubles as the extracted directory name. */
  name: string;
  /** Legacy release archives are pinned as archives. Browser bundles pin payloads. */
  sha256?: string;
  bytes?: number;
  files?: readonly TtsModelFile[];
  attribution?: readonly string[];
  lang?: string;
  sampleRate?: number;
  numSpeakers?: number;
  /** Human note: architecture / voice count / size. */
  note: string;
  family: TtsModelFamily;
  /** Acoustic model onnx filename inside the extracted directory. */
  model: string;
  tokens: string;
  /** Speaker embedding bank; kokoro only. */
  voices?: string;
};

/** Payload pins from the approved local research records. Explicit provisioning only. */
export const LOCAL_TTS_MODELS = {
  "piper-kristin-medium": {
    name: "vits-piper-en_US-kristin-medium",
    note: "Piper Kristin medium, pinned fp32, single voice, 22050 Hz",
    family: "vits",
    model: "en_US-kristin-medium.onnx",
    tokens: "tokens.txt",
    lang: "en",
    sampleRate: 22_050,
    numSpeakers: 1,
    files: [
      { name: "en_US-kristin-medium.onnx", bytes: 63_531_507, sha256: "f9a5c375c70ddc33ccdb3969c76a4d7cc5255ac08119a312b0034a1d07041be6" },
      { name: "tokens.txt", bytes: 940, sha256: "ef3a7e4a8d1af0c9d4dc45aaae1a6242ebe24a7ed6f3d025a49eb29682784c6d" },
      { name: "en_US-kristin-medium.onnx.json", bytes: 4_968, sha256: "5681426d4aead22195de70531eeeeddb46493cfaffc5764b2ea3db73428b651c" },
    ],
    attribution: ["MODEL_CARD"],
  },
  "kokoro-v1.0": {
    name: "kokoro-multi-lang-v1_0",
    note: "Kokoro v1.0 fp32, exact 54 speaker bundle, 24000 Hz",
    family: "kokoro",
    model: "model.onnx",
    tokens: "tokens.txt",
    voices: "voices.bin",
    lang: "en-us",
    sampleRate: 24_000,
    numSpeakers: 54,
    files: [
      { name: "model.onnx", bytes: 325_560_556, sha256: "b40f62b166ac8164b0627ef48a0b358eda0985e272fb03ef5252e7206305da11" },
      { name: "voices.bin", bytes: 28_200_960, sha256: "1c5a5b983d3d50d8586d437a51f3faa2da7919ce76a013c081e65671a3447c29" },
      { name: "tokens.txt", bytes: 687, sha256: "6ebb6bb288f20f3ae8d004d3c2ca27697da27c037d75e81a60e2a6a663f95425" },
      { name: "LICENSE", bytes: 11_358, sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" },
    ],
    attribution: ["LICENSE", "README.md"],
  },
} as const satisfies Record<string, TtsModel>;

/** Legacy CLI sweep registry, including its original iteration order. */
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
export type LocalTtsModelId = keyof typeof LOCAL_TTS_MODELS;
export type SynthModelId = TtsModelId | LocalTtsModelId;

export function getTtsModel(id: SynthModelId): TtsModel {
  return Object.hasOwn(LOCAL_TTS_MODELS, id)
    ? LOCAL_TTS_MODELS[id as LocalTtsModelId]
    : TTS_MODELS[id as TtsModelId];
}

export type TtsModelPaths = {
  dir: string;
  archive: string;
  url: string;
  sha256?: string;
  files?: readonly TtsModelFile[];
  model: string;
  tokens: string;
  voices?: string;
  /** Archive files synthesis reads; install skips the rest, espeak-ng-data included. */
  members: readonly string[];
  /** The model's licence and attribution files, kept when its archive has them. */
  attribution: readonly string[];
};

/** Licence and attribution files the release archives carry beside the model. */
const ATTRIBUTION_FILES = ["LICENSE", "MODEL_CARD", "README.md"] as const;

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
    files: model.files,
    model: join(dir, model.model),
    tokens: join(dir, model.tokens),
    voices: model.voices ? join(dir, model.voices) : undefined,
    members: model.files?.map((file) => file.name) ?? [model.model, model.tokens, ...(model.voices ? [model.voices] : [])],
    attribution: model.attribution ?? ATTRIBUTION_FILES,
  };
}

/**
 * Verify provisioned browser payloads, or download and extract a legacy tarball.
 * Download the legacy tarball and extract its members and attribution unless
 * already present. An install made before members were listed keeps its
 * files: the completion marker is the archive digest either way.
 */
export async function ensureTtsModel(paths: TtsModelPaths): Promise<void> {
  if (paths.files) {
    await verifyTtsFiles(paths.dir, paths.files, paths.attribution);
    return;
  }
  if (!paths.sha256) throw new Error("TTS archive has no digest");
  await ensureAsset({
    url: paths.url,
    sha256: paths.sha256,
    archive: paths.archive,
    dir: paths.dir,
    members: paths.members,
    optionalMembers: paths.attribution,
  });
}

/** Read only verification. Completion markers never substitute for payload hashes. */
export async function verifyTtsFiles(
  dir: string,
  files: readonly TtsModelFile[],
  attribution: readonly string[] = [],
): Promise<void> {
  for (const file of files) {
    const path = join(dir, file.name);
    const info = await stat(path).catch(() => undefined);
    if (!info?.isFile() || info.size !== file.bytes) {
      throw new Error(`TTS asset missing or wrong size: ${file.name}`);
    }
    if (await sha256File(path) !== file.sha256) {
      throw new Error(`TTS asset digest mismatch: ${file.name}`);
    }
  }
  for (const name of attribution) {
    const info = await stat(join(dir, name)).catch(() => undefined);
    if (!info?.isFile() || info.size === 0) throw new Error(`TTS notice missing: ${name}`);
  }
}

export function isTtsModelId(value: string): value is TtsModelId {
  return Object.hasOwn(TTS_MODELS, value);
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
