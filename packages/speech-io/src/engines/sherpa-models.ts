import { join } from "node:path";
import type { AssetFile, AssetSpec, FileSetSpec } from "./assets.ts";

/**
 * Registry of sherpa-onnx STT models: streaming Zipformer transducers for
 * SherpaEngine and offline models for SherpaOfflineEngine.
 *
 * Each descriptor names the release tarball (or, for a model published as
 * loose files, each file), its pinned digest and size, the per-model file
 * layout (names vary across releases), the licence
 * terms a consent surface shows, and the longest audio the engine should
 * decode in one utterance. Every entry is selectable by configuration;
 * swapping models is config-only: add a descriptor here, select it by id.
 *
 * Digests: "GitHub digest" is the release asset's published sha256 and
 * "publisher digest" the sha256 on the Hugging Face file listing or card; the
 * rest were computed 2026-10-31 by streaming the release asset through
 * `shasum -a 256`, since GitHub publishes no digest for them. Releases are
 * mutable, so an upstream change fails closed in `ensureAsset`.
 */

const ASR_MODELS_BASE =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

export type ModelArchive = {
  url: string;
  sha256: string;
  bytes: number;
};

export type ModelFile = AssetFile & { bytes: number };

/** Where a model downloads from: one release tarball, or pinned loose files. */
type ModelSource = { archive: ModelArchive } | { files: readonly ModelFile[] };

export type ModelLicence = {
  /** SPDX id, or a LicenseRef for unlisted terms. */
  id: string;
  terms: string;
  commercialUse: boolean;
  redistribution: boolean;
  /** Credit line a consumer must show, or null when none is required. */
  attribution: string | null;
  source: string;
};

type ModelCommon = ModelSource & {
  /** Release tarball basename; doubles as the extracted directory name. */
  name: string;
  /** Human note: architecture / training data / param scale. */
  note: string;
  licence: ModelLicence;
  /** Longest utterance a host should hold open for this model. */
  maxAudioMs: number;
};

type TransducerFiles = {
  encoder: string;
  decoder: string;
  joiner: string;
  tokens: string;
};

export type StreamingSherpaModel = ModelCommon &
  TransducerFiles & {
    kind: "streaming";
    /** bpe vocab for modelingUnit=bpe; omit for models shipped without one. */
    bpe?: string;
  };

export type NemoTransducerModel = ModelCommon &
  TransducerFiles & {
    kind: "offline";
    family: "nemo-transducer";
  };

export type MoonshineModel = ModelCommon & {
  kind: "offline";
  family: "moonshine";
  preprocessor: string;
  encoder: string;
  uncachedDecoder: string;
  cachedDecoder: string;
  tokens: string;
};

/** NeMo Canary encoder decoder; transcribes `language` with punctuation. */
export type CanaryModel = ModelCommon & {
  kind: "offline";
  family: "canary";
  encoder: string;
  decoder: string;
  tokens: string;
  language: string;
};

/** Qwen3 ASR: its vocabulary is a tokenizer directory, not a tokens file. */
export type Qwen3AsrModel = ModelCommon & {
  kind: "offline";
  family: "qwen3-asr";
  convFrontend: string;
  encoder: string;
  decoder: string;
  tokenizer: string;
};

export type OfflineSherpaModel =
  | NemoTransducerModel
  | MoonshineModel
  | CanaryModel
  | Qwen3AsrModel;
export type SherpaModel = StreamingSherpaModel | OfflineSherpaModel;

const STREAMING_MAX_AUDIO_MS = 120_000;

const APACHE_2_0 = {
  id: "Apache-2.0",
  terms:
    "Apache License 2.0: use, modify and redistribute; keep the licence and notices when redistributing.",
  commercialUse: true,
  redistribution: true,
  attribution: null,
} as const;

const CC_BY_4_0 = {
  id: "CC-BY-4.0",
  terms:
    "Creative Commons Attribution 4.0: use, adapt and redistribute, including commercially, with attribution.",
  commercialUse: true,
  redistribution: true,
} as const;

const PARAKEET_ULTRA_BASE =
  "https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8/resolve/b86784a3a5356cbd126f710b61cd90c5fe5898f1";

export const SHERPA_MODELS = {
  "en-parakeet-tdt-0.6b-v2-int8": {
    kind: "offline",
    family: "nemo-transducer",
    name: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    note: "NVIDIA Parakeet TDT 0.6B v2, offline, int8, punctuation and case",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2`,
      sha256: "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad", // GitHub digest
      bytes: 482_468_385,
    },
    licence: {
      ...CC_BY_4_0,
      attribution:
        "Speech recognition by NVIDIA parakeet-tdt-0.6b-v2, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/main/README.md",
    },
    maxAudioMs: 60_000,
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    joiner: "joiner.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-moonshine-base-int8": {
    kind: "offline",
    family: "moonshine",
    name: "sherpa-onnx-moonshine-base-en-int8",
    note: "Useful Sensors Moonshine base (original), offline, int8, punctuation and case",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-moonshine-base-en-int8.tar.bz2`,
      sha256: "21870cecaa2e44e4e2bf63e02d1072bed183ccd10284871353bd9d24dad14e5e", // computed
      bytes: 250_807_309,
    },
    licence: {
      id: "MIT",
      terms:
        "MIT: use, modify and redistribute, including commercially; include the copyright and permission notice when redistributing.",
      commercialUse: true,
      redistribution: true,
      attribution: null,
      source: "https://github.com/moonshine-ai/moonshine/blob/main/LICENSE",
    },
    maxAudioMs: 30_000,
    preprocessor: "preprocess.onnx",
    encoder: "encode.int8.onnx",
    uncachedDecoder: "uncached_decode.int8.onnx",
    cachedDecoder: "cached_decode.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-parakeet-ultra-int8": {
    kind: "offline",
    family: "nemo-transducer",
    name: "parakeet-ultra-sherpa-onnx-int8",
    note: "Moondream Parakeet Ultra (Parakeet TDT 0.6B v3 fine tune), sherpa export, offline, int8, punctuation and case",
    files: [
      {
        name: "encoder.int8.onnx",
        url: `${PARAKEET_ULTRA_BASE}/encoder.int8.onnx`,
        sha256: "f4d06712d32431b1718eceba2153c314ac4425e8fe708a3534f5e56c5ad53697", // publisher digest
        bytes: 652_282_347,
      },
      {
        name: "decoder.int8.onnx",
        url: `${PARAKEET_ULTRA_BASE}/decoder.int8.onnx`,
        sha256: "94373c6405c8149d1c9d1e9862514e0e3e88dab1c7de18fac9d5df9c6ba5efdd", // publisher digest
        bytes: 11_845_275,
      },
      {
        name: "joiner.int8.onnx",
        url: `${PARAKEET_ULTRA_BASE}/joiner.int8.onnx`,
        sha256: "8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2", // publisher digest
        bytes: 6_355_277,
      },
      {
        name: "tokens.txt",
        url: `${PARAKEET_ULTRA_BASE}/tokens.txt`,
        sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d", // publisher digest (card)
        bytes: 93_939,
      },
    ],
    licence: {
      ...CC_BY_4_0,
      attribution:
        "Speech recognition by Moondream parakeet-ultra, fine tuned from NVIDIA parakeet-tdt-0.6b-v3 and converted to sherpa-onnx by tpmaerk, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source:
        "https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8/blob/b86784a3a5356cbd126f710b61cd90c5fe5898f1/README.md",
    },
    maxAudioMs: 60_000,
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    joiner: "joiner.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-canary-180m-flash-int8": {
    kind: "offline",
    family: "canary",
    name: "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
    note: "NVIDIA Canary 180M Flash, offline, int8, English in and out, punctuation and case",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2`,
      sha256: "7a38ed8b13f014ad632b09ff8d22e0c6f1359dd046af9235d281dfae841b9ab9", // GitHub digest
      bytes: 153_692_328,
    },
    licence: {
      ...CC_BY_4_0,
      attribution:
        "Speech recognition by NVIDIA canary-180m-flash, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source:
        "https://huggingface.co/nvidia/canary-180m-flash/blob/b12ab418510d093e83890178fd0e8b0d0f7918a6/README.md",
    },
    // Card: "designed to handle input audio smaller than 40 seconds".
    maxAudioMs: 40_000,
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    tokens: "tokens.txt",
    language: "en",
  },
  "en-qwen3-asr-0.6b-int8-2026-03-25": {
    kind: "offline",
    family: "qwen3-asr",
    name: "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    note: "Qwen3 ASR 0.6B, offline, int8, punctuation and case, no token timestamps or scores",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2`,
      sha256: "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96", // GitHub digest
      bytes: 878_702_423,
    },
    licence: {
      ...APACHE_2_0,
      source:
        "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md",
    },
    maxAudioMs: 30_000,
    convFrontend: "conv_frontend.onnx",
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    tokenizer: "tokenizer",
  },
  "en-2023-06-21": {
    kind: "streaming",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-21",
    note: "Libri+Giga stateless7 streaming (large), 187MB int8 encoder",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2`,
      sha256: "455f40e556aa2b20ac9d3bffd603b58002075c1193b4070938540c11efe0a4da", // computed
      bytes: 506_956_414,
    },
    licence: {
      ...APACHE_2_0,
      source:
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21/blob/main/README.md",
    },
    maxAudioMs: STREAMING_MAX_AUDIO_MS,
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-kroko-2025-08-06": {
    kind: "streaming",
    name: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    note: "Banafo Kroko-ASR streaming, fp32-only (no int8 shipped)",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2`,
      sha256: "c8676e5ff9ac2a85296e53ee0fd4d5fb1db6770e7a7647166eeafe349ade6834", // GitHub digest
      bytes: 57_267_600,
    },
    licence: {
      id: "LicenseRef-Banafo-Kroko-noncommercial",
      terms:
        "Publisher card: \"the .onnx are still non-commercial only.\" Commercial use needs a licence from Banafo; redistribution terms are unpublished.",
      commercialUse: false,
      redistribution: false,
      attribution: null,
      source:
        "https://huggingface.co/Banafo/Kroko-ASR/blob/de5691bd6bc42533dd620fca38ee09edb96b8e29/README.md",
    },
    maxAudioMs: STREAMING_MAX_AUDIO_MS,
    encoder: "encoder.onnx",
    decoder: "decoder.onnx",
    joiner: "joiner.onnx",
    tokens: "tokens.txt",
  },
  "en-2023-06-26": {
    kind: "streaming",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    note: "LibriSpeech stateless7 streaming, ~66M, int8, bpe",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2`,
      sha256: "639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282", // computed
      bytes: 310_414_022,
    },
    licence: {
      ...APACHE_2_0,
      source:
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/blob/main/README.md",
    },
    maxAudioMs: STREAMING_MAX_AUDIO_MS,
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    tokens: "tokens.txt",
    bpe: "bpe.model",
  },
  "en-2023-02-21": {
    kind: "streaming",
    name: "sherpa-onnx-streaming-zipformer-en-2023-02-21",
    note: "LibriSpeech-only stateless7 streaming, int8",
    archive: {
      url: `${ASR_MODELS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2`,
      sha256: "54360f7a50b6cca9511fb318e472cbff87e9b48ba5f8f389bce46aeba7c31602", // computed
      bytes: 397_939_030,
    },
    licence: {
      ...APACHE_2_0,
      source:
        "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21/blob/main/README.md",
    },
    maxAudioMs: STREAMING_MAX_AUDIO_MS,
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    tokens: "tokens.txt",
  },
} as const satisfies Record<string, SherpaModel>;

export type SherpaModelId = keyof typeof SHERPA_MODELS;

// Verdict 2026-07-04: kroko wins the corpus sweep (12.8% normalized WER vs the
// 34.0% of en-2023-06-26) at the lowest latency, so it is speak-easy's CLI
// default. See MODEL-SWEEP.md. Both bench and demo default here when --model
// is omitted. Hosts choose their own model by configuration.
export const DEFAULT_SHERPA_MODEL: SherpaModelId = "en-kroko-2025-08-06";

/**
 * Resolve a descriptor to its pinned asset under `root`; descriptor file names
 * join onto the returned `dir`.
 */
export function resolveModelPaths(
  model: SherpaModel,
  root: string,
): AssetSpec | FileSetSpec {
  const dir = join(root, model.name);
  if ("files" in model) {
    return { dir, files: model.files };
  }
  return {
    url: model.archive.url,
    sha256: model.archive.sha256,
    archive: join(root, `${model.name}.tar.bz2`),
    dir,
  };
}

/** Validate a model id against the registry. */
export function resolveSherpaModel(id: string): SherpaModel {
  if (isSherpaModelId(id)) {
    return SHERPA_MODELS[id];
  }
  throw new Error(
    `Unknown sherpa model "${id}"; available: ${Object.keys(SHERPA_MODELS).join(", ")}`,
  );
}

export function isSherpaModelId(value: string): value is SherpaModelId {
  return Object.hasOwn(SHERPA_MODELS, value);
}

/** Parse a CLI `--model` value into a typed id, or throw with the valid set. */
export function parseSherpaModelId(value: string): SherpaModelId {
  if (isSherpaModelId(value)) {
    return value;
  }
  throw new Error(
    `--model must be one of ${Object.keys(SHERPA_MODELS).join(", ")}; received ${value}`,
  );
}
