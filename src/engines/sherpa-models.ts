import { join } from "node:path";

/**
 * Registry of streaming Zipformer models the SherpaEngine can load.
 *
 * Each descriptor names the release tarball plus its per-model onnx/tokens
 * filenames, which vary across releases (chunk-size suffixes, int8 vs fp32,
 * some ship a bpe.model and some do not). Swapping models is config-only:
 * add a descriptor here, select it with `--model <id>`. The bench harness and
 * demo both construct the one parameterized SherpaEngine from these ids.
 */

const ASR_MODELS_BASE =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

/** On-disk root for downloaded + extracted sherpa models. */
export const SHERPA_ROOT = join(process.cwd(), "models", "sherpa");

export type SherpaModel = {
  /** Release tarball basename; doubles as the extracted directory name. */
  name: string;
  /** Human note: architecture / training data / param scale. */
  note: string;
  encoder: string;
  decoder: string;
  joiner: string;
  tokens: string;
  /** bpe vocab for modelingUnit=bpe; omit for models shipped without one. */
  bpe?: string;
};

export const SHERPA_MODELS = {
  "en-2023-06-26": {
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    note: "GigaSpeech+LibriSpeech stateless7 streaming, ~66M, int8, bpe",
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    tokens: "tokens.txt",
    bpe: "bpe.model",
  },
  "en-2023-06-21": {
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-21",
    note: "Libri+Giga stateless7 streaming (large), 187MB int8 encoder",
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-2023-02-21": {
    name: "sherpa-onnx-streaming-zipformer-en-2023-02-21",
    note: "LibriSpeech-only stateless7 streaming, int8",
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    tokens: "tokens.txt",
  },
  "en-kroko-2025-08-06": {
    name: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    note: "Banafo Kroko-ASR streaming, fp32-only (no int8 shipped)",
    encoder: "encoder.onnx",
    decoder: "decoder.onnx",
    joiner: "joiner.onnx",
    tokens: "tokens.txt",
  },
} as const satisfies Record<string, SherpaModel>;

export type SherpaModelId = keyof typeof SHERPA_MODELS;

// Verdict 2026-07-04: kroko wins the corpus sweep (12.8% normalized WER vs the
// 34.0% of en-2023-06-26) at the lowest latency, so it is the default. See
// MODEL-SWEEP.md. Both bench and demo default here when --model is omitted.
export const DEFAULT_SHERPA_MODEL: SherpaModelId = "en-kroko-2025-08-06";

export type SherpaModelPaths = {
  dir: string;
  archive: string;
  url: string;
  encoder: string;
  decoder: string;
  joiner: string;
  tokens: string;
  bpe?: string;
};

/** Resolve a descriptor to absolute on-disk paths and the download URL. */
export function resolveModelPaths(model: SherpaModel): SherpaModelPaths {
  const dir = join(SHERPA_ROOT, model.name);
  return {
    dir,
    archive: join(SHERPA_ROOT, `${model.name}.tar.bz2`),
    url: `${ASR_MODELS_BASE}/${model.name}.tar.bz2`,
    encoder: join(dir, model.encoder),
    decoder: join(dir, model.decoder),
    joiner: join(dir, model.joiner),
    tokens: join(dir, model.tokens),
    bpe: model.bpe ? join(dir, model.bpe) : undefined,
  };
}

/** Validate a `--model` value against the registry. */
export function resolveSherpaModel(id: string): SherpaModel {
  if (id in SHERPA_MODELS) {
    return SHERPA_MODELS[id as SherpaModelId];
  }
  throw new Error(
    `Unknown sherpa model "${id}"; available: ${Object.keys(SHERPA_MODELS).join(", ")}`,
  );
}

export function isSherpaModelId(value: string): value is SherpaModelId {
  return value in SHERPA_MODELS;
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
