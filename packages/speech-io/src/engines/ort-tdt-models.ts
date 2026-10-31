import { join } from "node:path";
import type { FileSetSpec } from "./assets.ts";
import type { ModelFile, ModelLicence } from "./sherpa-models.ts";

/**
 * Registry of Parakeet TDT exports that run on onnxruntime-node rather than
 * sherpa-onnx, for OrtTdtEngine. None loads through sherpa's nemo_transducer
 * config: their encoders carry no sherpa metadata, Phonon-2 and Redux fuse
 * the prediction network with the joint, and Fluid splits its encoder.
 *
 * Two graph layouts share one greedy TDT decoder:
 * - "fused" (onnx-asr nemo-conformer-tdt): a preprocessor graph turns audio
 *   into features, then encoder, then one decoder_joint graph per step.
 * - "split" (Fluid Speech Mini): log-mel in JavaScript, a convolution front
 *   end run in time chunks, encoder, then separate decoder and joint graphs.
 *
 * Every file is pinned to a repository revision. Digests are the publisher's
 * LFS sha256, except where marked "computed": small text files the publisher
 * pins only by Git blob id, hashed 2026-10-31 after their blob ids matched.
 */

export type TdtDecoding = {
  /** Token logits, blank included; duration logits follow them. */
  vocabSize: number;
  blankId: number;
  durations: readonly number[];
  /** Tokens one encoder frame may emit before the decoder must advance. */
  maxSymbolsPerFrame: number;
  /** Encoder frame stride. */
  frameMs: number;
  /** LSTM prediction network state shape: [layers, 1, hidden]. */
  predictor: { layers: number; hidden: number };
  /** A token that ends the utterance, when the reference decoder stops on one. */
  eosId?: number;
};

export type MelConfig = {
  nFft: number;
  hop: number;
  win: number;
  preemphasis: number;
};

type OrtTdtCommon = {
  kind: "offline";
  /** Directory name under the model root. */
  name: string;
  note: string;
  /** Files at one pinned repository revision, under the model directory. */
  files: readonly ModelFile[];
  licence: ModelLicence;
  /** Longest utterance a host should hold open for this model. */
  maxAudioMs: number;
  decoding: TdtDecoding;
  encoder: string;
};

export type FusedTdtModel = OrtTdtCommon & {
  layout: "fused";
  preprocessor: string;
  decoderJoint: string;
  /** onnx-asr vocab.txt: one "piece id" per line. */
  vocab: string;
};

export type SplitTdtModel = OrtTdtCommon & {
  layout: "split";
  /** numpy [mels, nFft / 2 + 1] float32 filter bank. */
  melFilters: string;
  mel: MelConfig;
  frontend: string;
  /** Front end output frames per chunk. */
  frontendChunkFrames: number;
  subsampling: { kernel: number; stride: number; layers: number };
  decoder: string;
  joint: string;
  /** Hugging Face tokenizer.json. */
  tokenizer: string;
};

export type OrtTdtModel = FusedTdtModel | SplitTdtModel;

const CC_BY = {
  id: "CC-BY-4.0",
  terms:
    "Creative Commons Attribution 4.0: use, adapt and redistribute, including commercially, with attribution.",
  commercialUse: true,
  redistribution: true,
} as const;

/** Parakeet TDT 0.6B v3 decoding, shared by all three exports. */
const PARAKEET_V3_TDT: TdtDecoding = {
  vocabSize: 8_193,
  blankId: 8_192,
  durations: [0, 1, 2, 3, 4],
  maxSymbolsPerFrame: 10,
  frameMs: 80,
  predictor: { layers: 2, hidden: 640 },
};

const FLUID_BASE =
  "https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-onnx/resolve/fc592620325027419883cf272e70588d8a788c0a";
const PHONON_BASE =
  "https://huggingface.co/tiyuvta/Phonon-2-ONNX/resolve/12c9688bbc4fc52d23c1a66ca873fd3ac6ed4408";
const REDUX_BASE =
  "https://huggingface.co/eschmidbauer/parakeet-redux-onnx/resolve/1c285ba78d6f1880803f65355133a5ad3559a407";

function pinned(base: string, name: string, bytes: number, sha256: string): ModelFile {
  return { name, url: `${base}/${name}`, sha256, bytes };
}

export const ORT_TDT_MODELS = {
  "en-fluid-speech-mini": {
    kind: "offline",
    layout: "split",
    name: "fluid-speech-mini-onnx",
    note: "FluidVoice Fluid Speech Mini, Parakeet TDT v3 derivative, 2 and 4 bit MatMulNBits, punctuation and case",
    files: [
      pinned(FLUID_BASE, "frontend.onnx", 8_697_130, "cdbea06631920f25f6183e1389129e9c37106a766c6e130b7f0389b9531b43b7"),
      pinned(FLUID_BASE, "encoder.onnx", 426_102, "f8b3a45e71a33b44179a21ef6d6474779747846c4b01c9fd690abcebb88bea86"),
      pinned(FLUID_BASE, "encoder.onnx.data", 253_429_760, "f865c582bec1bd57aa7f2242f3739f38987b7bf471fee4dca7d5c2714eec0747"),
      pinned(FLUID_BASE, "decoder.onnx", 24_461_296, "9ac7e6c22952c242d56df0992ba19489a05b87a8ed37e23798469b9316099af8"),
      pinned(FLUID_BASE, "joint.onnx", 11_840_320, "2e704908834b9f8cc7db88757b8966bac266e19bc00a527c46e9950281205253"),
      pinned(FLUID_BASE, "mel_filters.npy", 131_712, "ce607f7ae1ce68fd01c02e1177fb1d11ee209c6176d35e47011917ca7d9cd4f2"),
      pinned(FLUID_BASE, "tokenizer.json", 1_159_959, "7c200d88ac60da4de246cbd38e981a732f2e82580fdd285cb757d8d1f4af9627"), // computed
      pinned(FLUID_BASE, "LICENSE", 18_657, "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"), // computed
      pinned(FLUID_BASE, "NOTICE", 507, "29ac6c3c2bf16b44f2f544df9bed2bf5aaf1afd8c5674aedfe99ddce2cdcb223"), // computed
    ],
    licence: {
      ...CC_BY,
      attribution:
        "Fluid Speech Mini by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source:
        "https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-onnx/blob/fc592620325027419883cf272e70588d8a788c0a/NOTICE",
    },
    // config.json max_single_pass_seconds; the runner splits longer audio.
    maxAudioMs: 40_000,
    decoding: { ...PARAKEET_V3_TDT, eosId: 3 },
    melFilters: "mel_filters.npy",
    mel: { nFft: 512, hop: 160, win: 400, preemphasis: 0.97 },
    frontend: "frontend.onnx",
    frontendChunkFrames: 50,
    subsampling: { kernel: 3, stride: 2, layers: 3 },
    encoder: "encoder.onnx",
    decoder: "decoder.onnx",
    joint: "joint.onnx",
    tokenizer: "tokenizer.json",
  },
  "en-phonon-2-int8": {
    kind: "offline",
    layout: "fused",
    name: "phonon-2-onnx-int8",
    note: "Fermion Research Phonon-2, Parakeet TDT v3 derivative, int8 encoder, punctuation and case",
    files: [
      pinned(PHONON_BASE, "preprocessor-model.onnx", 1_193_996, "8184d564f7d34d1daf04e4b35a0222fb72c54668b38dcd2b8ddda3517676614b"),
      pinned(PHONON_BASE, "encoder-model.int8.onnx", 614_486_780, "3c100e38ca2e70623c928ab5c5414c62848603ea3d943a7f2e1c3ce1d92fc04b"),
      pinned(PHONON_BASE, "decoder_joint-model.onnx", 72_518_934, "420125e0e13596692320c35ef648eee9bf4583718c7896c8732ebf6f50b9ca0d"),
      pinned(PHONON_BASE, "vocab.txt", 93_939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"), // computed
      pinned(PHONON_BASE, "NOTICE", 3_306, "8d9d079319a53da70cda7589f597f0c5b2fce8ad61b6ef7ec67d59a0bbbffdad"), // computed
    ],
    licence: {
      ...CC_BY,
      attribution:
        "Phonon-2 by Fermion Research, derived from NVIDIA Parakeet TDT 0.6B v3; ONNX conversion by Tiyuvta. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source:
        "https://huggingface.co/tiyuvta/Phonon-2-ONNX/blob/12c9688bbc4fc52d23c1a66ca873fd3ac6ed4408/README.md#license-and-attribution",
    },
    // Export card: audio longer than about 35 s should be cut at pauses.
    maxAudioMs: 35_000,
    decoding: PARAKEET_V3_TDT,
    preprocessor: "preprocessor-model.onnx",
    encoder: "encoder-model.int8.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocab: "vocab.txt",
  },
  "en-parakeet-redux": {
    kind: "offline",
    layout: "fused",
    name: "parakeet-redux-onnx",
    note: "Moondream Parakeet Redux, ternary Parakeet TDT v3 in 4 bit MatMulNBits, punctuation and case",
    files: [
      pinned(REDUX_BASE, "preprocessor.onnx", 1_224_294, "3de5742f5780fc8615c1d883a2a60e28621ff13d2e20d24d76e94b9e4168445f"),
      pinned(REDUX_BASE, "encoder-model.onnx", 343_841_943, "ade2c65c188776b3662db620a553ab82ac7e9ca11a7be6b696bcc5190f75382e"),
      pinned(REDUX_BASE, "decoder_joint-model.onnx", 72_552_270, "97ae0cb049f32a5d7a6572e91020b8d87d66baffa0ff9de2cf6de4c0960e92f8"),
      pinned(REDUX_BASE, "vocab.txt", 93_939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"), // computed
    ],
    licence: {
      ...CC_BY,
      attribution:
        "Parakeet Redux by Moondream, derived from NVIDIA Parakeet TDT 0.6B v3; ONNX export by eschmidbauer. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
      source:
        "https://huggingface.co/eschmidbauer/parakeet-redux-onnx/blob/1c285ba78d6f1880803f65355133a5ad3559a407/README.md",
    },
    // The reference runner cuts recordings over 30 s at pauses.
    maxAudioMs: 30_000,
    decoding: PARAKEET_V3_TDT,
    preprocessor: "preprocessor.onnx",
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocab: "vocab.txt",
  },
} as const satisfies Record<string, OrtTdtModel>;

export type OrtTdtModelId = keyof typeof ORT_TDT_MODELS;

export function isOrtTdtModelId(value: string): value is OrtTdtModelId {
  return Object.hasOwn(ORT_TDT_MODELS, value);
}

export function resolveOrtTdtModel(id: string): OrtTdtModel {
  if (isOrtTdtModelId(id)) {
    return ORT_TDT_MODELS[id];
  }
  throw new Error(
    `Unknown ORT TDT model "${id}"; available: ${Object.keys(ORT_TDT_MODELS).join(", ")}`,
  );
}

/** Total download size, the figure a consent surface shows. */
export function downloadBytes(model: OrtTdtModel): number {
  return model.files.reduce((total, file) => total + file.bytes, 0);
}

export function resolveOrtTdtPaths(model: OrtTdtModel, root: string): FileSetSpec {
  return { dir: join(root, model.name), files: model.files };
}
