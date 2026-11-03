import {
  APACHE_2_0,
  hasModelId,
  lookupModel,
  pinned,
  type FileSetSource,
  type OfflineCommon,
} from "./model.ts";

/**
 * Registry of Whistle encoder decoder models for OrtWhistleEngine, run on
 * onnxruntime-node with the vendored reference pipeline (vendor/whistle).
 *
 * The export publishes loose files, no archive, so each descriptor pins every
 * file's URL, size and sha256 at one revision; `encoder`, `decoder` and
 * `pack` name entries of `files`. ONNX external data (`*.onnx.data`) must sit
 * beside its graph. Sizes and digests are the export's published
 * `siblings[].size` and `lfs.sha256` (research/cand/whistle.md).
 */

const WHISTLE_ONNX_BASE =
  "https://huggingface.co/mrfakename/whistle-ONNX/resolve/14fc92cb09ad5c271fe9a4ec4be70274790050d2";

export type WhistleModel = OfflineCommon<FileSetSource> & {
  family: "whistle";
  /** Language codes the decoder accepts as a hint; see `vendor/whistle/whistle.js::LANGS`. */
  languages: readonly string[];
  /** onnxruntime intra op threads; omit for `ort-models.ts::ORT_THREADS`. */
  threads?: number;
  encoder: string;
  decoder: string;
  pack: string;
};

export const ORT_WHISTLE_MODELS = {
  "whistle-onnx-fp32": {
    kind: "offline",
    family: "whistle",
    name: "whistle-onnx-fp32-14fc92c",
    note: "Cactus Whistle, community ONNX export, fp32, greedy, 7 languages, word timestamps",
    files: [
      pinned(WHISTLE_ONNX_BASE, "onnx/encoder.onnx", 526_519, "feda0a9192f65921a30f94d37bcfce533a2545a1fc3415639ae83aeee4800ae4"),
      pinned(WHISTLE_ONNX_BASE, "onnx/encoder.onnx.data", 72_119_172, "8fd06fd50ef2972808ec1a15e206a2b9fd2a73e3a2f9c1fa3f137b163c4f332f"),
      pinned(WHISTLE_ONNX_BASE, "onnx/decoder.onnx", 423_496, "d2309759f3db92a23cefb8265132b744dd7ca602cb9c30902451a8b2587d3a75"),
      pinned(WHISTLE_ONNX_BASE, "onnx/decoder.onnx.data", 73_005_472, "222af8eda1e19756d38bbd5edd469630ffa67a4dd4ea4df6526e001bc032415d"),
      pinned(WHISTLE_ONNX_BASE, "whistle.pack", 17_015_856, "1edf33dcf5772dad142d6f19571371f98999053a7a3b4397364bea0d8f4679d0"),
    ],
    licence: {
      ...APACHE_2_0,
      source:
        "https://huggingface.co/Cactus-Compute/whistle/blob/741118687e1f008f06d1351a402a731274789790/LICENSE",
    },
    languages: ["en", "de", "fr", "es", "it", "nl", "pl"],
    // The original card's single pass window.
    maxAudioMs: 30_000,
    // Answers silence and noise with "Thank you." as confidently as speech
    // (mean word log probability -0.01 to -0.08), so no floor separates them.
    silenceGate: { logProbFloor: null },
    encoder: "onnx/encoder.onnx",
    decoder: "onnx/decoder.onnx",
    pack: "whistle.pack",
  },
} as const satisfies Record<string, WhistleModel>;

export type WhistleModelId = keyof typeof ORT_WHISTLE_MODELS;

export function isWhistleModelId(value: string): value is WhistleModelId {
  return hasModelId(ORT_WHISTLE_MODELS, value);
}

export function resolveWhistleModel(id: string): WhistleModel {
  return lookupModel(ORT_WHISTLE_MODELS, id, "Whistle");
}
