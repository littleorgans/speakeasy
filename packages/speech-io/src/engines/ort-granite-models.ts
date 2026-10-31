import { join } from "node:path";
import type { FileSetSpec } from "./assets.ts";
import type { ModelFile, ModelLicence } from "./sherpa-models.ts";

/**
 * Registry of Granite Speech CTC models for `OrtGraniteEngine`, published as
 * loose files on Hugging Face rather than a sherpa release tarball.
 *
 * Every URL resolves at a pinned commit, so the bytes are immutable. Graph
 * digests are the published LFS sha256; tokenizer.json is not LFS, so its
 * digest was computed 2026-10-31 by fetching it at the pinned commit through
 * `shasum -a 256`. Only the graph and tokenizer are fetched: the frontend
 * constants of `preprocessor_config.json` are ported into granite-frontend.ts.
 * The separate `-nc` IBM weights are research only and never listed.
 */

const GRANITE_EXPORT =
  "https://huggingface.co/qwertz92/granite-speech-5.0-470m-turboctc-onnx/resolve/e6e3b4d6f590ac51a05f25028b53591d4b260786";

export type GraniteCtcModel = {
  kind: "offline";
  family: "granite-ctc";
  /** Directory name under the model root. */
  name: string;
  note: string;
  graph: ModelFile;
  tokenizer: ModelFile;
  /** The CTC blank, which is the tokenizer's pad token. */
  blankToken: string;
  licence: ModelLicence;
  /** Longest utterance a host should hold open for this model. */
  maxAudioMs: number;
};

export const GRANITE_MODELS = {
  "en-granite-speech-5.0-470m-turboctc-int8": {
    kind: "offline",
    family: "granite-ctc",
    name: "granite-speech-5.0-470m-turboctc-onnx-int8",
    note: "IBM Granite Speech 5.0 470M TurboCTC, English, qwertz92 dynamic int8 ONNX export, lowercase without punctuation",
    graph: {
      name: "model_int8.onnx",
      url: `${GRANITE_EXPORT}/onnx/model_int8.onnx`,
      sha256: "8173a50ea67f864801971a40622a2e5c7d62fd230912bde94b745f92c74d60e9", // LFS digest
      bytes: 551_294_349,
    },
    tokenizer: {
      name: "tokenizer.json",
      url: `${GRANITE_EXPORT}/tokenizer.json`,
      sha256: "3ee80b02f0119a040a70eb909c20fac8271c173d7e71d195a3b35f77780061e6", // computed
      bytes: 1_137_492,
    },
    blankToken: "<|blank|>",
    licence: {
      id: "Apache-2.0",
      terms:
        "Apache License 2.0: use, modify and redistribute, including commercially; keep the licence, notices and change markings when redistributing.",
      commercialUse: true,
      redistribution: true,
      attribution: null,
      source:
        "https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/blob/18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0/README.md",
    },
    // The export's validation covers clips up to 29.4 s.
    maxAudioMs: 30_000,
  },
} as const satisfies Record<string, GraniteCtcModel>;

export type GraniteModelId = keyof typeof GRANITE_MODELS;

export function isGraniteModelId(value: string): value is GraniteModelId {
  return Object.hasOwn(GRANITE_MODELS, value);
}

export function resolveGraniteModel(id: string): GraniteCtcModel {
  if (isGraniteModelId(id)) {
    return GRANITE_MODELS[id];
  }
  throw new Error(
    `Unknown Granite model "${id}"; available: ${Object.keys(GRANITE_MODELS).join(", ")}`,
  );
}

/** The pinned file set under `root`; file names join onto the returned `dir`. */
export function resolveGraniteFiles(model: GraniteCtcModel, root: string): FileSetSpec {
  return { dir: join(root, model.name), files: [model.graph, model.tokenizer] };
}

