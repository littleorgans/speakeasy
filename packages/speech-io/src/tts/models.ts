import { stat } from "node:fs/promises";
import { homedir } from "node:os";
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
/** Where a cached worker voice's payload lives, whatever the output root: never inside a repo. */
export const TTS_CACHE_ROOT = join(homedir(), "Library", "Caches", "speakeasy", "tts");

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

/**
 * A voice a host runs in its own worker runtime (MLX, Core ML, a native
 * binary), never through sherpa. Its payload is pinned here like any other;
 * the host supplies the runtime.
 */
export type WorkerTtsModel = {
  name: string;
  note: string;
  family: "worker";
  sampleRate: number;
  numSpeakers: number;
  files: readonly TtsModelFile[];
  attribution: readonly string[];
  /** A runtime that takes phonemes: the host converts text with this sherpa voice's Piper Plus frontend. */
  phonemesFrom?: "piper-kristin-medium";
  /** The Hugging Face repo and revision its files come from; the payload lives under TTS_CACHE_ROOT. */
  hub?: { repo: string; revision: string };
};

/** A voice payload's directory: under the cache root for a hub voice, else under the output root. */
export function payloadDir(model: TtsModel | WorkerTtsModel, root: string): string {
  return join("hub" in model && model.hub ? TTS_CACHE_ROOT : root, model.name);
}

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
  "kokoro-mlx-bf16": {
    name: "kokoro-82m-mlx-bf16-a71e4d38",
    note: "Kokoro v1.0 bf16 for MLX, mlx-community/Kokoro-82M-bf16@a71e4d38, two voices, 24000 Hz",
    family: "worker",
    sampleRate: 24_000,
    numSpeakers: 2,
    files: [
      { name: "config.json", bytes: 2_351, sha256: "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f" },
      { name: "kokoro-v1_0.safetensors", bytes: 327_115_152, sha256: "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8" },
      { name: "voices/af_heart.safetensors", bytes: 522_320, sha256: "2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094" },
      { name: "voices/am_michael.safetensors", bytes: 522_320, sha256: "3940147ded35deba0bb52e8132f89b719298e0520258c34584358aa5a24da2ea" },
    ],
    attribution: ["README.md"],
  },
  "pocket-18m": {
    name: "pocket-tts-english-18m-2026-10-a7056f50",
    note: "Pocket TTS English 18M, kyutai/pocket-tts-without-voice-cloning@a7056f50, bill_boerst voice, 24000 Hz",
    family: "worker",
    sampleRate: 24_000,
    numSpeakers: 1,
    files: [
      { name: "languages/english_18m_2026-10/model.safetensors", bytes: 111_728_528, sha256: "599329dcea2f2a8d17b575e1c6e11b78822453c9073040d55c26faadf4883188" },
      { name: "languages/english_18m_2026-10/tokenizer.model", bytes: 59_339, sha256: "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6" },
      { name: "languages/english_18m_2026-10/embeddings/bill_boerst.safetensors", bytes: 4_360_600, sha256: "927a31d7627daf996efce71ee7898a799a9386738077c3bd401bd1445f6f6657" },
    ],
    attribution: ["README.md"],
  },
  "piper-kristin-split": {
    name: "piper-kristin-split-e7229f7d",
    note: "Piper Kristin medium split into encoder and decoder for libpiper streaming, from rhasspy/piper-voices@e7229f7d",
    family: "worker",
    sampleRate: 22_050,
    numSpeakers: 1,
    files: [
      { name: "en_US-kristin-medium.encoder.onnx", bytes: 57_248_767, sha256: "380d38540eaf9cb33e6ce43bb6e2458f55da0d19405c5f65852bb7a0e74573e4" },
      { name: "en_US-kristin-medium.decoder.onnx", bytes: 6_670_623, sha256: "ecf60b852d2e7edd4d51796ec8a00c1560b19ee85b3ff819daf6a82ce6a768cb" },
      { name: "en_US-kristin-medium.onnx.json", bytes: 4_967, sha256: "986bba3f536ab08d92c92e4e2223229eed686eda933d30a57604e981dcea0c24" },
    ],
    // Kristin's MODEL_CARD stays with the voice it derives from, checked through phonemesFrom.
    attribution: [],
    phonemesFrom: "piper-kristin-medium",
  },
  "kokoro-coreml": {
    name: "kokoro-ane-fluidaudio",
    note: "Kokoro v1.0 Core ML stages and English G2P, FluidInference/kokoro-82m-coreml@2568276a, legacy route",
    family: "worker",
    sampleRate: 24_000,
    numSpeakers: 2,
    files: [
      { name: "kokoro/g2p_vocab.json", bytes: 1_665, sha256: "295ed64b86c2820cd665b0602ae50c6947c0e82ac643082873e0be87dca282ce" },
      { name: "kokoro/us_lexicon_cache.json", bytes: 10_444_631, sha256: "6b36ba313202227d6914ad32cd684a0304bd2757e9ec4158ea7bc36ec40e224e" },
      { name: "kokoro-82m-coreml/ANE/.fluidaudio-revision", bytes: 41, sha256: "63e60995f4b782b49549c6797f7fb315505951027eee75bb54dd995e1cb2be8a" },
      { name: "kokoro-82m-coreml/ANE/af_heart.bin", bytes: 522_240, sha256: "d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b" },
      { name: "kokoro-82m-coreml/ANE/am_michael.bin", bytes: 522_240, sha256: "1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1" },
      { name: "kokoro-82m-coreml/ANE/vocab.json", bytes: 1_416, sha256: "8d65b0188b77eafc60751dac42bbac7ab5f5685074af44db91d1877b42dc1d7c" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlbert.mlmodelc/coremldata.bin", bytes: 433, sha256: "70b6d2f8429229f6800dda9480341669f7ac0eabf05a82934d8632ab2b4b63a6" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlbert.mlmodelc/metadata.json", bytes: 2_480, sha256: "fe1d005481f646707a948267ac089dfb2cd2ccea7d5827a14c28587eb4c89930" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlbert.mlmodelc/model.mil", bytes: 101_485, sha256: "2038154d06a20e399a8ae35ec5c9242702cb23db8dd24dde7679cd53708e4eae" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlbert.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "14a61873d8759a38b79c93f9021ae865f408f56301209a0168dbfc2283265ccd" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlbert.mlmodelc/weights/weight.bin", bytes: 5_718_848, sha256: "36089a39359b800d3e2c60e5e8ac9217d8f2d1010a8b9273192290e621f1fabc" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlignment.mlmodelc/coremldata.bin", bytes: 484, sha256: "9a0fb4a536f665a052914d7f17b0e6ac80a3614f702e4be87ac0302c1143a4ea" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlignment.mlmodelc/metadata.json", bytes: 3_021, sha256: "6f610d23b9f93c7f1a968d6a861efa99725a191de696bf1c3be334ead39b7f00" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlignment.mlmodelc/model.mil", bytes: 8_194, sha256: "eb3a618bda0cdf95cdffab586ef5c76333d0d0a1dcb881190b3ef414f3421b3e" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlignment.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "f6074d1039a9151d0f97dc6ec9ee0cd9c7b865f1af646d718ab38b659fa84f3f" },
      { name: "kokoro-82m-coreml/ANE/KokoroAlignment.mlmodelc/weights/weight.bin", bytes: 4_128, sha256: "2e7d69128b59d615fc3d3cf85637a687235fc086b1eb136359adb11a61615f6b" },
      { name: "kokoro-82m-coreml/ANE/KokoroNoise_v2.mlmodelc/coremldata.bin", bytes: 440, sha256: "9911047f924b41f8b92811c32c50b7c718bd7b0c14842b3bc1b66b9ffe341a19" },
      { name: "kokoro-82m-coreml/ANE/KokoroNoise_v2.mlmodelc/metadata.json", bytes: 3_020, sha256: "eb3102217532d952479c6f1b04d26d7ac4cdbc59875b6a23e8aae86e5e02c2bc" },
      { name: "kokoro-82m-coreml/ANE/KokoroNoise_v2.mlmodelc/model.mil", bytes: 93_152, sha256: "60233949d896f15ef38aea19afda558935d7569fa77a3ca4babddd3ffe845a36" },
      { name: "kokoro-82m-coreml/ANE/KokoroNoise_v2.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "53af6bf61482f6002bdb6e3a62f30774cde6e96411aebd888222a34b369f3d04" },
      { name: "kokoro-82m-coreml/ANE/KokoroNoise_v2.mlmodelc/weights/weight.bin", bytes: 4_580_160, sha256: "1102fc2d31dfcfe3de3978a4c78b65202ff8b0a4d55a9304213bd4e8bda66bc2" },
      { name: "kokoro-82m-coreml/ANE/KokoroPostAlbert.mlmodelc/coremldata.bin", bytes: 556, sha256: "86de3ab0c1e8c6f8842b57bc24695a5099590c49b864e3fa2737b1fb5b15ba3b" },
      { name: "kokoro-82m-coreml/ANE/KokoroPostAlbert.mlmodelc/metadata.json", bytes: 4_171, sha256: "2e3be1af412a76e340120a01cd04a502666371ada86b1fdaf5a622896e0bf979" },
      { name: "kokoro-82m-coreml/ANE/KokoroPostAlbert.mlmodelc/model.mil", bytes: 60_047, sha256: "450b73a3b179e1702e7b2210bad008eac20d58b2d6076e3c2de76290a66e5fef" },
      { name: "kokoro-82m-coreml/ANE/KokoroPostAlbert.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "6640044c875505382edbc361cdd56f3f1c30f082953e0f9473a31ae3f71e6c43" },
      { name: "kokoro-82m-coreml/ANE/KokoroPostAlbert.mlmodelc/weights/weight.bin", bytes: 13_806_464, sha256: "e4f300a23cc2e05d38680d9fc94681cc722d445076d1d248f83255122ba091c8" },
      { name: "kokoro-82m-coreml/ANE/KokoroProsody_v2.mlmodelc/coremldata.bin", bytes: 421, sha256: "db8776a4896f690fd254cdb68a4551c168411c9b3fb98219630215bc3c357986" },
      { name: "kokoro-82m-coreml/ANE/KokoroProsody_v2.mlmodelc/model.mil", bytes: 73_353, sha256: "e6acad9f44d70b1dba0aea6a5d25ee4edf68fe8f2933399ee81f70ace61cbc8b" },
      { name: "kokoro-82m-coreml/ANE/KokoroProsody_v2.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "804d7d2e9a8fed345cedb57da61bd27fdd95db22a070cfa7d42ccd4f8e80a443" },
      { name: "kokoro-82m-coreml/ANE/KokoroProsody_v2.mlmodelc/weights/weight.bin", bytes: 8_513_664, sha256: "70eea4cd2523992fa1fea840e31ab6f4fdd69c1598e220985a2ae6ea5da1e944" },
      { name: "kokoro-82m-coreml/ANE/KokoroTail_v2.mlmodelc/coremldata.bin", bytes: 392, sha256: "527271f6df56cc308a5f89374c92108aef7b837ab15f0ba4900d555588dbd905" },
      { name: "kokoro-82m-coreml/ANE/KokoroTail_v2.mlmodelc/metadata.json", bytes: 1_872, sha256: "7708ecc145eecf8e3ef5ef8979ea7a4f77d04c8da787454c6d9190e5300fc50b" },
      { name: "kokoro-82m-coreml/ANE/KokoroTail_v2.mlmodelc/model.mil", bytes: 7_014, sha256: "b0b8fd573bac76ba7eb85730eeb25538fc7f1c666ecbf939fd4b3a4ad4495ad7" },
      { name: "kokoro-82m-coreml/ANE/KokoroTail_v2.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "a06e4b91c2b8a8be3ff701558a5f1f6484921f948202663fe6df78f248d42222" },
      { name: "kokoro-82m-coreml/ANE/KokoroTail_v2.mlmodelc/weights/weight.bin", bytes: 81_088, sha256: "8ede31ec20df7d86a124287ac12d13507651e2a693e9d4f35f0e7f92cba422c7" },
      { name: "kokoro-82m-coreml/ANE/KokoroVocoder.mlmodelc/coremldata.bin", bytes: 626, sha256: "71bac80d2f2f077fefc0d66cdc39c0faaa96540b3a6149093ef29de01b1f8764" },
      { name: "kokoro-82m-coreml/ANE/KokoroVocoder.mlmodelc/metadata.json", bytes: 4_331, sha256: "a8e768f151225a52476d35f6ab90805ae00f1f78b83be4ca21327029c800c183" },
      { name: "kokoro-82m-coreml/ANE/KokoroVocoder.mlmodelc/model.mil", bytes: 309_000, sha256: "52f2febec45094e78a5948a3a47f023be8fc57cfd48cc99e8d4c358a13674ef2" },
      { name: "kokoro-82m-coreml/ANE/KokoroVocoder.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "1cc4f1e6436597c6d458a845570d9d0ace458813ff90f225d08e0e713496b463" },
      { name: "kokoro-82m-coreml/ANE/KokoroVocoder.mlmodelc/weights/weight.bin", bytes: 48_889_920, sha256: "6d1f96eb50218ab687b12d6d862d2ae854c12b7165c3cd9b6b5cef261ef02ff1" },
      { name: "kokoro/G2PDecoder.mlmodelc/coremldata.bin", bytes: 545, sha256: "607e960f19b4d9a30317a5a11869fcce84b300a909fcab2cc756c0d98e2dacd9" },
      { name: "kokoro/G2PDecoder.mlmodelc/metadata.json", bytes: 3_064, sha256: "e54e98484fd60d26f22fd3c4e7fe87b0d92a5d2de1f958cc3c4bb36d4ae06a44" },
      { name: "kokoro/G2PDecoder.mlmodelc/model.mil", bytes: 19_737, sha256: "fe647c598e0d9454d360b8ee49a59ae57ca147fc5330863ba84ccb90dce482ad" },
      { name: "kokoro/G2PDecoder.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "dbf1767747fdc188222d467a45b04608b396c76c71db3abb18a5fb3680ef9827" },
      { name: "kokoro/G2PDecoder.mlmodelc/weights/weight.bin", bytes: 828_030, sha256: "cbaeb4e743359f607ab161af0c6d8a817462fdaec622ee788ef8ef952c5f8214" },
      { name: "kokoro/G2PEncoder.mlmodelc/coremldata.bin", bytes: 398, sha256: "0f14d46ca9fd06c68b4717294575b2b99449e67d40b7a2c56f926bf05cd90b11" },
      { name: "kokoro/G2PEncoder.mlmodelc/metadata.json", bytes: 2_034, sha256: "c8e0cfd7f494ac1b3662ff8f1914b2b45f79ffb2791724cdc0576981996732e1" },
      { name: "kokoro/G2PEncoder.mlmodelc/model.mil", bytes: 20_392, sha256: "8c617e569f37286b056dad800d862dc145be9a95fa9ed43857bb646ba199d7da" },
      { name: "kokoro/G2PEncoder.mlmodelc/analytics/coremldata.bin", bytes: 243, sha256: "cf7fbd7e7a65529b2d2bf3941e458a0ab6dff7a298bf48e205a1727c81c26a99" },
      { name: "kokoro/G2PEncoder.mlmodelc/weights/weight.bin", bytes: 694_592, sha256: "6926bcd2827d21fec82839487b987e06f85fd8a6a5bb896bc4f6062461d014ec" },
    ],
    attribution: [],
  },
  "phonon-31m": {
    name: "phonon-31m-en-8c64b55e",
    note: "Phonon 31M English, gradium/phonon-31m-en@8c64b55e (the ptts 0.6.0 default), default voice, 24000 Hz",
    family: "worker",
    sampleRate: 24_000,
    numSpeakers: 1,
    hub: { repo: "gradium/phonon-31m-en", revision: "8c64b55ec301728e2d1695bfea87a08a267d0c2d" },
    files: [
      { name: "config.json", bytes: 2_851, sha256: "7062ea376e6b158543c8abec8e5af67df171c49ad70eb7e2a0fb25c16f265e06" },
      { name: "model.safetensors", bytes: 102_823_920, sha256: "8a04cf833cfe69b67c2ec968b2d31244655b9accc84ade9c0ebcc176f2074607" },
      { name: "tokenizer.json", bytes: 277_967, sha256: "691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936" },
      { name: "default-voice.safetensors", bytes: 256_136, sha256: "f739e8c4c76ee7f92e7c94a82002748c8cd22e99582ac49f7b9b94cd63786e18" },
      { name: "README.md", bytes: 6_604, sha256: "232a0ab89f323cc8e003719c32ce72b6b2ce07de0c42caf18fa2f0ae71397508" },
    ],
    attribution: ["README.md"],
  },
} as const satisfies Record<string, TtsModel | WorkerTtsModel>;

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

/** The sherpa descriptor for a model id; worker voices never load through sherpa. */
export function getTtsModel(id: SynthModelId): TtsModel {
  if (!Object.hasOwn(LOCAL_TTS_MODELS, id)) return TTS_MODELS[id as TtsModelId];
  const model: TtsModel | WorkerTtsModel = LOCAL_TTS_MODELS[id as LocalTtsModelId];
  if (model.family === "worker") throw new Error(`${id} runs in a worker runtime, not sherpa`);
  return model;
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
