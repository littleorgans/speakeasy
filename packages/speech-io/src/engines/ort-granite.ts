import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { ensureAsset, type FetchArchive, type FileSetSpec } from "./assets.ts";
import { ByteLevelVocab, greedyCtc } from "./ctc-decode.ts";
import {
  GRANITE_FEATURE_DIM,
  GRANITE_FEATURE_FRAME_MS,
  GRANITE_MIN_FEATURE_FRAMES,
  graniteFeatureFrames,
  graniteFeatures,
} from "./granite-frontend.ts";
import { OfflineLifecycle } from "./offline-engine.ts";
import type { SilenceGate } from "./offline-session.ts";
import type { CtcGraph, CtcRuntime } from "./ort-ctc-runtime.ts";
import {
  resolveGraniteFiles,
  resolveGraniteModel,
  type GraniteCtcModel,
  type GraniteModelId,
} from "./ort-granite-models.ts";
import { ortThreads } from "./ort-models.ts";
import { resultDetail, type ResultDetail } from "./result.ts";

/**
 * OrtGraniteEngine: push to talk STT for Granite Speech CTC exports on
 * onnxruntime-node. Same session contract as `SherpaOfflineEngine` (buffer
 * while held, decode on flush, `final` or `empty`, `maxAudioMs` cap).
 *
 * The graph takes stacked log mel features and returns per frame logits; the
 * frontend (granite-frontend.ts) and greedy CTC (ctc-decode.ts) run in
 * JavaScript. Finals carry one token per emitted CTC token with its first
 * frame's time and log probability. A hold shorter than the graph's minimum
 * input decodes to empty without running the graph.
 */

/** Output frames per feature frame: the graph's `floor(frames / 4)`. */
const GRAPH_SUBSAMPLING = 4;
const OUTPUT_FRAME_SECONDS = (GRAPH_SUBSAMPLING * GRANITE_FEATURE_FRAME_MS) / 1_000;

export type OrtGraniteEngineOptions = {
  model: GraniteModelId;
  /** Folder holding downloaded model directories. */
  modelRoot: string;
  runtime: CtcRuntime;
  /** Tests inject a fake download. */
  fetchFile?: FetchArchive;
  /** The silence gate every session runs before decode. */
  silenceGate?: SilenceGate;
};

type LoadedModel = { graph: CtcGraph; vocab: ByteLevelVocab };

export class OrtGraniteEngine implements VoiceToText {
  readonly model: GraniteCtcModel;
  readonly #files: FileSetSpec;
  readonly #runtime: CtcRuntime;
  readonly #lifecycle: OfflineLifecycle<void, LoadedModel>;

  constructor(options: OrtGraniteEngineOptions) {
    this.model = resolveGraniteModel(options.model);
    this.#files = resolveGraniteFiles(this.model, options.modelRoot);
    this.#runtime = options.runtime;
    // One graph serves every session; a failed load is retried on next use.
    this.#lifecycle = new OfflineLifecycle({
      name: "OrtGraniteEngine",
      maxAudioMs: this.model.maxAudioMs,
      silenceGate: options.silenceGate,
      prepare: "load",
      preflight: () => ensureAsset(this.#files, options.fetchFile),
      load: () => this.#load(),
      release: ({ graph }) => graph.release(),
      decoder: (loaded) => (audio) => transcribe(loaded, audio),
    });
  }

  get label(): string {
    return `onnxruntime-node-${this.#runtime.version}:${this.model.name}`;
  }

  get threads(): number {
    return ortThreads(this.model);
  }

  prepare(): Promise<void> {
    return this.#lifecycle.prepare();
  }

  open(config?: STTConfig): Promise<STTSession> {
    return this.#lifecycle.open(config);
  }

  /** Free the native graph; a later open() loads it again. */
  release(): Promise<void> {
    return this.#lifecycle.release();
  }

  async #load(): Promise<LoadedModel> {
    const vocab = ByteLevelVocab.fromTokenizerJson(
      await readFile(join(this.#files.dir, this.model.tokenizer.name), "utf8"),
      this.model.blankToken,
    );
    const graph = await this.#runtime.loadGraph(join(this.#files.dir, this.model.graph.name), {
      threads: this.threads,
    });
    return { graph, vocab };
  }
}

async function transcribe(
  { graph, vocab }: LoadedModel,
  audio: Float32Array,
): Promise<ResultDetail> {
  if (graniteFeatureFrames(audio.length) < GRANITE_MIN_FEATURE_FRAMES) {
    return { text: "", tokens: [] };
  }
  const features = graniteFeatures(audio);
  const logits = await graph.run(features.data, features.frames, GRANITE_FEATURE_DIM);
  return resultDetail(greedyCtc(logits, vocab, OUTPUT_FRAME_SECONDS));
}
