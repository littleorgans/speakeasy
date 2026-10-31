import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { ensureAsset, type FileSetSpec } from "./assets.ts";
import { BufferedOfflineSession, type SilenceGate } from "./offline-session.ts";
import { greedyTdt, tdtResult } from "./ort-tdt-decode.ts";
import { loadTdtGraphs, type OrtBinding, type TdtGraphs } from "./ort-tdt-graphs.ts";
import {
  resolveOrtTdtModel,
  resolveOrtTdtPaths,
  type OrtTdtModel,
  type OrtTdtModelId,
} from "./ort-tdt-models.ts";
import { ortThreads } from "./ort-models.ts";
import { resultDetail, type ResultDetail } from "./result.ts";
import { assertSampleRate, SHERPA_SAMPLE_RATE } from "./sherpa.ts";

/**
 * OrtTdtEngine: push to talk STT for Parakeet TDT exports on onnxruntime-node
 * (Fluid Speech Mini, Phonon-2, Parakeet Redux), with the same session
 * behaviour as SherpaOfflineEngine: frames buffer while held up to the
 * model's `maxAudioMs`, decode once on flush(), and answer `final` with text
 * and per token timing and log probabilities, or `empty`.
 */

/** Two STFT hops: the shortest audio both reference runners transcribe. */
const MIN_DECODE_SAMPLES = 320;

export type OrtTdtEngineOptions = {
  model: OrtTdtModelId;
  /** Folder holding downloaded models, one directory each. */
  modelRoot: string;
  binding: OrtBinding;
  /** The silence gate every session runs before decode. */
  silenceGate?: SilenceGate;
};

export class OrtTdtEngine implements VoiceToText {
  readonly model: OrtTdtModel;
  readonly paths: FileSetSpec;
  readonly #binding: OrtBinding;
  readonly #silenceGate: SilenceGate | undefined;
  #graphs: Promise<TdtGraphs> | undefined;

  constructor(options: OrtTdtEngineOptions) {
    this.model = resolveOrtTdtModel(options.model);
    this.paths = resolveOrtTdtPaths(this.model, options.modelRoot);
    this.#binding = options.binding;
    this.#silenceGate = options.silenceGate;
  }

  get label(): string {
    return `onnxruntime-node-${this.#binding.version}:${this.model.name}`;
  }

  get threads(): number {
    return ortThreads(this.model);
  }

  async prepare(): Promise<void> {
    await ensureAsset(this.paths);
  }

  async open(config?: STTConfig): Promise<STTSession> {
    assertSampleRate("OrtTdtEngine", config);
    const graphs = await this.#load();
    const maxSamples = (this.model.maxAudioMs * SHERPA_SAMPLE_RATE) / 1_000;
    return new BufferedOfflineSession(
      maxSamples,
      (audio) => this.#decode(graphs, audio),
      this.#silenceGate,
    );
  }

  /** Free the native sessions; a later open() loads them again. */
  async release(): Promise<void> {
    const graphs = this.#graphs;
    this.#graphs = undefined;
    await (await graphs)?.release();
  }

  #load(): Promise<TdtGraphs> {
    // One set of graphs serves every session; a failed load is retried.
    this.#graphs ??= this.prepare()
      .then(() => loadTdtGraphs(this.model, this.paths.dir, this.#binding, this.threads))
      .catch((err: unknown) => {
        this.#graphs = undefined;
        throw err;
      });
    return this.#graphs;
  }

  async #decode(graphs: TdtGraphs, audio: Float32Array): Promise<ResultDetail> {
    if (audio.length < MIN_DECODE_SAMPLES) {
      return { text: "", tokens: [] };
    }
    const { decoding } = this.model;
    const encoded = await graphs.encode(audio);
    const tokens = await greedyTdt(graphs.stepper(encoded), encoded.count, decoding);
    return resultDetail(tdtResult(tokens, graphs.vocabulary, decoding.frameMs));
  }
}
