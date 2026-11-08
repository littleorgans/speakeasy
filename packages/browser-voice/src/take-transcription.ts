import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import type { HotwordFileSnapshot } from "@speakeasy/speech-io";
import {
  isAssetComplete, resolveModelPaths, sha256File, SHERPA_MODELS,
  type AnySherpaEngine, type FinalEvent, type Hotwords, type SherpaBinding,
} from "@speakeasy/speech-io/stt";
import { prepareOfflineHotwords } from "../../speech-io/src/engines/offline-hotwords.ts";
import type { OfflineRecognizerConfig, OfflineStream } from "../../speech-io/src/engines/sherpa-api.ts";
import { PROMPT_SAMPLE_RATE } from "./prompt-reader.ts";
import {
  CPU_DECODING, cpuNames, NAMES_SOURCE, namesContent, TRANSCRIBE_MODEL_ID, TranscriptionError, validateCpuSamples,
  type CpuLoadMetadata, type TranscriptionMode, type Transcript,
} from "./transcriber.ts";

/** One hotword per name; an empty list has no hotwords. */
export function namesSnapshot(names: readonly string[]): HotwordFileSnapshot {
  const phrases = [...cpuNames(names)];
  Object.freeze(phrases);
  return Object.freeze({ source: NAMES_SOURCE, content: namesContent(phrases), hotwords: phrases.length
    ? Object.freeze({ phrases, score: CPU_DECODING.hotwordsScore, file: NAMES_SOURCE }) : undefined });
}

export type CpuMeasurements = {
  modelConstructionMs: number | null;
  processingMs: number | null;
  rawText?: string;
  /** What the native recognizer was constructed with. */
  decodingMethod?: string;
};

/** One names snapshot per request; one active recognizer and decode at a time. */
export class TakeTranscription {
  readonly #createEngine: (hotwords: Hotwords | undefined) => Promise<AnySherpaEngine>;
  readonly #fixed: boolean;
  readonly #measurements: (() => CpuMeasurements) | undefined;
  #engine: AnySherpaEngine | undefined;
  #content: string | undefined;
  #queue: Promise<unknown> = Promise.resolve();
  #load: CpuLoadMetadata = { loadMs: 0, modelConstructionMs: null };
  #used = false;

  constructor(options: {
    mode?: TranscriptionMode;
    measurements?(): CpuMeasurements;
    createEngine(hotwords: Hotwords | undefined): Promise<AnySherpaEngine>;
  }) {
    this.#fixed = (options.mode ?? "legacy") !== "legacy";
    this.#createEngine = options.createEngine;
    this.#measurements = options.measurements;
  }

  /**
   * Fixed ready means construction completed for these names, with no warmup
   * inference; other names than the last rebuild the recognizer here.
   */
  prepare(names: readonly string[]): Promise<CpuLoadMetadata> {
    let snapshot: HotwordFileSnapshot;
    try { snapshot = namesSnapshot(names); } catch (error) { return Promise.reject(error); }
    const result = this.#queue.then(async () => {
      await this.#ensure(snapshot);
      return { ...this.#load };
    });
    this.#queue = result.catch(() => {});
    return result;
  }

  transcribe(samples: Float32Array, names: readonly string[]): Promise<Transcript> {
    let snapshot: HotwordFileSnapshot;
    try {
      if (this.#fixed) validateCpuSamples(samples);
      // Taken now, so a take waiting behind an earlier decode keeps its own names.
      snapshot = namesSnapshot(names);
    } catch (error) { return Promise.reject(error); }
    const result = this.#queue.then(async () => {
      const { source, content, hotwords } = snapshot;
      const status = {
        path: source, count: hotwords?.phrases.length ?? 0,
        revision: createHash("sha256").update(content).digest("hex"),
      };
      try {
        const started = performance.now();
        const engine = await this.#ensure(snapshot);
        const firstUse = !this.#used;
        this.#used = true;
        const result = await decode(engine, samples);
        const measured = this.#measurements?.();
        return { ...result, model: TRANSCRIBE_MODEL_ID, hotwords: status,
          ...(this.#fixed ? {
            text: measured?.rawText ?? result.text,
            comparison: { ...this.#load, processingMs: measured?.processingMs ?? null,
              firstUse, workerWallMs: performance.now() - started,
              ...(measured?.decodingMethod === undefined ? {} : { decodingMethod: measured.decodingMethod }) },
          } : {}) };
      } catch (error) {
        throw new TranscriptionError(error instanceof Error ? error.message : String(error), TRANSCRIBE_MODEL_ID, status);
      }
    });
    this.#queue = result.catch(() => {});
    return result;
  }

  async #ensure({ content, hotwords }: HotwordFileSnapshot): Promise<AnySherpaEngine> {
    if (!this.#engine || this.#content !== content) {
      this.#engine = undefined;
      this.#content = undefined;
      this.#used = false;
      const started = performance.now();
      const engine = await this.#createEngine(hotwords);
      await engine.prepare();
      if (this.#fixed) {
        // Sherpa constructs lazily in open(), not prepare(). No audio or flush.
        const session = await engine.open({ sampleRate: PROMPT_SAMPLE_RATE });
        await session.end();
      }
      this.#load = { loadMs: performance.now() - started,
        modelConstructionMs: this.#measurements?.().modelConstructionMs ?? null };
      this.#engine = engine;
      this.#content = content;
    }
    return this.#engine;
  }
}

/**
 * Hash checks precede engine preparation, which must never install a model.
 * They cover the assets alone; each recognizer build validates its names.
 */
export async function preflightCpuModel(modelRoot: string): Promise<Record<string, string>> {
  const model = SHERPA_MODELS[TRANSCRIBE_MODEL_ID];
  const paths = resolveModelPaths(model, modelRoot);
  if (!(await isAssetComplete(paths))) throw new Error("CPU model is not installed with its pinned completion marker");
  const hashes: Record<string, string> = {};
  // Reject small corrupt assets before streaming the large encoder hash.
  for (const file of [...model.files].sort((a, b) => a.bytes - b.bytes)) {
    const path = join(paths.dir, file.name);
    if ((await stat(path)).size !== file.bytes || (await sha256File(path)) !== file.sha256) {
      throw new Error(`CPU model asset mismatch: ${file.name}`);
    }
    hashes[file.name] = file.sha256;
  }
  const table = new TextDecoder("utf-8", { fatal: true }).decode(await readFile(join(paths.dir, model.tokens)));
  // The vocabulary derives from the token table alone, so no names are needed to check it.
  const prepared = prepareOfflineHotwords(model, table, { phrases: [], score: CPU_DECODING.hotwordsScore });
  if (createHash("sha256").update(prepared.vocabulary).digest("hex") !== CPU_DECODING.vocabularySha256) {
    throw new Error("CPU vocabulary does not match the measured preset");
  }
  return hashes;
}

export function validateCpuRuntime(binding: SherpaBinding & { gitSha1?: string; onnxruntimeVersion?: string }): void {
  const runtime = binding;
  if (runtime.version !== CPU_DECODING.runtimeVersion || runtime.gitSha1 !== CPU_DECODING.runtimeGitSha1 ||
      runtime.onnxruntimeVersion !== CPU_DECODING.onnxruntimeVersion) {
    throw new Error("CPU comparison requires the pinned sherpa 2.0.1 parent and ONNX Runtime 1.28.2");
  }
}

/**
 * Instrument the shared engine's real native boundary without another decoder.
 * The shared engine decodes greedily when a list is empty; the CPU presets pin
 * one decoding for every list, so each recognizer is constructed with it.
 */
export function instrumentCpuBinding(binding: SherpaBinding): { binding: SherpaBinding; measurements: CpuMeasurements } {
  const measurements: CpuMeasurements = { modelConstructionMs: null, processingMs: null };
  return { measurements, binding: { ...binding, OfflineRecognizer: class {
    readonly #native: InstanceType<SherpaBinding["OfflineRecognizer"]>;
    #started = 0;
    constructor(config: OfflineRecognizerConfig) {
      const pinned = { ...config, decodingMethod: CPU_DECODING.decodingMethod, maxActivePaths: CPU_DECODING.maxActivePaths };
      const started = performance.now();
      this.#native = new binding.OfflineRecognizer(pinned);
      measurements.modelConstructionMs = performance.now() - started;
      measurements.decodingMethod = pinned.decodingMethod;
    }
    createStream(): OfflineStream {
      measurements.processingMs = null;
      delete measurements.rawText;
      this.#started = performance.now();
      return this.#native.createStream();
    }
    async decodeAsync(stream: OfflineStream) {
      const result = await this.#native.decodeAsync(stream);
      measurements.processingMs = performance.now() - this.#started;
      if (typeof result.text !== "string") throw new Error("CPU native result is missing text");
      measurements.rawText = result.text;
      return result;
    }
  } } };
}

async function decode(engine: AnySherpaEngine, samples: Float32Array): Promise<{ text: string; decodeMs: number }> {
  const session = await engine.open({ sampleRate: PROMPT_SAMPLE_RATE });
  try {
    return await new Promise((resolve, reject) => {
      session.once("final", ({ text, decodeMs }: FinalEvent) => resolve({ text, decodeMs: decodeMs ?? 0 }));
      session.once("empty", ({ decodeMs }: { decodeMs?: number }) => resolve({ text: "", decodeMs: decodeMs ?? 0 }));
      session.once("error", ({ err }: { err: unknown }) => reject(err));
      session.pushAudio(samples);
      session.flush();
    });
  } finally {
    await session.end();
  }
}
