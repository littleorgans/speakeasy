import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { loadHotwords, type HotwordFileSnapshot } from "@speakeasy/speech-io";
import {
  isAssetComplete, resolveModelPaths, sha256File, SHERPA_MODELS,
  type AnySherpaEngine, type FinalEvent, type Hotwords, type SherpaBinding,
} from "@speakeasy/speech-io/stt";
import { prepareOfflineHotwords } from "../../speech-io/src/engines/offline-hotwords.ts";
import type { OfflineRecognizerConfig, OfflineStream } from "../../speech-io/src/engines/sherpa-api.ts";
import { PROMPT_SAMPLE_RATE } from "./prompt-reader.ts";
import {
  CPU_PRESET, TRANSCRIBE_MODEL_ID, TranscriptionError, validateCpuSamples,
  type CpuLoadMetadata, type TranscriptionMode, type Transcript,
} from "./transcriber.ts";

/** The comparison list is independent of every editable reader file. */
export function fixedCpuSnapshot(): HotwordFileSnapshot {
  const phrases = ["launch", "claude", "grok"];
  Object.freeze(phrases);
  return Object.freeze({ source: CPU_PRESET.mode, content: `${phrases.join("\n")}\n`,
    hotwords: Object.freeze({ phrases, score: CPU_PRESET.hotwordsScore, file: CPU_PRESET.mode }) });
}

export type CpuMeasurements = {
  modelConstructionMs: number | null;
  processingMs: number | null;
  rawText?: string;
};

/** One file snapshot per request; one active recognizer and decode at a time. */
export class TakeTranscription {
  readonly #read: () => Promise<HotwordFileSnapshot>;
  readonly #createEngine: (hotwords: Hotwords | undefined) => Promise<AnySherpaEngine>;
  readonly #mode: TranscriptionMode;
  readonly #measurements: (() => CpuMeasurements) | undefined;
  #engine: AnySherpaEngine | undefined;
  #content: string | undefined;
  #queue: Promise<unknown> = Promise.resolve();
  #load: CpuLoadMetadata = { loadMs: 0, modelConstructionMs: null };
  #used = false;

  constructor(options: {
    hotwordsPath?: string;
    mode?: TranscriptionMode;
    readSnapshot?(): Promise<HotwordFileSnapshot>;
    measurements?(): CpuMeasurements;
    createEngine(hotwords: Hotwords | undefined): Promise<AnySherpaEngine>;
  }) {
    this.#mode = options.mode ?? "legacy";
    const fixed = fixedCpuSnapshot();
    this.#read = this.#mode === CPU_PRESET.mode ? async () => fixed
      : options.readSnapshot ?? (() => loadHotwords({ snapshot: true, source: options.hotwordsPath }));
    this.#createEngine = options.createEngine;
    this.#measurements = options.measurements;
  }

  /** Fixed ready means construction completed, with no warmup inference. */
  prepare(): Promise<CpuLoadMetadata> {
    const result = this.#queue.then(async () => {
      await this.#ensure(await this.#read());
      return { ...this.#load };
    });
    this.#queue = result.catch(() => {});
    return result;
  }

  transcribe(samples: Float32Array): Promise<Transcript> {
    if (this.#mode === CPU_PRESET.mode) {
      try { validateCpuSamples(samples); } catch (error) { return Promise.reject(error); }
    }
    // Read immediately, before an earlier decode can delay this take. Handle
    // read rejection now as well, so queued failures are never unhandled.
    const snapshot = this.#read().then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    );
    const result = this.#queue.then(async () => {
      const captured = await snapshot;
      if ("error" in captured) throw new TranscriptionError(String(captured.error), TRANSCRIBE_MODEL_ID);
      const { source, content, hotwords } = captured.value;
      const status = {
        path: source, count: hotwords?.phrases.length ?? 0,
        revision: createHash("sha256").update(content).digest("hex"),
      };
      try {
        const started = performance.now();
        const engine = await this.#ensure(captured.value);
        const firstUse = !this.#used;
        this.#used = true;
        const result = await decode(engine, samples);
        const measured = this.#measurements?.();
        return { ...result, model: TRANSCRIBE_MODEL_ID, hotwords: status,
          ...(this.#mode === CPU_PRESET.mode ? {
            text: measured?.rawText ?? result.text,
            comparison: { ...this.#load, processingMs: measured?.processingMs ?? null,
              firstUse, workerWallMs: performance.now() - started },
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
      if (this.#mode === CPU_PRESET.mode) {
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

/** Hash checks precede engine preparation, which must never install a model. */
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
  const prepared = prepareOfflineHotwords(model, table, fixedCpuSnapshot().hotwords!);
  if (createHash("sha256").update(prepared.vocabulary).digest("hex") !== CPU_PRESET.vocabularySha256) {
    throw new Error("CPU vocabulary does not match the measured preset");
  }
  return hashes;
}

export function validateCpuRuntime(binding: SherpaBinding & { gitSha1?: string; onnxruntimeVersion?: string }): void {
  const runtime = binding;
  if (runtime.version !== CPU_PRESET.runtimeVersion || runtime.gitSha1 !== CPU_PRESET.runtimeGitSha1 ||
      runtime.onnxruntimeVersion !== CPU_PRESET.onnxruntimeVersion) {
    throw new Error("CPU comparison requires the pinned sherpa 2.0.1 parent and ONNX Runtime 1.28.2");
  }
}

/** Instrument the shared engine's real native boundary without another decoder. */
export function instrumentCpuBinding(binding: SherpaBinding): { binding: SherpaBinding; measurements: CpuMeasurements } {
  const measurements: CpuMeasurements = { modelConstructionMs: null, processingMs: null };
  return { measurements, binding: { ...binding, OfflineRecognizer: class {
    readonly #native: InstanceType<SherpaBinding["OfflineRecognizer"]>;
    #started = 0;
    constructor(config: OfflineRecognizerConfig) {
      const started = performance.now();
      this.#native = new binding.OfflineRecognizer(config);
      measurements.modelConstructionMs = performance.now() - started;
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
