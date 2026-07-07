import { EventEmitter } from "node:events";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import llamaTokenizer from "llama-tokenizer-js";
import type { STTConfig, STTSession, VoiceToText } from "../contract.ts";
import { downloadFile, hasNonEmptyFile } from "./assets.ts";

/**
 * MoonshineEngine: a local-first Moonshine tiny runtime backed by
 * onnxruntime-node. It downloads Moonshine's quantized tiny ONNX weights into
 * ./models on first use, caches the loaded sessions process-wide, emits a
 * speculative partial from accumulated streaming audio, and emits endpoint only
 * when file input ends so endpoint->final measures model finalization work.
 */

const MODEL_BASE_URL = "https://download.moonshine.ai/model/tiny/quantized";
const MODEL_DIR = join(process.cwd(), "models", "moonshine", "tiny", "quantized");
const ENCODER_MODEL = "encoder_model.onnx";
const DECODER_MODEL = "decoder_model_merged.onnx";
const SAMPLE_RATE = 16_000;
const PARTIAL_AFTER_SAMPLES = SAMPLE_RATE;
const DECODER_START_TOKEN_ID = 1;
const EOS_TOKEN_ID = 2;
const TOKENS_PER_SECOND = 6;
const MIN_TOKENS = 4;

type MoonshineShape = {
  numLayers: number;
  numKVHeads: number;
  headDim: number;
};

type MoonshineRuntime = {
  encoder: ort.InferenceSession;
  decoder: ort.InferenceSession;
  provider: "coreml" | "cpu";
};

type PastKeyValues = Record<string, ort.Tensor>;

const TINY_SHAPE: MoonshineShape = {
  numLayers: 6,
  numKVHeads: 8,
  headDim: 36,
};

let runtimePromise: Promise<MoonshineRuntime> | undefined;

export class MoonshineEngine implements VoiceToText {
  #provider: MoonshineRuntime["provider"] | undefined;

  get label(): string {
    return this.#provider ? `moonshine-${this.#provider}` : "moonshine";
  }

  async prepare(): Promise<void> {
    const runtime = await loadRuntime();
    this.#provider = runtime.provider;
  }

  async open(config?: STTConfig): Promise<STTSession> {
    if (config?.sampleRate && config.sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `MoonshineEngine expects ${SAMPLE_RATE} Hz frames, received ${config.sampleRate} Hz`,
      );
    }
    const runtime = await loadRuntime();
    this.#provider = runtime.provider;
    return new MoonshineSession(runtime);
  }
}

class MoonshineSession extends EventEmitter implements STTSession {
  #runtime: MoonshineRuntime;
  #chunks: Float32Array[] = [];
  #sampleCount = 0;
  #partialStarted = false;
  #ended = false;

  constructor(runtime: MoonshineRuntime) {
    super();
    this.#runtime = runtime;
  }

  pushAudio(frame: Float32Array): void {
    if (this.#ended) {
      throw new Error("Cannot push audio after end()");
    }
    const copy = new Float32Array(frame);
    this.#chunks.push(copy);
    this.#sampleCount += copy.length;

    if (!this.#partialStarted && this.#sampleCount >= PARTIAL_AFTER_SAMPLES) {
      this.#partialStarted = true;
      const snapshot = concatFrames(this.#chunks, this.#sampleCount);
      void this.#emitPartial(snapshot);
    }
  }

  async end(): Promise<void> {
    if (this.#ended) {
      return;
    }
    this.#ended = true;
    await this.#commit(true);
  }

  flush(): void {
    void this.#commit(false);
  }

  reset(): void {
    this.#chunks = [];
    this.#sampleCount = 0;
    this.#partialStarted = false;
  }

  async #commit(emitEndpoint: boolean): Promise<void> {
    const audio = concatFrames(this.#chunks, this.#sampleCount);
    if (emitEndpoint) {
      this.emit("endpoint", {});
    }
    try {
      const text = await generate(this.#runtime, audio);
      this.emit("final", { text });
    } catch (err) {
      this.emit("error", { err });
      throw err;
    }
  }

  async #emitPartial(audio: Float32Array): Promise<void> {
    try {
      const text = await generate(this.#runtime, audio);
      this.emit("partial", { text });
    } catch (err) {
      this.emit("error", { err });
    }
  }
}

async function loadRuntime(): Promise<MoonshineRuntime> {
  runtimePromise ??= createRuntime();
  return runtimePromise;
}

async function createRuntime(): Promise<MoonshineRuntime> {
  await ensureModelFile(ENCODER_MODEL);
  await ensureModelFile(DECODER_MODEL);

  const encoderPath = join(MODEL_DIR, ENCODER_MODEL);
  const decoderPath = join(MODEL_DIR, DECODER_MODEL);

  try {
    return await createRuntimeWithProvider("coreml", encoderPath, decoderPath);
  } catch (coreMlError) {
    console.warn(
      `Moonshine CoreML provider unavailable, falling back to CPU: ${formatError(coreMlError)}`,
    );
    return createRuntimeWithProvider("cpu", encoderPath, decoderPath);
  }
}

async function createRuntimeWithProvider(
  provider: "coreml" | "cpu",
  encoderPath: string,
  decoderPath: string,
): Promise<MoonshineRuntime> {
  const options = createSessionOptions(provider);
  const encoder = await ort.InferenceSession.create(encoderPath, options);
  const decoder = await ort.InferenceSession.create(decoderPath, options);
  return { encoder, decoder, provider };
}

function createSessionOptions(
  provider: "coreml" | "cpu",
): ort.InferenceSession.SessionOptions {
  return {
    executionProviders:
      provider === "coreml"
        ? [{ name: "coreml", coreMlFlags: 0x002 | 0x020 }, "cpu"]
        : ["cpu"],
    graphOptimizationLevel: "all",
    executionMode: "sequential",
    logSeverityLevel: 3,
  };
}

async function ensureModelFile(filename: string): Promise<void> {
  const destination = join(MODEL_DIR, filename);
  if (await hasNonEmptyFile(destination)) {
    return;
  }

  const url = `${MODEL_BASE_URL}/${filename}`;
  await downloadFile(url, destination);
}

async function generate(
  runtime: MoonshineRuntime,
  audio: Float32Array,
): Promise<string> {
  const maxLen = Math.max(
    MIN_TOKENS,
    Math.trunc((audio.length / SAMPLE_RATE) * TOKENS_PER_SECOND),
  );
  const encoderOutput = await runtime.encoder.run({
    input_values: new ort.Tensor("float32", audio, [1, audio.length]),
  });
  const encoderHiddenStates = encoderOutput.last_hidden_state;
  if (!encoderHiddenStates) {
    throw new Error("Moonshine encoder did not return last_hidden_state");
  }

  const pastKeyValues = createPastKeyValues(TINY_SHAPE);
  const tokens = [DECODER_START_TOKEN_ID];
  let inputIds = [DECODER_START_TOKEN_ID];

  for (let index = 0; index < maxLen; index += 1) {
    const decoderInput: ort.InferenceSession.FeedsType = {
      input_ids: tensorInt64(inputIds),
      encoder_hidden_states: encoderHiddenStates,
      use_cache_branch: new ort.Tensor("bool", [index > 0], [1]),
      ...pastKeyValues,
    };

    const decoderOutput = await runtime.decoder.run(decoderInput);
    const logits = decoderOutput.logits;
    if (!(logits instanceof ort.Tensor)) {
      throw new Error("Moonshine decoder did not return logits");
    }
    const nextToken = argMax(logits.data as Float32Array);
    tokens.push(nextToken);

    if (nextToken === EOS_TOKEN_ID) {
      break;
    }

    inputIds = [nextToken];
    updatePastKeyValues(pastKeyValues, decoderOutput, index > 0);
  }

  return llamaTokenizer.decode(tokens.slice(0, -1)).trim();
}

function createPastKeyValues(shape: MoonshineShape): PastKeyValues {
  const pastKeyValues: PastKeyValues = {};
  for (let layer = 0; layer < shape.numLayers; layer += 1) {
    for (const source of ["decoder", "encoder"] as const) {
      for (const field of ["key", "value"] as const) {
        pastKeyValues[`past_key_values.${layer}.${source}.${field}`] =
          new ort.Tensor("float32", new Float32Array(0), [
            0,
            shape.numKVHeads,
            1,
            shape.headDim,
          ]);
      }
    }
  }
  return pastKeyValues;
}

function updatePastKeyValues(
  pastKeyValues: PastKeyValues,
  decoderOutput: ort.InferenceSession.ReturnType,
  useCache: boolean,
): void {
  const presentValues = Object.entries(decoderOutput)
    .filter(([key]) => key.includes("present"))
    .map(([, value]) => value);

  Object.keys(pastKeyValues).forEach((key, index) => {
    const value = presentValues[index];
    if (value instanceof ort.Tensor && (!useCache || key.includes("decoder"))) {
      pastKeyValues[key] = value;
    }
  });
}

function tensorInt64(values: number[]): ort.Tensor {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values.map((value) => BigInt(value))),
    [1, values.length],
  );
}

function argMax(array: Float32Array): number {
  if (array.length === 0) {
    throw new Error("Cannot argMax an empty logits tensor");
  }
  let maxIndex = 0;
  let maxValue = array[0]!;
  for (let index = 1; index < array.length; index += 1) {
    const value = array[index]!;
    if (value > maxValue) {
      maxValue = value;
      maxIndex = index;
    }
  }
  return maxIndex;
}

function concatFrames(
  frames: Float32Array[],
  sampleCount: number,
): Float32Array {
  const audio = new Float32Array(sampleCount);
  let offset = 0;
  for (const frame of frames) {
    audio.set(frame, offset);
    offset += frame.length;
  }
  return audio;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
