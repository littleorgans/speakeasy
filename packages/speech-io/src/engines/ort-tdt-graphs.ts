import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  frontendChunks,
  logMel,
  parseMelFilters,
  subsampledFrames,
  type LogMel,
} from "./ort-tdt-mel.ts";
import {
  parseTokenizerJson,
  parseVocabTxt,
  type TdtStepper,
  type Vocabulary,
} from "./ort-tdt-decode.ts";
import type { FusedTdtModel, OrtTdtModel, SplitTdtModel } from "./ort-tdt-models.ts";

/**
 * The ONNX graphs behind each TDT layout, over a narrow onnxruntime seam the
 * host injects (as SherpaBinding is for sherpa), so tests fake sessions and
 * the engine never imports the native package itself.
 *
 * Both layouts reduce audio to encoder frames, then hand the shared greedy
 * decoder a stepper:
 * - fused: one decoder_joint run per step from the previous token and LSTM
 *   state; its output state becomes current only when the step emits a token
 *   (onnx-asr `NemoConformerRnnt._decode`, Redux transcribe.py `greedy`).
 * - split: the decoder runs once per emitted token and the joint once per
 *   step (Fluid transcribe.py `greedy_tdt`).
 */

export type OrtTensorType = "float32" | "int32" | "int64";

export type OrtTensor = {
  readonly type: string;
  readonly data: unknown;
  readonly dims: readonly number[];
};

export interface OrtGraph {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
}

export type OrtBinding = {
  /** onnxruntime-node version, for engine labels. */
  version: string;
  createSession(path: string, options: { threads: number }): Promise<OrtGraph>;
  tensor(
    type: OrtTensorType,
    data: Float32Array | Int32Array | BigInt64Array,
    dims: readonly number[],
  ): OrtTensor;
};

/** Encoder threads; matches SherpaOfflineEngine's numThreads for bench parity. */
const ENCODER_THREADS = 2;
/** The per step graphs are small; the Fluid runner gives them one thread. */
const STEP_THREADS = 1;

export type EncodedAudio = {
  /** Row major [frames, dim]. */
  frames: Float32Array;
  count: number;
  dim: number;
};

export interface TdtGraphs {
  readonly vocabulary: Vocabulary;
  encode(audio: Float32Array): Promise<EncodedAudio>;
  stepper(encoded: EncodedAudio): TdtStepper;
  release(): Promise<void>;
}

export async function loadTdtGraphs(
  model: OrtTdtModel,
  dir: string,
  binding: OrtBinding,
): Promise<TdtGraphs> {
  return model.layout === "fused"
    ? loadFusedGraphs(model, dir, binding)
    : loadSplitGraphs(model, dir, binding);
}

async function loadFusedGraphs(
  model: FusedTdtModel,
  dir: string,
  { createSession, tensor }: OrtBinding,
): Promise<TdtGraphs> {
  const [vocabulary, preprocessor, encoder, decoderJoint] = await Promise.all([
    readFile(join(dir, model.vocab), "utf8").then(parseVocabTxt),
    createSession(join(dir, model.preprocessor), { threads: ENCODER_THREADS }),
    createSession(join(dir, model.encoder), { threads: ENCODER_THREADS }),
    createSession(join(dir, model.decoderJoint), { threads: STEP_THREADS }),
  ]);
  const { blankId, predictor } = model.decoding;
  const stateDims = [predictor.layers, 1, predictor.hidden];
  const zeros = () => tensor("float32", new Float32Array(predictor.layers * predictor.hidden), stateDims);
  return {
    vocabulary,
    encode: async (audio) => {
      const features = await preprocessor.run({
        waveforms: tensor("float32", audio, [1, audio.length]),
        waveforms_lens: tensor("int64", BigInt64Array.of(BigInt(audio.length)), [1]),
      });
      const encoded = await encoder.run({
        audio_signal: output(features, "features"),
        length: output(features, "features_lens"),
      });
      // outputs is channel major [1, dim, frames]; the decoder wants frames.
      const outputs = output(encoded, "outputs");
      const [, dim = 0, total = 0] = outputs.dims;
      const count = Math.min(total, scalar(output(encoded, "encoded_lengths")));
      return { frames: transpose(outputs.data as Float32Array, dim, total, count), count, dim };
    },
    stepper: (encoded) => {
      let state1 = zeros();
      let state2 = zeros();
      let previous = blankId;
      let pending: Record<string, OrtTensor> | undefined;
      return {
        joint: async (frame) => {
          pending = await decoderJoint.run({
            encoder_outputs: tensor("float32", frameAt(encoded, frame), [1, encoded.dim, 1]),
            targets: tensor("int32", Int32Array.of(previous), [1, 1]),
            target_length: tensor("int32", Int32Array.of(1), [1]),
            input_states_1: state1,
            input_states_2: state2,
          });
          return output(pending, "outputs").data as Float32Array;
        },
        emit: async (token) => {
          if (!pending) {
            throw new Error("emit() before joint()");
          }
          previous = token;
          state1 = output(pending, "output_states_1");
          state2 = output(pending, "output_states_2");
        },
      };
    },
    release: () => releaseAll(preprocessor, encoder, decoderJoint),
  };
}

async function loadSplitGraphs(
  model: SplitTdtModel,
  dir: string,
  { createSession, tensor }: OrtBinding,
): Promise<TdtGraphs> {
  const [vocabulary, filters, frontend, encoder, decoder, joint] = await Promise.all([
    readFile(join(dir, model.tokenizer), "utf8").then(parseTokenizerJson),
    readFile(join(dir, model.melFilters)).then(parseMelFilters),
    createSession(join(dir, model.frontend), { threads: ENCODER_THREADS }),
    createSession(join(dir, model.encoder), { threads: ENCODER_THREADS }),
    createSession(join(dir, model.decoder), { threads: STEP_THREADS }),
    createSession(join(dir, model.joint), { threads: STEP_THREADS }),
  ]);
  const { blankId, predictor } = model.decoding;
  const predict = (token: number, h: OrtTensor, c: OrtTensor) =>
    decoder.run({
      token: tensor("int64", BigInt64Array.of(BigInt(token)), [1, 1]),
      h_in: h,
      c_in: c,
    });

  async function frontendFrames(mel: LogMel): Promise<{ hidden: Float32Array; dim: number }> {
    const bands = filters.bands;
    const parts: Float32Array[] = [];
    let dim = 0;
    for (const chunk of frontendChunks(mel.frames, model.frontendChunkFrames, model.subsampling)) {
      const length = chunk.end - chunk.start;
      const mask = new BigInt64Array(length);
      for (let index = 0; index < length; index += 1) {
        mask[index] = chunk.start + index < mel.valid ? 1n : 0n;
      }
      const result = await frontend.run({
        input_features: tensor(
          "float32",
          mel.features.subarray(chunk.start * bands, chunk.end * bands),
          [1, length, bands],
        ),
        attention_mask: tensor("int64", mask, [1, length]),
      });
      const hidden = output(result, "hidden_states");
      dim = hidden.dims[2] ?? 0;
      const data = hidden.data as Float32Array;
      parts.push(data.subarray(chunk.skip * dim, (chunk.skip + chunk.count) * dim));
    }
    return { hidden: concat(parts), dim };
  }

  return {
    vocabulary,
    encode: async (audio) => {
      const mel = logMel(audio, model.mel, filters);
      const { hidden, dim } = await frontendFrames(mel);
      const total = dim ? hidden.length / dim : 0;
      const valid = subsampledFrames(mel.valid, model.subsampling);
      const encoded = await encoder.run({
        hidden_states: tensor("float32", hidden, [1, total, dim]),
        length: tensor("int64", BigInt64Array.of(BigInt(valid)), [1]),
      });
      const frames = output(encoded, "encoder_out").data as Float32Array;
      return { frames, count: Math.min(valid, total), dim };
    },
    stepper: (encoded) => {
      const zeros = tensor(
        "float32",
        new Float32Array(predictor.layers * predictor.hidden),
        [predictor.layers, 1, predictor.hidden],
      );
      let state = predict(blankId, zeros, zeros);
      return {
        joint: async (frame) => {
          const result = await joint.run({
            encoder_frame: tensor("float32", frameAt(encoded, frame), [1, encoded.dim]),
            decoder_out: output(await state, "decoder_out"),
          });
          return output(result, "logits").data as Float32Array;
        },
        emit: async (token) => {
          const previous = await state;
          state = predict(token, output(previous, "h_out"), output(previous, "c_out"));
          await state;
        },
      };
    },
    release: () => releaseAll(frontend, encoder, decoder, joint),
  };
}

async function releaseAll(...graphs: OrtGraph[]): Promise<void> {
  await Promise.all(graphs.map((graph) => graph.release()));
}

function output(outputs: Record<string, OrtTensor>, name: string): OrtTensor {
  const value = outputs[name];
  if (!value) {
    throw new Error(`ONNX graph returned no "${name}" output`);
  }
  return value;
}

function scalar(value: OrtTensor): number {
  const data = value.data as ArrayLike<number | bigint>;
  return Number(data[0] ?? 0);
}

function frameAt(encoded: EncodedAudio, frame: number): Float32Array {
  return encoded.frames.subarray(frame * encoded.dim, (frame + 1) * encoded.dim);
}

/** [dim, total] channel major to [count, dim] frame major. */
function transpose(data: Float32Array, dim: number, total: number, count: number): Float32Array {
  const out = new Float32Array(count * dim);
  for (let channel = 0; channel < dim; channel += 1) {
    for (let frame = 0; frame < count; frame += 1) {
      out[frame * dim + channel] = data[channel * total + frame]!;
    }
  }
  return out;
}

function concat(parts: readonly Float32Array[]): Float32Array {
  if (parts.length === 1) {
    return parts[0]!;
  }
  const out = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}
