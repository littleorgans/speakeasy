import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { after, test } from "node:test";
import type { EmptyEvent, FinalEvent } from "../contract.ts";
import { COMPLETE_MARKER, completeMarkerValue } from "./assets.ts";
import type { OrtBinding, OrtGraph, OrtTensor } from "./ort-tdt-graphs.ts";
import { OrtTdtEngine } from "./ort-tdt.ts";
import {
  ORT_TDT_MODELS,
  resolveOrtTdtPaths,
  type OrtTdtModelId,
} from "./ort-tdt-models.ts";

/**
 * OrtTdtEngine over fake onnxruntime sessions: each fake graph answers with
 * the tensor names and shapes of the real export and records its feeds, and
 * the decode step graphs replay scripted logits.
 */

const VOCAB = 8_193;
const BLANK = 8_192;
const DIM = 4;
const PIECES: Record<number, string> = { 10: "▁Open", 11: "▁the", 12: "▁file", 13: "." };

type Feeds = Record<string, OrtTensor>;
type Fake = {
  runs: Record<string, Feeds[]>;
  steps: Float32Array[];
  created: string[];
  failure?: Error;
};

function tensor(type: string, data: unknown, dims: readonly number[]): OrtTensor {
  return { type, data, dims };
}

/** Logits picking `token` with the duration at `durationIndex`. */
function step(token: number, durationIndex: number): Float32Array {
  const logits = new Float32Array(VOCAB + 5);
  logits[token] = 10;
  logits[VOCAB + durationIndex] = 10;
  return logits;
}

function nextStep(fake: Fake): Float32Array {
  if (fake.failure) throw fake.failure;
  return fake.steps.shift() ?? step(BLANK, 4);
}

/** One fake per graph file, named by the descriptor's file roles. */
function fakeGraph(file: string, fake: Fake): (feeds: Feeds) => Feeds {
  const frames = (samples: number) => Math.floor(samples / 160) + 1;
  const subsample = (count: number) => Math.floor((count - 1) / 8) + 1;
  const graphs: Record<string, (feeds: Feeds) => Feeds> = {
    preprocessor: (feeds) => {
      const count = frames(feeds.waveforms!.dims[1]!);
      return {
        features: tensor("float32", new Float32Array(128 * count), [1, 128, count]),
        features_lens: tensor("int64", BigInt64Array.of(BigInt(count)), [1]),
      };
    },
    fusedEncoder: (feeds) => {
      const count = subsample(feeds.audio_signal!.dims[2]!);
      // Channel major: channel c of frame t holds t + c / 10.
      const data = Float32Array.from({ length: DIM * count }, (_, i) => (i % count) + Math.floor(i / count) / 10);
      return {
        outputs: tensor("float32", data, [1, DIM, count]),
        encoded_lengths: tensor("int64", BigInt64Array.of(BigInt(count)), [1]),
      };
    },
    decoderJoint: () => {
      const index = fake.runs.decoderJoint!.length;
      return {
        outputs: tensor("float32", nextStep(fake), [1, 1, 1, VOCAB + 5]),
        output_states_1: tensor("float32", Float32Array.of(index), [2, 1, 640]),
        output_states_2: tensor("float32", Float32Array.of(-index), [2, 1, 640]),
      };
    },
    frontend: (feeds) => {
      const count = subsample(feeds.input_features!.dims[1]!);
      return { hidden_states: tensor("float32", new Float32Array(count * DIM), [1, count, DIM]) };
    },
    splitEncoder: (feeds) => ({ encoder_out: feeds.hidden_states! }),
    decoder: (feeds) => ({
      decoder_out: tensor("float32", Float32Array.of(Number((feeds.token!.data as BigInt64Array)[0])), [1, 640]),
      h_out: tensor("float32", new Float32Array(1), [2, 1, 640]),
      c_out: tensor("float32", new Float32Array(1), [2, 1, 640]),
    }),
    joint: () => ({ logits: tensor("float32", nextStep(fake), [1, VOCAB + 5]) }),
  };
  const roles: Record<string, string> = {
    "preprocessor.onnx": "preprocessor",
    "preprocessor-model.onnx": "preprocessor",
    "encoder-model.onnx": "fusedEncoder",
    "encoder-model.int8.onnx": "fusedEncoder",
    "decoder_joint-model.onnx": "decoderJoint",
    "frontend.onnx": "frontend",
    "encoder.onnx": "splitEncoder",
    "decoder.onnx": "decoder",
    "joint.onnx": "joint",
  };
  const role = roles[file];
  if (!role) throw new Error(`no fake for ${file}`);
  fake.runs[role] = [];
  return (feeds) => {
    fake.runs[role]!.push(feeds);
    return graphs[role]!(feeds);
  };
}

function fakeBinding(fake: Fake): OrtBinding {
  return {
    version: "fake",
    createSession: async (path) => {
      fake.created.push(basename(path));
      const run = fakeGraph(basename(path), fake);
      const graph: OrtGraph = { run: async (feeds) => run(feeds), release: async () => {} };
      return graph;
    },
    tensor: (type, data, dims) => tensor(type, data, dims),
  };
}

function melFiltersNpy(bands: number, bins: number): Uint8Array {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${bands}, ${bins}), }`.padEnd(117) + "\n";
  const bytes = new Uint8Array(10 + header.length + bands * bins * 4);
  bytes.set([0x93, ...Buffer.from("NUMPY"), 1, 0]);
  new DataView(bytes.buffer).setUint16(8, header.length, true);
  bytes.set(Buffer.from(header, "latin1"), 10);
  new Float32Array(bytes.buffer, 10 + header.length).fill(0.01);
  return bytes;
}

/** A model root whose entry looks installed: marker plus the text files read at load. */
async function seededRoot(id: OrtTdtModelId): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-ort-tdt-"));
  after(() => rm(root, { recursive: true, force: true }));
  const model = ORT_TDT_MODELS[id];
  const paths = resolveOrtTdtPaths(model, root);
  await mkdir(paths.dir, { recursive: true });
  await writeFile(join(paths.dir, COMPLETE_MARKER), completeMarkerValue(paths));
  const entries = Object.entries(PIECES);
  if (model.layout === "fused") {
    const lines = entries.map(([id, piece]) => `${piece} ${id}`);
    await writeFile(join(paths.dir, model.vocab), [...lines, `<blk> ${BLANK}`].join("\n"));
  } else {
    const vocab = Object.fromEntries(entries.map(([id, piece]) => [piece, Number(id)]));
    await writeFile(join(paths.dir, model.tokenizer), JSON.stringify({ model: { vocab } }));
    await writeFile(join(paths.dir, model.melFilters), melFiltersNpy(8, 257));
  }
  return root;
}

async function openSession(id: OrtTdtModelId, steps: Float32Array[] = []) {
  const fake: Fake = { runs: {}, steps, created: [] };
  const engine = new OrtTdtEngine({ model: id, modelRoot: await seededRoot(id), binding: fakeBinding(fake) });
  const session = await engine.open({ endpoint: { mode: "manual" } });
  return { engine, fake, session };
}

const OPEN_THE_FILE = [step(10, 1), step(11, 2), step(12, 0), step(13, 1)];

for (const id of ["en-phonon-2-int8", "en-fluid-speech-mini"] as const) {
  test(`${id}: final carries text, token times and log probabilities`, async () => {
    const { engine, session } = await openSession(id, [...OPEN_THE_FILE]);
    assert.equal(engine.label, `onnxruntime-node-fake:${ORT_TDT_MODELS[id].name}`);
    session.pushAudio(new Float32Array(16_000));
    session.flush();
    const [final] = (await once(session, "final")) as [FinalEvent];
    assert.equal(final.text, "Open the file.");
    assert.equal(final.audioMs, 1_000);
    assert.ok((final.decodeMs ?? -1) >= 0);
    assert.deepEqual(final.tokens?.map((token) => [token.text, token.startMs]), [
      [" Open", 0],
      [" the", 80],
      [" file", 240],
      [".", 240],
    ]);
    // Log softmax of logit 10 against 8192 zero logits.
    const expected = -Math.log(1 + (VOCAB - 1) * Math.exp(-10));
    for (const token of final.tokens ?? []) {
      assert.ok(Math.abs((token.logProb ?? 0) - expected) < 1e-6);
    }
  });
}

test("fused layout feeds frames and adopts LSTM state only on emitted tokens", async () => {
  const { fake, session } = await openSession("en-parakeet-redux", [step(10, 1), step(BLANK, 1), step(11, 1)]);
  session.pushAudio(new Float32Array(4_000));
  session.flush();
  await once(session, "final");
  const [preprocessed] = fake.runs.preprocessor!;
  assert.deepEqual(preprocessed!.waveforms!.dims, [1, 4_000]);
  assert.deepEqual([...(preprocessed!.waveforms_lens!.data as BigInt64Array)], [4_000n]);
  const runs = fake.runs.decoderJoint!;
  assert.deepEqual(runs.map((feeds) => (feeds.targets!.data as Int32Array)[0]), [BLANK, 10, 10, 11]);
  // Run k returns state k; runs 1 and 3 emit, so only their states carry on.
  assert.deepEqual(runs.map((feeds) => (feeds.input_states_1!.data as Float32Array)[0]), [0, 1, 1, 3]);
  // Frame t is passed frame major, [1, dim, 1]: channel c holds t + c / 10.
  assert.deepEqual(runs[1]!.encoder_outputs!.dims, [1, DIM, 1]);
  const frame = [...(runs[1]!.encoder_outputs!.data as Float32Array)];
  assert.deepEqual(frame.map((value) => Math.round(value * 10) / 10), [1, 1.1, 1.2, 1.3]);
});

test("split layout chunks the front end and runs the decoder per emitted token", async () => {
  const { fake, session } = await openSession("en-fluid-speech-mini", [step(10, 0), step(11, 4)]);
  session.pushAudio(new Float32Array(16_000 * 10));
  session.flush();
  await once(session, "final");
  // 1001 mel frames -> 126 front end frames -> chunks of 50.
  const chunks = fake.runs.frontend!.map((feeds) => feeds.input_features!.dims);
  assert.deepEqual(chunks, [[1, 408, 8], [1, 416, 8], [1, 209, 8]]);
  const [encoded] = fake.runs.splitEncoder!;
  assert.deepEqual(encoded!.hidden_states!.dims, [1, 126, DIM]);
  // 1000 valid mel frames subsample to 125 valid encoder frames.
  assert.deepEqual([...(encoded!.length!.data as BigInt64Array)], [125n]);
  const tokens = fake.runs.decoder!.map((feeds) => (feeds.token!.data as BigInt64Array)[0]);
  assert.deepEqual(tokens, [BigInt(BLANK), 10n, 11n]);
});

test("a silent hold decodes to empty", async () => {
  const { session } = await openSession("en-phonon-2-int8");
  session.pushAudio(new Float32Array(3_200));
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 200);
});

test("audio too short for one STFT never reaches the graphs", async () => {
  const { fake, session } = await openSession("en-fluid-speech-mini");
  session.pushAudio(new Float32Array(319));
  session.flush();
  await once(session, "empty");
  assert.deepEqual(fake.runs.frontend, []);
});

test("audio past maxAudioMs is dropped before decode", async () => {
  const id = "en-parakeet-redux";
  const { fake, session } = await openSession(id, [step(10, 4)]);
  for (let second = 0; second < 32; second += 1) {
    session.pushAudio(new Float32Array(16_000));
  }
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  const capSamples = (ORT_TDT_MODELS[id].maxAudioMs * 16_000) / 1_000;
  assert.deepEqual(fake.runs.preprocessor!.map((feeds) => feeds.waveforms!.dims[1]), [capSamples]);
  assert.equal(final.audioMs, ORT_TDT_MODELS[id].maxAudioMs);
});

test("a decode failure surfaces as an error event", async () => {
  const { fake, session } = await openSession("en-phonon-2-int8");
  fake.failure = new Error("joint failed");
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [error] = (await once(session, "error")) as [{ err: unknown }];
  assert.equal((error.err as Error).message, "joint failed");
});

test("the engine loads each graph once across sessions and refuses other rates", async () => {
  const { engine, fake } = await openSession("en-phonon-2-int8");
  await engine.open();
  assert.deepEqual(fake.created.sort(), [
    "decoder_joint-model.onnx",
    "encoder-model.int8.onnx",
    "preprocessor-model.onnx",
  ]);
  await assert.rejects(engine.open({ sampleRate: 48_000 }), /16000 Hz/);
});
