import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import type { EmptyEvent, FinalEvent } from "../contract.ts";
import { COMPLETE_MARKER } from "./assets.ts";
import { logitsFor, TINY_TOKENIZER } from "./ctc-test-support.ts";
import { fileAssetMarker } from "./file-assets.ts";
import type { CtcRuntime } from "./ort-ctc-runtime.ts";
import { OrtGraniteEngine } from "./ort-granite.ts";
import {
  GRANITE_MODELS,
  resolveGraniteFiles,
  type GraniteModelId,
} from "./ort-granite-models.ts";

const MODEL: GraniteModelId = "en-granite-speech-5.0-470m-turboctc-int8";
const VOCAB_SIZE = 7;

type FakeRuntime = CtcRuntime & {
  loads: string[];
  runs: { frames: number; featureDim: number }[];
  /** Argmax id per output frame for the next run; missing frames are blank. */
  script: (number[] | Error)[];
  failLoads: number;
};

function fakeRuntime(script: (number[] | Error)[] = []): FakeRuntime {
  const fake: FakeRuntime = {
    version: "fake",
    loads: [],
    runs: [],
    script,
    failLoads: 0,
    async loadGraph(path) {
      fake.loads.push(path);
      if (fake.failLoads > 0) {
        fake.failLoads -= 1;
        throw new Error("graph failed to load");
      }
      return {
        async run(_features, frames, featureDim) {
          fake.runs.push({ frames, featureDim });
          const next = fake.script.shift() ?? [];
          if (next instanceof Error) throw next;
          const outFrames = Math.floor(frames / 4);
          const ids = Array.from({ length: outFrames }, (_, frame) => next[frame] ?? 0);
          return logitsFor(ids, VOCAB_SIZE);
        },
      };
    },
  };
  return fake;
}

/** A model root whose file set already looks installed. */
async function seededRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-granite-"));
  after(() => rm(root, { recursive: true, force: true }));
  const spec = resolveGraniteFiles(GRANITE_MODELS[MODEL], root);
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, GRANITE_MODELS[MODEL].tokenizer.path), TINY_TOKENIZER);
  await writeFile(join(spec.dir, COMPLETE_MARKER), fileAssetMarker(spec));
  return root;
}

async function openSession(runtime = fakeRuntime()) {
  const engine = new OrtGraniteEngine({ model: MODEL, modelRoot: await seededRoot(), runtime });
  return { engine, runtime, session: await engine.open({ endpoint: { mode: "manual" } }) };
}

test("final carries text and CTC tokens with frame times and log probabilities", async () => {
  const { runtime, session } = await openSession(fakeRuntime([[0, 1, 1, 0, 2, 3]]));
  session.pushAudio(new Float32Array(16_000));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.deepEqual(runtime.runs, [{ frames: 50, featureDim: 320 }]);
  assert.equal(final.text, "open the file");
  assert.equal(final.audioMs, 1_000);
  assert.ok((final.decodeMs ?? -1) >= 0);
  assert.deepEqual(final.tokens?.map((token) => [token.text, token.startMs]), [
    ["open", 80],
    [" the", 320],
    [" file", 400],
  ]);
  for (const token of final.tokens ?? []) {
    assert.ok(token.logProb !== null && token.logProb < 0 && token.logProb > -0.1);
  }
});

test("an all blank decode is empty", async () => {
  const { session } = await openSession();
  session.pushAudio(new Float32Array(8_000));
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 500);
});

test("a hold below the graph minimum is empty without running the graph", async () => {
  const { runtime, session } = await openSession();
  session.pushAudio(new Float32Array(1_000));
  session.flush();
  await once(session, "empty");
  assert.deepEqual(runtime.runs, []);
});

test("audio past maxAudioMs is dropped before decode", async () => {
  const { runtime, session } = await openSession(fakeRuntime([[1]]));
  const second = new Float32Array(16_000);
  for (let index = 0; index < 31; index += 1) {
    session.pushAudio(second);
  }
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.audioMs, GRANITE_MODELS[MODEL].maxAudioMs);
  assert.deepEqual(runtime.runs, [{ frames: 1_500, featureDim: 320 }]);
});

test("a graph failure surfaces as an error event", async () => {
  const { session } = await openSession(fakeRuntime([new Error("graph failed")]));
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [error] = (await once(session, "error")) as [{ err: unknown }];
  assert.equal((error.err as Error).message, "graph failed");
});

test("one graph serves every session; a failed load is retried", async () => {
  const runtime = fakeRuntime();
  runtime.failLoads = 1;
  const engine = new OrtGraniteEngine({ model: MODEL, modelRoot: await seededRoot(), runtime });
  await assert.rejects(() => engine.open(), /graph failed to load/);
  await engine.open();
  await engine.open();
  assert.equal(runtime.loads.length, 2);
  assert.match(runtime.loads[1]!, /granite-speech-5\.0-470m-turboctc-onnx-int8\/model_int8\.onnx$/);
  assert.equal(engine.label, "onnxruntime-node-fake:granite-speech-5.0-470m-turboctc-onnx-int8");
});

test("the engine refuses other sample rates and unknown models", async () => {
  const { engine } = await openSession();
  await assert.rejects(() => engine.open({ sampleRate: 48_000 }), /16000 Hz/);
  assert.throws(
    () => new OrtGraniteEngine({ model: "nope" as GraniteModelId, modelRoot: "/unused", runtime: fakeRuntime() }),
    /Unknown Granite model/,
  );
});
