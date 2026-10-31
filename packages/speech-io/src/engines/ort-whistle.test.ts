import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import type { EmptyEvent, FinalEvent, STTErrorEvent } from "../contract.ts";
import { COMPLETE_MARKER, completeMarkerValue } from "./assets.ts";
import {
  OrtWhistleEngine,
  whistleDetail,
  type WhistleFiles,
  type WhistleRuntime,
} from "./ort-whistle.ts";
import {
  ORT_WHISTLE_MODELS,
  resolveWhistleFiles,
  type WhistleModelId,
} from "./ort-whistle-models.ts";
import { ORT_THREADS } from "./ort-models.ts";
import { LANGS, type TranscribeResult } from "./vendor/whistle/whistle.js";

const MODEL: WhistleModelId = "whistle-onnx-fp32";

type FakeRuntime = WhistleRuntime & {
  created: WhistleFiles[];
  threads: number[];
  calls: { samples: number; language: string | undefined }[];
  maxConcurrent: number;
};

/** A runtime whose recognizer answers from `results` in order. */
function fakeRuntime(
  results: (TranscribeResult | Error)[] = [],
  failCreate = 0,
): FakeRuntime {
  let active = 0;
  let failures = failCreate;
  const runtime: FakeRuntime = {
    version: "fake",
    created: [],
    threads: [],
    calls: [],
    maxConcurrent: 0,
    createRecognizer: async (files, { threads }) => {
      if (failures-- > 0) throw new Error("load failed");
      runtime.created.push(files);
      runtime.threads.push(threads);
      return {
        transcribe: async (pcm, language) => {
          runtime.calls.push({ samples: pcm.length, language });
          active += 1;
          runtime.maxConcurrent = Math.max(runtime.maxConcurrent, active);
          await new Promise((resolve) => setTimeout(resolve, 5));
          active -= 1;
          const result = results.shift() ?? said("");
          if (result instanceof Error) throw result;
          return result;
        },
      };
    },
  };
  return runtime;
}

function said(text: string, words: TranscribeResult["words"] = []): TranscribeResult {
  return { text, language: "en", words };
}

async function seededRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "speak-easy-ort-"));
  after(() => rm(root, { recursive: true, force: true }));
  const spec = resolveWhistleFiles(ORT_WHISTLE_MODELS[MODEL], root);
  await mkdir(spec.dir, { recursive: true });
  await writeFile(join(spec.dir, COMPLETE_MARKER), completeMarkerValue(spec));
  return root;
}

async function engineWith(runtime: WhistleRuntime): Promise<OrtWhistleEngine> {
  return new OrtWhistleEngine({ model: MODEL, modelRoot: await seededRoot(), runtime });
}

test("a hold decodes once on flush to final with word tokens and timing", async () => {
  const runtime = fakeRuntime([
    said(" Turn off the lights. ", [
      { word: "Turn", probability: 0.5, start: 0, end: 0.24 },
      { word: "off", probability: 1, start: 0.24, end: 0.56 },
      { word: "the", probability: 0, start: 0.56, end: 0.72 },
      { word: "lights.", probability: 0.92, start: 0.72, end: 1.6 },
    ]),
  ]);
  const engine = await engineWith(runtime);
  const session = await engine.open();
  session.pushAudio(new Float32Array(16_000));
  session.pushAudio(new Float32Array(8_000));
  assert.equal(runtime.calls.length, 0);
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.text, "Turn off the lights.");
  assert.deepEqual(final.tokens, [
    { text: " Turn", startMs: 0, logProb: Math.log(0.5) },
    { text: " off", startMs: 240, logProb: 0 },
    { text: " the", startMs: 560, logProb: null },
    { text: " lights.", startMs: 720, logProb: Math.log(0.92) },
  ]);
  assert.equal(final.audioMs, 1_500);
  assert.ok(typeof final.decodeMs === "number" && final.decodeMs >= 0);
  assert.deepEqual(runtime.calls, [{ samples: 24_000, language: undefined }]);
  const dir = engine.paths.dir;
  assert.deepEqual(runtime.created, [
    {
      encoder: join(dir, "onnx/encoder.onnx"),
      decoder: join(dir, "onnx/decoder.onnx"),
      pack: join(dir, "whistle.pack"),
    },
  ]);
});

test("a silent hold decodes to empty", async () => {
  const session = await (await engineWith(fakeRuntime([said("")]))).open();
  session.pushAudio(new Float32Array(3_200));
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 200);
});

test("a hold with no audio is empty without touching the recognizer", async () => {
  const runtime = fakeRuntime();
  const session = await (await engineWith(runtime)).open();
  session.flush();
  const [empty] = (await once(session, "empty")) as [EmptyEvent];
  assert.equal(empty.audioMs, 0);
  assert.deepEqual(runtime.calls, []);
});

test("audio past maxAudioMs is dropped before decode", async () => {
  const runtime = fakeRuntime([said("long")]);
  const session = await (await engineWith(runtime)).open();
  for (let second = 0; second < 32; second += 1) {
    session.pushAudio(new Float32Array(16_000));
  }
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  const capSamples = (ORT_WHISTLE_MODELS[MODEL].maxAudioMs * 16_000) / 1_000;
  assert.deepEqual(
    runtime.calls.map((call) => call.samples),
    [capSamples],
  );
  assert.equal(final.audioMs, ORT_WHISTLE_MODELS[MODEL].maxAudioMs);
});

test("decodes are serialized across sessions and answer in flush order", async () => {
  const runtime = fakeRuntime([said("one"), said("two")]);
  const engine = await engineWith(runtime);
  const [first, second] = await Promise.all([engine.open(), engine.open()]);
  first.pushAudio(new Float32Array(1_600));
  second.pushAudio(new Float32Array(1_600));
  first.flush();
  second.flush();
  const [[a], [b]] = (await Promise.all([
    once(first, "final"),
    once(second, "final"),
  ])) as [[FinalEvent], [FinalEvent]];
  assert.deepEqual([a.text, b.text], ["one", "two"]);
  assert.equal(runtime.maxConcurrent, 1);
  assert.equal(runtime.created.length, 1);
});

test("a decode error is reported and the next decode still runs", async () => {
  const runtime = fakeRuntime([new Error("ort failed"), said("after")]);
  const session = await (await engineWith(runtime)).open();
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [error] = (await once(session, "error")) as [STTErrorEvent];
  assert.match(String(error.err), /ort failed/);
  session.pushAudio(new Float32Array(1_600));
  session.flush();
  const [final] = (await once(session, "final")) as [FinalEvent];
  assert.equal(final.text, "after");
});

test("end() decodes held audio without an empty event", async () => {
  const runtime = fakeRuntime([said("")]);
  const session = await (await engineWith(runtime)).open();
  let empties = 0;
  session.on("empty", () => (empties += 1));
  session.pushAudio(new Float32Array(1_600));
  await session.end();
  assert.equal(runtime.calls.length, 1);
  assert.equal(empties, 0);
  assert.throws(() => session.pushAudio(new Float32Array(1)), /after end/);
});

test("a supported language hint is passed by primary subtag; others use language ID", async () => {
  const runtime = fakeRuntime([said("a"), said("b"), said("c")]);
  const engine = await engineWith(runtime);
  for (const language of ["en-GB", "ja", undefined]) {
    const session = await engine.open({ language });
    session.pushAudio(new Float32Array(1_600));
    session.flush();
    await once(session, "final");
  }
  assert.deepEqual(
    runtime.calls.map((call) => call.language),
    ["en", undefined, undefined],
  );
});

test("a failed load is retried on the next open", async () => {
  const runtime = fakeRuntime([], 1);
  const engine = await engineWith(runtime);
  await assert.rejects(engine.open(), /load failed/);
  await engine.open();
  assert.equal(runtime.created.length, 1);
});

test("open rejects a sample rate other than 16 kHz", async () => {
  const engine = await engineWith(fakeRuntime());
  await assert.rejects(engine.open({ sampleRate: 48_000 }), /expects 16000 Hz/);
});

test("the label names the runtime version and model", async () => {
  const engine = await engineWith(fakeRuntime());
  assert.equal(engine.label, "onnxruntime-node-fake:whistle-onnx-fp32-14fc92c");
});

test("the recognizer loads on the engine's intra op threads", async () => {
  const runtime = fakeRuntime();
  const engine = await engineWith(runtime);
  await engine.open();
  assert.equal(engine.threads, ORT_THREADS);
  assert.deepEqual(runtime.threads, [ORT_THREADS]);
});

test("words without timestamps give text and no tokens", () => {
  assert.deepEqual(
    whistleDetail(said("hi there", [{ word: "hi", probability: 0.9 }])),
    { text: "hi there", tokens: [] },
  );
});

test("descriptors pin every file, name files in the list, and match the decoder's languages", () => {
  for (const model of Object.values(ORT_WHISTLE_MODELS)) {
    assert.equal(model.kind, "offline");
    const paths = model.files.map((file) => file.name);
    for (const name of [model.encoder, model.decoder, model.pack]) {
      assert.ok(paths.includes(name), name);
    }
    for (const graph of [model.encoder, model.decoder]) {
      assert.ok(paths.includes(`${graph}.data`), `${graph}.data`);
    }
    for (const file of model.files) {
      assert.match(file.sha256, /^[0-9a-f]{64}$/);
      assert.ok(file.bytes > 0);
      assert.ok(file.url.endsWith(`/${file.name}`));
    }
    assert.deepEqual(model.languages, LANGS);
    assert.equal(model.licence.id, "Apache-2.0");
  }
});
