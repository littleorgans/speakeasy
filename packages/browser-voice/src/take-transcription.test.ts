import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { loadHotwords } from "@speakeasy/speech-io";
import { createSherpaEngine, resolveModelPaths, SHERPA_MODELS } from "@speakeasy/speech-io/stt";
import type { OfflineRecognizerConfig } from "../../speech-io/src/engines/sherpa-api.ts";
import { bindingWith, seededModelRoot } from "../../speech-io/src/engines/test-support.ts";
import { fixedCpuSnapshot, instrumentCpuBinding, preflightCpuModel, TakeTranscription, validateCpuRuntime } from "./take-transcription.ts";
import { readBrowserVoiceOptions } from "./server-options.ts";
import { ChildTranscriber, CPU_PRESET, TRANSCRIBE_MODEL_ID, TranscriptionError, type TranscriptionMode } from "./transcriber.ts";

async function setup(decode?: () => Promise<void>, hotwordsPath?: string, mode?: TranscriptionMode) {
  const modelRoot = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  const source = join(modelRoot, "hotwords.txt");
  const pieces = ["<unk>", "<s>", "</s>", ...Array.from({ length: 271 }, (_, i) => `<control${i}>`),
    "en", "▁", "a", "b", ..."lunchdegrok", "<blk>"];
  await writeFile(join(resolveModelPaths(SHERPA_MODELS[TRANSCRIBE_MODEL_ID], modelRoot).dir, "tokens.txt"),
    pieces.map((piece, id) => `${piece} ${id}\n`).join(""));
  const configs: OfflineRecognizerConfig[] = [];
  const constructedWords: string[] = [];
  const sampleCounts: number[] = [];
  let reads = 0;
  const snapshots = new EventEmitter();
  let created = 0;
  let active = 0;
  let maxActive = 0;
  const binding = bindingWith({ OfflineRecognizer: class {
    readonly words: string;
    constructor(config: OfflineRecognizerConfig) {
      configs.push(config);
      this.words = config.hotwordsFile ? readFileSync(config.hotwordsFile, "utf8") : "greedy";
      constructedWords.push(this.words);
    }
    createStream() { return { acceptWaveform({ samples }: { samples: Float32Array }) { sampleCounts.push(samples.length); } }; }
    async decodeAsync() {
      maxActive = Math.max(maxActive, ++active);
      await decode?.();
      active -= 1;
      return { text: this.words };
    }
  } });
  const service = new TakeTranscription({
    mode,
    ...(hotwordsPath === undefined ? { async readSnapshot() {
      const snapshot = await loadHotwords({ snapshot: true, source });
      reads += 1;
      snapshots.emit("read");
      return snapshot;
    } } : { hotwordsPath }),
    async createEngine(hotwords) {
      created += 1;
      return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot, hotwords, binding });
    },
  });
  return { service, modelRoot, source, configs, constructedWords, sampleCounts, snapshots, reads: () => reads, created: () => created,
    maxActive: () => maxActive, take: () => service.transcribe(new Float32Array(320).fill(0.25)) };
}

test("every take reloads the file; unchanged content reuses and changed content rebuilds on CPU", async () => {
  const h = await setup();
  await writeFile(h.source, "a\n");
  const first = await h.take();
  await writeFile(h.source, "a\n");
  const unchanged = await h.take();
  assert.equal(h.reads(), 2);
  assert.equal(h.configs.length, 1);
  assert.equal(first.hotwords?.count, 1);
  assert.equal(first.hotwords?.revision, unchanged.hotwords?.revision);
  assert.equal(first.model, "en-parakeet-ultra-int8");
  await writeFile(h.source, "b\n");
  const changed = await h.take();
  assert.equal(changed.text, "b");
  assert.notEqual(changed.hotwords?.revision, first.hotwords?.revision);
  assert.equal(h.configs.length, 2);
  await writeFile(h.source, "b\n# changed comment\n");
  await h.take();
  assert.equal(h.configs.length, 3, "compare content, including comments, rather than just parsed phrases");
  assert.deepEqual(h.constructedWords, ["a\n", "b\n", "b\n"]);
  assert.ok(h.configs.every((config) => config.modelConfig.provider === "cpu" && config.decodingMethod === "modified_beam_search"));
  assert.ok(h.configs.every((config) => config.modelConfig.numThreads === 2 && config.featConfig.sampleRate === 16_000 &&
    config.featConfig.featureDim === 80 && config.maxActivePaths === 4 && config.hotwordsScore === 2));
  assert.equal(first.text, "a", "old transcript remains associated with its snapshot");
});

test("fixed CPU snapshots never read the editable list and construct before first processing", async () => {
  const h = await setup(undefined, undefined, "sweep/score-2");
  await writeFile(h.source, "a\n");
  const loaded = await h.service.prepare();
  assert.equal(h.configs.length, 1, "ready includes actual recognizer construction");
  assert.deepEqual(h.sampleCounts, [], "prepare performs no warmup decode");
  assert.ok(loaded.loadMs >= 0);
  const first = await h.take();
  await writeFile(h.source, "b\n");
  const second = await h.take();
  assert.equal(h.reads(), 0);
  assert.equal(h.configs.length, 1);
  assert.deepEqual(h.constructedWords, ["launch\nclaude\ngrok\n"]);
  assert.equal(first.hotwords?.revision, "0f5d821e73171e540a688d50c481007f58662609e1e7892b20829ba7092ea79b");
  assert.equal(first.comparison?.firstUse, true);
  assert.equal(second.comparison?.firstUse, false);
  assert.equal(first.comparison?.processingMs, null, "no invented native processing measurement");
});

test("fixed CPU rejects over 60 seconds and invalid PCM before snapshot or model load", async () => {
  const h = await setup(undefined, undefined, "sweep/score-2");
  await assert.rejects(h.service.transcribe(new Float32Array(960001)), /60 seconds/);
  await assert.rejects(h.service.transcribe(new Float32Array([NaN])), /finite/);
  await assert.rejects(h.service.transcribe(new Float32Array()), /sample/);
  assert.equal(h.created(), 0);
  assert.equal(h.reads(), 0);
  await h.service.transcribe(new Float32Array(960000));
  assert.deepEqual(h.sampleCounts, [960000], "the full boundary take is decoded without truncation");
});

test("configured external list reaches every snapshot, native read and transcript metadata", async (t) => {
  const root = await mkdtemp("/tmp/speak-easy-external-hotwords-");
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "owner-words.txt");
  const { hotwordsPath } = readBrowserVoiceOptions(["--hotwords", source], {});
  const h = await setup(undefined, hotwordsPath);
  const missing = await h.take();
  assert.equal(missing.text, "greedy");
  assert.equal(missing.hotwords?.path, source);
  assert.equal(missing.hotwords?.count, 0);
  await writeFile(source, "a\n");
  const first = await h.take();
  const unchanged = await h.take();
  assert.equal(first.text, "a");
  assert.equal(first.hotwords?.path, source);
  assert.equal(first.hotwords?.count, 1);
  assert.equal(unchanged.hotwords?.revision, first.hotwords?.revision);
  assert.deepEqual(h.constructedWords, ["greedy", "a\n"]);
  await writeFile(source, "a☃\n");
  await assert.rejects(h.take(), (error: unknown) => {
    assert.ok(error instanceof TranscriptionError);
    assert.equal(error.hotwords?.path, source);
    assert.equal(error.hotwords?.count, 1);
    assert.match(error.message, /unencodable UTF8 coverage/);
    return true;
  });
  assert.equal(h.configs.length, 2, "invalid content never reaches native decode");
  await writeFile(source, "b\n");
  const changed = await h.take();
  assert.equal(changed.text, "b");
  assert.notEqual(changed.hotwords?.revision, first.hotwords?.revision);
  await writeFile(source, "");
  const empty = await h.take();
  assert.equal(empty.text, "greedy");
  assert.equal(empty.hotwords?.path, source);
  assert.equal(empty.hotwords?.count, 0);
  assert.deepEqual(h.constructedWords, ["greedy", "a\n", "b\n", "greedy"]);
});

test("the reader sends a variable final audio length without zero padding", async () => {
  const h = await setup();
  await h.service.transcribe(new Float32Array(333).fill(0.25));
  assert.deepEqual(h.sampleCounts, [333]);
});

test("missing and empty files decode greedily; list removal replaces the biased recognizer", async () => {
  const h = await setup();
  const missing = await h.take();
  assert.equal(missing.text, "greedy");
  assert.equal(missing.hotwords?.count, 0);
  await writeFile(h.source, "");
  await h.take();
  assert.equal(h.configs.length, 1, "missing and empty both have empty content");
  await writeFile(h.source, "a\n");
  await h.take();
  await writeFile(h.source, "");
  const empty = await h.take();
  assert.equal(empty.text, "greedy");
  assert.equal(empty.hotwords?.count, 0);
  assert.deepEqual(h.configs.map((config) => config.decodingMethod), ["greedy_search", "modified_beam_search", "greedy_search"]);
});

test("invalid snapshots reject atomically with every reason, then correction recovers", async () => {
  const h = await setup();
  await writeFile(h.source, "a\n");
  await h.take();
  await writeFile(h.source, "a\na☃\na :2\n");
  await assert.rejects(h.take(), (error: unknown) => {
    assert.ok(error instanceof TranscriptionError);
    assert.equal(error.model, "en-parakeet-ultra-int8");
    assert.equal(error.hotwords?.count, 3);
    assert.equal(error.hotwords?.path, h.source);
    assert.match(error.message, /a☃.*unencodable UTF8 coverage/);
    assert.match(error.message, /a :2.*reserved native score syntax/);
    return true;
  });
  assert.equal(h.configs.length, 1, "neither old nor greedy recognizer decoded the invalid list");
  await writeFile(h.source, "b\n");
  assert.equal((await h.take()).text, "b");
  assert.equal(h.configs.length, 2);
});

test("queued takes snapshot before waiting and never overlap native decode", async () => {
  let release!: () => void;
  let started!: () => void;
  const decoding = new Promise<void>((resolve) => { started = resolve; });
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  const h = await setup(async () => { started(); await blocked; });
  await writeFile(h.source, "a\n");
  const first = h.take();
  await decoding;
  await writeFile(h.source, "b\n");
  const reread = once(h.snapshots, "read");
  const second = h.take();
  await reread;
  assert.equal(h.reads(), 2, "the queued request has already read its own file");
  await writeFile(h.source, "a\na\n");
  release();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a.text, "a");
  assert.equal(b.text, "b");
  assert.equal(b.hotwords?.count, 1);
  assert.equal(h.maxActive(), 1);
  assert.equal(h.created(), 2);
});

test("native construction and CPU processing have separate clocks and preserve raw empty or formatted text", async (t) => {
  const h = await setup();
  let clock = 0;
  let rawText = " \u0000raw text\n";
  t.mock.method(performance, "now", () => clock);
  const timed = instrumentCpuBinding(bindingWith({ OfflineRecognizer: class {
    constructor(config: OfflineRecognizerConfig) {
      assert.equal(config.modelConfig.provider, "cpu");
      assert.equal(config.modelConfig.numThreads, 2);
      assert.equal(config.featConfig.featureDim, 80);
      assert.equal(config.modelConfig.modelType, "nemo_transducer");
      assert.equal(config.modelConfig.modelingUnit, "bpe");
      assert.equal(config.decodingMethod, "modified_beam_search");
      assert.equal(config.maxActivePaths, 4);
      assert.equal(config.hotwordsScore, 2);
      assert.equal(Object.hasOwn(config, "lengthReward"), false, "retain the pinned parent's default zero reward");
      clock += 37;
    }
    createStream() {
      clock += 2;
      return { acceptWaveform({ samples }: { samples: Float32Array }) {
        assert.equal(samples.length, 333);
        assert.equal(Object.is(samples[0], -0), true);
        assert.equal(samples[1], 1.5);
        clock += 3;
      } };
    }
    async decodeAsync() { clock += 5; return { text: rawText }; }
  } }));
  const service = new TakeTranscription({ mode: CPU_PRESET.mode, measurements: () => timed.measurements,
    async createEngine(hotwords) {
      return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot: h.modelRoot, hotwords, binding: timed.binding });
    } });
  const loaded = await service.prepare();
  assert.equal(loaded.modelConstructionMs, 37);
  assert.equal(timed.measurements.processingMs, null, "ready performs no native stream or decode");
  const backing = new Float32Array(337);
  const samples = backing.subarray(2, 335);
  samples[0] = -0;
  samples[1] = 1.5;
  const first = await service.transcribe(samples);
  assert.equal(first.text, rawText, "comparison bypasses legacy result normalization");
  assert.equal(first.comparison?.processingMs, 10);
  assert.equal(first.comparison?.modelConstructionMs, 37);
  assert.equal(first.comparison?.firstUse, true);
  rawText = "";
  const empty = await service.transcribe(samples);
  assert.equal(empty.text, "");
  assert.equal(empty.comparison?.firstUse, false);
  assert.equal(empty.comparison?.processingMs, 10);
});

test("fixed snapshot is immutable, and the fixed runtime rejects the experimental decoder", () => {
  const snapshot = fixedCpuSnapshot();
  assert.throws(() => snapshot.hotwords!.phrases.push("extra"), TypeError);
  assert.throws(() => { snapshot.hotwords!.score = 9; }, TypeError);
  const parent = { ...bindingWith({}), version: "2.0.1", gitSha1: "af2d8d81", onnxruntimeVersion: "1.28.2" };
  validateCpuRuntime(parent);
  assert.throws(() => validateCpuRuntime({ ...parent, version: "2.0.2" }), /pinned sherpa/);
  assert.throws(() => validateCpuRuntime({ ...parent, gitSha1: "3898de13" }), /pinned sherpa/);
  assert.throws(() => validateCpuRuntime({ ...parent, onnxruntimeVersion: "1.27.0" }), /pinned sherpa/);
});

test("CPU asset preflight rejects an absent marker and a corrupt completed model without installation", async () => {
  const root = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  await assert.rejects(preflightCpuModel(join(root, "absent")), /not installed/);
  const paths = resolveModelPaths(SHERPA_MODELS[TRANSCRIBE_MODEL_ID], root);
  const tokens = SHERPA_MODELS[TRANSCRIBE_MODEL_ID].files.find((file) => file.name === "tokens.txt")!;
  await writeFile(join(paths.dir, tokens.name), "corrupt");
  await assert.rejects(preflightCpuModel(root), /asset mismatch: tokens/);
  await writeFile(join(paths.dir, tokens.name), Buffer.alloc(tokens.bytes));
  await assert.rejects(preflightCpuModel(root), /asset mismatch: tokens/, "a correct size cannot bypass SHA256 validation");
});

const ready = {
  type: "ready", version: 1, preset: CPU_PRESET, load: { loadMs: 7, modelConstructionMs: 3 },
  assets: Object.fromEntries(SHERPA_MODELS[TRANSCRIBE_MODEL_ID].files.map((file) => [file.name, file.sha256])),
};
const healthyWorker = `
  process.send(ready);
  let firstUse = true;
  process.on("message", ({ id, samples }) => {
    process.send({ id, text: String(process.pid) + ":" + samples.length + ":" + samples[1], decodeMs: 2,
      hotwords: { path: "sweep/score-2", count: 3, revision: ready.preset.hotwordsSha256 },
      comparison: { ...ready.load, processingMs: 1, workerWallMs: 4, firstUse } });
    firstUse = false;
  });
`;
async function fakeChild(t: TestContext, body = healthyWorker, options: {
  mode?: TranscriptionMode; timeoutMs?: number; startupTimeoutMs?: number;
} = {}) {
  const root = await mkdtemp("/tmp/speak-easy-cpu-ipc-");
  const workerPath = join(root, "worker.mjs");
  const write = (source: string) => writeFile(workerPath,
    `const ready = ${JSON.stringify(ready)};\n${source}\nprocess.on("disconnect", () => process.exit(0));\n`);
  await write(body);
  const child = new ChildTranscriber({ mode: CPU_PRESET.mode, workerPath, startupTimeoutMs: 2_000, ...options });
  t.after(async () => { await child.closeAndWait(); await rm(root, { recursive: true, force: true }); });
  return { child, write, workerPath };
}

test("advanced IPC preserves offset PCM, separates fixed instances, and keeps warm first use metadata", async (t) => {
  const a = await fakeChild(t);
  const b = await fakeChild(t);
  assert.deepEqual(await a.child.prepare(), ready);
  const backing = new Float32Array([99, 0.25, 1.5, -0.5, 88]);
  const first = await a.child.transcribe(backing.subarray(1, 4));
  const second = await a.child.transcribe(backing.subarray(1, 4));
  const other = await b.child.transcribe(backing.subarray(1, 4));
  assert.match(first.text, /:3:1.5$/);
  assert.equal(second.text, first.text);
  assert.notEqual(first.text.split(":")[0], other.text.split(":")[0]);
  assert.equal(first.comparison?.firstUse, true);
  assert.equal(second.comparison?.firstUse, false);
  assert.ok(first.comparison!.processStartupMs! >= 0);
  await a.child.closeAndWait();
  const recovered = await a.child.transcribe(new Float32Array([0.25]));
  assert.notEqual(recovered.text.split(":")[0], first.text.split(":")[0]);
});

test("legacy custom worker replies and the configured argv hotword path retain their interface", async (t) => {
  const { workerPath } = await fakeChild(t, `
    process.on("message", ({ id }) => process.send({ id, text: process.argv[2], decodeMs: 0,
      hotwords: { path: "list", count: 1, revision: "abc123" } }));
  `, { mode: "legacy" });
  const legacy = new ChildTranscriber({ workerPath, hotwordsPath: "relative-owner-list.txt" });
  t.after(() => legacy.closeAndWait());
  const result = await legacy.transcribe(new Float32Array(1));
  assert.match(result.text, /relative-owner-list.txt$/);
  assert.equal(result.comparison, undefined);
  assert.equal(result.hotwords?.revision, "abc123");
});

test("bad ready, missing ready, and malformed replies retire the child before explicit recovery", async (t) => {
  const h = await fakeChild(t, `ready.preset.hotwordsScore = 9; process.send(ready);`);
  await assert.rejects(h.child.prepare(), /preset mismatch/);
  await h.write(healthyWorker);
  assert.match((await h.child.transcribe(new Float32Array([0.25]))).text, /:1:/);
  await h.child.closeAndWait();
  await h.write(`process.send(ready); process.on("message", ({ id }) => process.send({ id, text: "oops", decodeMs: -1 }));`);
  await assert.rejects(h.child.transcribe(new Float32Array(1)), /Invalid CPU transcript/);
  await h.write(healthyWorker);
  await h.child.transcribe(new Float32Array(1));
  const missing = await fakeChild(t, "", { startupTimeoutMs: 75 });
  await assert.rejects(missing.child.prepare(), /ready timed out/);
});

test("processing timeout settles concurrent requests once and a later explicit request recovers", async (t) => {
  const h = await fakeChild(t, `process.send(ready); process.on("message", () => {});`, { timeoutMs: 75 });
  const first = h.child.transcribe(new Float32Array(1));
  const second = h.child.transcribe(new Float32Array(1));
  const outcomes = await Promise.allSettled([first, second]);
  assert.ok(outcomes.every((result) => result.status === "rejected" && /timed out/.test(String(result.reason))));
  await h.write(healthyWorker);
  const next = await h.child.transcribe(new Float32Array(1));
  assert.equal(next.comparison?.firstUse, true);
});

test("close during startup and before spawn cannot start an obsolete decode", async (t) => {
  const h = await fakeChild(t, "");
  const preSpawn = h.child.transcribe(new Float32Array(1));
  h.child.close();
  await assert.rejects(preSpawn, /closed/);
  const startup = h.child.prepare();
  await new Promise<void>((resolve) => setImmediate(resolve));
  const rejected = assert.rejects(startup, /closed/);
  await h.child.closeAndWait();
  await rejected;
  await h.write(healthyWorker);
  await h.child.transcribe(new Float32Array(1));
});

test("unknown request ids and oversized replies fail the owning CPU generation", async (t) => {
  const wrongId = await fakeChild(t, `process.send(ready);
    process.on("message", ({ id }) => process.send({ id: id + 1, text: "wrong", decodeMs: 0 }));`);
  await assert.rejects(wrongId.child.transcribe(new Float32Array(1)), /unknown request id/);
  const oversized = await fakeChild(t, `process.send(ready);
    process.on("message", ({ id }) => process.send({ id, text: "x".repeat(65537), decodeMs: 0 }));`);
  await assert.rejects(oversized.child.transcribe(new Float32Array(1)), /Invalid CPU worker reply/);
});

test("fixed ingress refuses oversized audio before spawning, and the real child refuses missing assets before loading native code", async () => {
  const root = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  const child = new ChildTranscriber({ mode: CPU_PRESET.mode, modelRoot: join(root, "absent") });
  try {
    await assert.rejects(child.transcribe(new Float32Array(960001)), /60 seconds/);
    await assert.rejects(child.prepare(), /not installed/);
  } finally { await child.closeAndWait(); }
});
