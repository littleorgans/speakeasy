import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { createSherpaEngine, resolveModelPaths, SHERPA_MODELS } from "@speakeasy/speech-io/stt";
import type { OfflineRecognizerConfig } from "../../speech-io/src/engines/sherpa-api.ts";
import { bindingWith, seededModelRoot } from "../../speech-io/src/engines/test-support.ts";
import { instrumentCpuBinding, namesSnapshot, preflightCpuModel, TakeTranscription, validateCpuRuntime } from "./take-transcription.ts";
import {
  ChildTranscriber, CPU_DECODING, CPU_PRESETS, TRANSCRIBE_MODEL_ID, TranscriptionError, type TranscriptionMode,
} from "./transcriber.ts";
import { SetupComparison } from "./setup-comparison.ts";
import { SetupWorkers } from "./setup-workers.ts";

const EIGHT = ["Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen"];
const lines = (names: string[]) => createHash("sha256").update(names.map((name) => `${name}\n`).join("")).digest("hex");

async function setup(decode?: () => Promise<void>, mode?: TranscriptionMode) {
  const modelRoot = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  const pieces = ["<unk>", "<s>", "</s>", ...Array.from({ length: 271 }, (_, i) => `<control${i}>`),
    "en", "▁", "a", "b", ..."lunchdegrok", "<blk>"];
  await writeFile(join(resolveModelPaths(SHERPA_MODELS[TRANSCRIBE_MODEL_ID], modelRoot).dir, "tokens.txt"),
    pieces.map((piece, id) => `${piece} ${id}\n`).join(""));
  const configs: OfflineRecognizerConfig[] = [];
  const constructedWords: string[] = [];
  const sampleCounts: number[] = [];
  let created = 0;
  let active = 0;
  let maxActive = 0;
  const binding = bindingWith({ OfflineRecognizer: class {
    readonly words: string;
    constructor(config: OfflineRecognizerConfig) {
      configs.push(config);
      this.words = config.hotwordsFile ? readFileSync(config.hotwordsFile, "utf8") : "no list";
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
    async createEngine(hotwords) {
      created += 1;
      return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot, hotwords, binding });
    },
  });
  return { service, modelRoot, configs, constructedWords, sampleCounts, created: () => created,
    maxActive: () => maxActive, take: (names: string[]) => service.transcribe(new Float32Array(320).fill(0.25), names) };
}

test("each take brings its names; the same names reuse the recognizer and other names rebuild it", async () => {
  const h = await setup();
  const first = await h.take(["a"]);
  const unchanged = await h.take(["a"]);
  assert.equal(h.configs.length, 1);
  assert.equal(first.hotwords?.count, 1);
  assert.equal(first.hotwords?.path, "names");
  assert.equal(first.hotwords?.revision, lines(["a"]));
  assert.equal(first.hotwords?.revision, unchanged.hotwords?.revision);
  assert.equal(first.model, "en-parakeet-ultra-int8");
  const changed = await h.take(["b"]);
  assert.equal(changed.text, "b");
  assert.notEqual(changed.hotwords?.revision, first.hotwords?.revision);
  await h.take(["b", "a"]);
  assert.equal(h.configs.length, 3, "order is part of the list");
  assert.deepEqual(h.constructedWords, ["a\n", "b\n", "b\na\n"]);
  assert.ok(h.configs.every((config) => config.modelConfig.provider === "cpu" && config.decodingMethod === "modified_beam_search"));
  assert.ok(h.configs.every((config) => config.modelConfig.numThreads === 2 && config.featConfig.sampleRate === 16_000 &&
    config.featConfig.featureDim === 80 && config.maxActivePaths === 4 && config.hotwordsScore === 2));
  assert.equal(first.text, "a", "old transcript remains associated with its snapshot");
});

test("fixed CPU prepare constructs for its names before first processing, and other names rebuild in prepare", async () => {
  const h = await setup(undefined, "sweep/score-2");
  const loaded = await h.service.prepare(["a"]);
  assert.equal(h.configs.length, 1, "ready includes actual recognizer construction");
  assert.deepEqual(h.sampleCounts, [], "prepare performs no warmup decode");
  assert.ok(loaded.loadMs >= 0);
  const first = await h.take(["a"]);
  const second = await h.take(["a"]);
  assert.equal(h.configs.length, 1);
  assert.equal(first.hotwords?.revision, lines(["a"]));
  assert.equal(first.comparison?.firstUse, true);
  assert.equal(second.comparison?.firstUse, false);
  assert.equal(first.comparison?.processingMs, null, "no invented native processing measurement");
  await h.service.prepare(["b"]);
  assert.equal(h.configs.length, 2, "a names change rebuilds during prepare, so it counts as load");
  const rebuilt = await h.take(["b"]);
  assert.equal(h.configs.length, 2);
  assert.equal(rebuilt.comparison?.firstUse, true);
  assert.deepEqual(h.constructedWords, ["a\n", "b\n"]);
  assert.deepEqual(h.sampleCounts, [320, 320, 320]);
});

test("fixed CPU rejects over 60 seconds, invalid PCM and malformed names before any model load", async () => {
  const h = await setup(undefined, "sweep/score-2");
  await assert.rejects(h.service.transcribe(new Float32Array(960001), ["a"]), /60 seconds/);
  await assert.rejects(h.service.transcribe(new Float32Array([NaN]), ["a"]), /finite/);
  await assert.rejects(h.service.transcribe(new Float32Array(), ["a"]), /sample/);
  for (const names of [["a\nb"], [" a"], [""], "a", [1]]) {
    await assert.rejects(h.service.transcribe(new Float32Array(1), names as string[]), /CPU names/);
    await assert.rejects(h.service.prepare(names as string[]), /CPU names/);
  }
  assert.equal(h.created(), 0);
  await h.service.transcribe(new Float32Array(960000), ["a"]);
  assert.deepEqual(h.sampleCounts, [960000], "the full boundary take is decoded without truncation");
});

test("the reader sends a variable final audio length without zero padding", async () => {
  const h = await setup();
  await h.service.transcribe(new Float32Array(333).fill(0.25), ["a"]);
  assert.deepEqual(h.sampleCounts, [333]);
});

test("the legacy reader decodes an empty list greedily; list removal replaces the biased recognizer", async () => {
  const h = await setup();
  const empty = await h.take([]);
  assert.equal(empty.text, "no list");
  assert.equal(empty.hotwords?.count, 0);
  assert.equal(empty.hotwords?.revision, lines([]));
  await h.take([]);
  assert.equal(h.configs.length, 1);
  await h.take(["a"]);
  const removed = await h.take([]);
  assert.equal(removed.text, "no list");
  assert.deepEqual(h.configs.map((config) => config.decodingMethod), ["greedy_search", "modified_beam_search", "greedy_search"]);
});

test("names the tokenizer refuses reject atomically, naming every name and reason, then correction recovers", async () => {
  const h = await setup();
  await h.take(["a"]);
  await assert.rejects(h.take(["a", "a☃", "a :2"]), (error: unknown) => {
    assert.ok(error instanceof TranscriptionError);
    assert.equal(error.model, "en-parakeet-ultra-int8");
    assert.equal(error.hotwords?.count, 3);
    assert.equal(error.hotwords?.path, "names");
    assert.match(error.message, /a☃.*unencodable UTF8 coverage/);
    assert.match(error.message, /a :2.*reserved native score syntax/);
    return true;
  });
  assert.equal(h.configs.length, 1, "neither the old nor an unbiased recognizer decoded the invalid list");
  assert.equal((await h.take(["b"])).text, "b");
  assert.equal(h.configs.length, 2);
});

test("queued takes keep the names they were called with and never overlap native decode", async () => {
  let release!: () => void;
  let started!: () => void;
  const decoding = new Promise<void>((resolve) => { started = resolve; });
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  const h = await setup(async () => { started(); await blocked; });
  const first = h.take(["a"]);
  await decoding;
  const names = ["b"];
  const second = h.take(names);
  names.push("a");
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
  const service = new TakeTranscription({ mode: "sweep/score-2", measurements: () => timed.measurements,
    async createEngine(hotwords) {
      return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot: h.modelRoot, hotwords, binding: timed.binding });
    } });
  const loaded = await service.prepare(["a"]);
  assert.equal(loaded.modelConstructionMs, 37);
  assert.equal(timed.measurements.processingMs, null, "ready performs no native stream or decode");
  const backing = new Float32Array(337);
  const samples = backing.subarray(2, 335);
  samples[0] = -0;
  samples[1] = 1.5;
  const first = await service.transcribe(samples, ["a"]);
  assert.equal(first.text, rawText, "comparison bypasses legacy result normalization");
  assert.equal(first.comparison?.decodingMethod, "modified_beam_search");
  assert.equal(first.comparison?.processingMs, 10);
  assert.equal(first.comparison?.modelConstructionMs, 37);
  assert.equal(first.comparison?.firstUse, true);
  rawText = "";
  const empty = await service.transcribe(samples, ["a"]);
  assert.equal(empty.text, "");
  assert.equal(empty.comparison?.firstUse, false);
  assert.equal(empty.comparison?.processingMs, 10);
});

test("an empty list on a CPU preset keeps the pinned beam search and reports what was constructed", async () => {
  const h = await setup();
  const configs: OfflineRecognizerConfig[] = [];
  const timed = instrumentCpuBinding(bindingWith({ OfflineRecognizer: class {
    constructor(config: OfflineRecognizerConfig) { configs.push(config); }
    createStream() { return { acceptWaveform() {} }; }
    async decodeAsync() { return { text: "plain" }; }
  } }));
  const service = new TakeTranscription({ mode: "tm/today", measurements: () => timed.measurements,
    async createEngine(hotwords) {
      return createSherpaEngine({ model: TRANSCRIBE_MODEL_ID, modelRoot: h.modelRoot, hotwords, binding: timed.binding });
    } });
  await service.prepare([]);
  const empty = await service.transcribe(new Float32Array(16), []);
  await service.prepare(["a"]);
  const listed = await service.transcribe(new Float32Array(16), ["a"]);
  assert.deepEqual(configs.map((config) => [config.decodingMethod, config.maxActivePaths, config.hotwordsFile !== undefined]),
    [["modified_beam_search", 4, false], ["modified_beam_search", 4, true]], "the shared engine's greedy default is overridden");
  assert.deepEqual([empty.comparison?.decodingMethod, empty.hotwords?.count, empty.hotwords?.revision],
    ["modified_beam_search", 0, lines([])]);
  assert.deepEqual([listed.comparison?.decodingMethod, listed.hotwords?.count], ["modified_beam_search", 1]);
});

test("a names snapshot is immutable, empty has no hotwords, and the fixed runtime rejects the experimental decoder", () => {
  const snapshot = namesSnapshot(["a", "b"]);
  assert.equal(snapshot.content, "a\nb\n");
  assert.throws(() => snapshot.hotwords!.phrases.push("extra"), TypeError);
  assert.throws(() => { snapshot.hotwords!.score = 9; }, TypeError);
  assert.equal(snapshot.hotwords!.score, CPU_DECODING.hotwordsScore);
  assert.deepEqual(namesSnapshot([]), { source: "names", content: "", hotwords: undefined });
  const parent = { ...bindingWith({}), version: "2.0.1", gitSha1: "af2d8d81", onnxruntimeVersion: "1.28.2" };
  validateCpuRuntime(parent);
  assert.throws(() => validateCpuRuntime({ ...parent, version: "2.0.2" }), /pinned sherpa/);
  assert.throws(() => validateCpuRuntime({ ...parent, gitSha1: "3898de13" }), /pinned sherpa/);
  assert.throws(() => validateCpuRuntime({ ...parent, onnxruntimeVersion: "1.27.0" }), /pinned sherpa/);
});

test("both CPU presets carry no list of their own and load exactly the names they are given", async () => {
  for (const preset of Object.values(CPU_PRESETS)) {
    assert.deepEqual({ ...preset, mode: undefined }, { ...CPU_DECODING, mode: undefined });
  }
  const loaded: string[][] = [];
  const service = new TakeTranscription({ mode: "tm/today",
    async createEngine(hotwords) { loaded.push([...hotwords!.phrases]); throw new Error("stop after the list"); } });
  await assert.rejects(service.prepare(EIGHT), /stop after the list/);
  assert.deepEqual(loaded, [EIGHT]);
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
  type: "ready", version: 1, preset: CPU_PRESETS["sweep/score-2"], load: { loadMs: 7, modelConstructionMs: 3 },
  assets: Object.fromEntries(SHERPA_MODELS[TRANSCRIBE_MODEL_ID].files.map((file) => [file.name, file.sha256])),
};
/** A fake child that answers prepare with ready and a take with metadata for the names it sent. */
function healthyWorker(options: { text?: string; beforeReply?: string; prepare?: string } = {}): string { return `
  let firstUse = true;
  process.on("message", ({ id, type, names, samples }) => {
    if (type === "prepare") { ${options.prepare ?? ""} firstUse = true; process.send({ ...ready, id }); return; }
    ${options.beforeReply ?? ""}
    const revision = createHash("sha256").update(names.map((name) => name + "\\n").join("")).digest("hex");
    process.send({ id, text: ${options.text ?? 'String(process.pid) + ":" + samples.length + ":" + samples[1]'}, decodeMs: 2,
      hotwords: { path: "names", count: names.length, revision },
      comparison: { ...ready.load, processingMs: 1, workerWallMs: 4, firstUse, decodingMethod: "modified_beam_search" } });
    firstUse = false;
  });
`; }
const answerPrepare = `process.on("message", ({ id, type }) => { if (type === "prepare") process.send({ ...ready, id }); });`;
async function fakeChild(t: TestContext, body = healthyWorker(), options: {
  mode?: TranscriptionMode; timeoutMs?: number; startupTimeoutMs?: number; namesSource?: string;
} = {}) {
  const root = await mkdtemp("/tmp/speak-easy-cpu-ipc-");
  const workerPath = join(root, "worker.mjs");
  const write = (source: string) => writeFile(workerPath, `import { createHash } from "node:crypto";\n` +
    `const ready = ${JSON.stringify(ready)};\n${source}\nprocess.on("disconnect", () => process.exit(0));\n`);
  await write(body);
  const child = new ChildTranscriber({ mode: "sweep/score-2", workerPath, startupTimeoutMs: 2_000, ...options });
  t.after(async () => { await child.closeAndWait(); await rm(root, { recursive: true, force: true }); });
  return { child, write, workerPath, root };
}

test("advanced IPC preserves offset PCM, separates fixed instances, and keeps warm first use metadata", async (t) => {
  const a = await fakeChild(t);
  const b = await fakeChild(t);
  const prepared = await a.child.prepare(EIGHT);
  assert.deepEqual(prepared, ready);
  assert.equal(await a.child.prepare([...EIGHT]), prepared, "the same names give the same ready, so the load stays warm");
  const backing = new Float32Array([99, 0.25, 1.5, -0.5, 88]);
  const first = await a.child.transcribe(backing.subarray(1, 4), EIGHT);
  const second = await a.child.transcribe(backing.subarray(1, 4), EIGHT);
  const other = await b.child.transcribe(backing.subarray(1, 4), EIGHT);
  assert.deepEqual(first.hotwords, { path: "names", count: 8, revision: lines(EIGHT) });
  assert.match(first.text, /:3:1.5$/);
  assert.equal(second.text, first.text);
  assert.notEqual(first.text.split(":")[0], other.text.split(":")[0]);
  assert.equal(first.comparison?.firstUse, true);
  assert.equal(second.comparison?.firstUse, false);
  assert.ok(first.comparison!.processStartupMs! >= 0);
  await a.child.closeAndWait();
  const recovered = await a.child.transcribe(new Float32Array([0.25]), EIGHT);
  assert.notEqual(recovered.text.split(":")[0], first.text.split(":")[0]);
});

test("a names change prepares again in the same child, and a result for other names retires it", async (t) => {
  const h = await fakeChild(t, healthyWorker({ prepare: "globalThis.prepares = (globalThis.prepares ?? 0) + 1;",
    text: 'process.pid + ":" + globalThis.prepares' }), { namesSource: "/owner/road-test-names.json" });
  const eight = await h.child.prepare(EIGHT);
  const first = await h.child.transcribe(new Float32Array([0.25]), EIGHT);
  const fewer = await h.child.prepare(["Grok", "Qwen"]);
  assert.notEqual(fewer, eight, "another list is a cold load");
  const second = await h.child.transcribe(new Float32Array([0.25]), ["Grok", "Qwen"]);
  const third = await h.child.transcribe(new Float32Array([0.25]), EIGHT);
  const [pid] = first.text.split(":");
  assert.deepEqual([first.text, second.text, third.text], [`${pid}:1`, `${pid}:2`, `${pid}:3`],
    "each change prepares once, in the same process, and a take prepares its own names first");
  assert.deepEqual(second.hotwords, { path: "/owner/road-test-names.json", count: 2, revision: lines(["Grok", "Qwen"]) });
  assert.equal(second.comparison?.firstUse, true);
  await h.write(`${answerPrepare} process.on("message", ({ id, type, names }) => { if (type !== "prepare") process.send({ id,
    text: "", decodeMs: 0, hotwords: { path: "names", count: names.length + 1, revision: "0" },
    comparison: { ...ready.load, processingMs: 1, workerWallMs: 1, firstUse: true, decodingMethod: "modified_beam_search" } }); });`);
  await h.child.closeAndWait();
  await assert.rejects(h.child.transcribe(new Float32Array(1), EIGHT), /Invalid fixed CPU result metadata/);
  await h.write(`${answerPrepare} process.on("message", ({ id, type, names }) => { if (type !== "prepare") process.send({ id,
    text: "", decodeMs: 0, hotwords: { path: "names", count: names.length, revision: createHash("sha256").update(
      names.map((name) => name + "\\n").join("")).digest("hex") },
    comparison: { ...ready.load, processingMs: 1, workerWallMs: 1, firstUse: true, decodingMethod: "greedy_search" } }); });`);
  await assert.rejects(h.child.transcribe(new Float32Array(1), EIGHT), /Invalid fixed CPU result metadata/,
    "a decoding other than the pinned one is refused");
});

test("a prepare the child cannot build fails with its reason and retires the child", async (t) => {
  const h = await fakeChild(t, `process.on("message", ({ id, names }) => process.send({ id,
    error: "en-parakeet-ultra-int8: hotword " + JSON.stringify(names.at(-1)) + ": unencodable UTF8 coverage" }));`);
  await assert.rejects(h.child.prepare(["Grok", "Qw☃n"]), /hotword "Qw☃n": unencodable/);
  await h.write(healthyWorker());
  assert.deepEqual(await h.child.prepare(["Grok"]), ready, "a fresh child prepares the corrected list");
  await assert.rejects(h.child.prepare(["Gr\nok"]), /CPU names/, "a name that would split a hotword line never reaches the child");
});

test("fixed CPU owns exact PCM before awaiting cold preparation", async (t) => {
  const h = await fakeChild(t, healthyWorker({
    text: 'Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength).toString("hex")',
  }));
  const backing = new Float32Array([99, -0, 1.5, -2, 88]);
  const samples = backing.subarray(1, 4);
  const expected = Buffer.from(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)).toString("hex");
  const pending = h.child.transcribe(samples, EIGHT);
  backing.fill(7);
  assert.equal((await pending).text, expected);
});

test("scheduler CPU cancellation awaits actual exit before queued exact PCM reaches a fresh child", async (t) => {
  const h = await fakeChild(t, healthyWorker({
    text: 'process.pid + ":" + Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength).toString("hex")',
    beforeReply: `
      if (samples[0] === 0.25) {
        process.getBuiltinModule("fs").writeFileSync(new URL("./active.json", import.meta.url), JSON.stringify({ pid: process.pid }));
        return;
      }
    `,
  }));
  const executor = new SetupWorkers({ cpu: { "sweep/score-2": h.child } });
  const scheduler = new SetupComparison({ executor });
  t.after(() => scheduler.close());
  const first = scheduler.compare({ samples: new Float32Array([0.25]), setupIds: ["sweep/score-2"] });
  const samples = new Float32Array([-0, 1.5, -2]);
  const expected = Buffer.from(samples.buffer).toString("hex");
  const queued = scheduler.compare({ samples, setupIds: ["sweep/score-2"] });
  samples.fill(7);
  let activePid = 0;
  const deadline = performance.now() + 3_000;
  while (!activePid && performance.now() < deadline) {
    try { activePid = JSON.parse(await readFile(join(h.root, "active.json"), "utf8")).pid; }
    catch { await new Promise((resolve) => setTimeout(resolve, 5)); }
  }
  assert.ok(activePid > 0, "the first CPU child reached its blocked operation");
  first.cancel();
  const cancelled = await first.done;
  assert.equal(cancelled.status, "cancelled");
  assert.equal(cancelled.rows[0].status, "cancelled");
  assert.throws(() => process.kill(activePid, 0), { code: "ESRCH" });
  const completed = await queued.done;
  assert.equal(completed.rows[0].status, "complete");
  const [pid, hex] = completed.rows[0].text!.split(":");
  assert.notEqual(Number(pid), activePid);
  assert.equal(hex, expected);
  await scheduler.close();
  assert.throws(() => process.kill(Number(pid), 0), { code: "ESRCH" });
});

test("legacy takes carry their names over IPC and the reply's path names the host's source", async (t) => {
  const { workerPath } = await fakeChild(t, `
    process.on("message", ({ id, names }) => process.send({ id, text: process.argv.slice(2).join(" ") + JSON.stringify(names),
      decodeMs: 0, hotwords: { path: "names", count: names.length, revision: "abc123" } }));
  `, { mode: "legacy" });
  const legacy = new ChildTranscriber({ workerPath, namesSource: "/owner/road-test-names.json" });
  t.after(() => legacy.closeAndWait());
  const result = await legacy.transcribe(new Float32Array(1), ["Grok", "Qwen"]);
  assert.equal(result.text, '["Grok","Qwen"]', "no file argv; the names travel with the take");
  assert.equal(result.comparison, undefined);
  assert.deepEqual(result.hotwords, { path: "/owner/road-test-names.json", count: 2, revision: "abc123" });
  await assert.rejects(legacy.prepare([]), /requires fixed CPU mode/);
});

test("bad ready, missing ready, and malformed replies retire the child before explicit recovery", async (t) => {
  const h = await fakeChild(t, `process.on("message", ({ id }) =>
    process.send({ ...ready, id, preset: { ...ready.preset, hotwordsScore: 9 } }));`);
  await assert.rejects(h.child.prepare(EIGHT), /preset mismatch/);
  await h.write(healthyWorker());
  assert.match((await h.child.transcribe(new Float32Array([0.25]), EIGHT)).text, /:1:/);
  await h.child.closeAndWait();
  await h.write(`${answerPrepare} process.on("message", ({ id, type }) => {
    if (type !== "prepare") process.send({ id, text: "oops", decodeMs: -1 }); });`);
  await assert.rejects(h.child.transcribe(new Float32Array(1), EIGHT), /Invalid CPU transcript/);
  await h.write(healthyWorker());
  await h.child.transcribe(new Float32Array(1), EIGHT);
  const missing = await fakeChild(t, "", { startupTimeoutMs: 75 });
  await assert.rejects(missing.child.prepare(EIGHT), /ready timed out/);
});

test("processing timeout settles concurrent requests once and a later explicit request recovers", async (t) => {
  const h = await fakeChild(t, answerPrepare, { timeoutMs: 75 });
  const first = h.child.transcribe(new Float32Array(1), EIGHT);
  const second = h.child.transcribe(new Float32Array(1), EIGHT);
  const outcomes = await Promise.allSettled([first, second]);
  assert.ok(outcomes.every((result) => result.status === "rejected" && /timed out/.test(String(result.reason))));
  await h.write(healthyWorker());
  const next = await h.child.transcribe(new Float32Array(1), EIGHT);
  assert.equal(next.comparison?.firstUse, true);
});

test("close during startup and before spawn cannot start an obsolete decode", async (t) => {
  const h = await fakeChild(t, "");
  const preSpawn = h.child.transcribe(new Float32Array(1), EIGHT);
  h.child.close();
  await assert.rejects(preSpawn, /closed/);
  const startup = h.child.prepare(EIGHT);
  await new Promise<void>((resolve) => setImmediate(resolve));
  const rejected = assert.rejects(startup, /closed/);
  await h.child.closeAndWait();
  await rejected;
  await h.write(healthyWorker());
  await h.child.transcribe(new Float32Array(1), EIGHT);
});

test("unknown request ids and oversized replies fail the owning CPU generation", async (t) => {
  const wrongId = await fakeChild(t, `${answerPrepare}
    process.on("message", ({ id, type }) => { if (type !== "prepare") process.send({ id: id + 1, text: "wrong", decodeMs: 0 }); });`);
  await assert.rejects(wrongId.child.transcribe(new Float32Array(1), EIGHT), /unknown request id/);
  const oversized = await fakeChild(t, `${answerPrepare}
    process.on("message", ({ id, type }) => { if (type !== "prepare") process.send({ id, text: "x".repeat(65537), decodeMs: 0 }); });`);
  await assert.rejects(oversized.child.transcribe(new Float32Array(1), EIGHT), /Invalid CPU worker reply/);
});

test("fixed ingress refuses oversized audio before spawning, and the real child refuses missing assets before loading native code", async () => {
  const root = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  const child = new ChildTranscriber({ mode: "sweep/score-2", modelRoot: join(root, "absent") });
  try {
    await assert.rejects(child.transcribe(new Float32Array(960001), EIGHT), /60 seconds/);
    await assert.rejects(child.prepare(EIGHT), /not installed/);
    await assert.rejects(child.prepare([]), /not installed/, "an empty list still checks the assets");
  } finally { await child.closeAndWait(); }
});
