import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { test } from "node:test";
import { loadHotwords } from "@speakeasy/speech-io";
import { createSherpaEngine, resolveModelPaths, SHERPA_MODELS } from "@speakeasy/speech-io/stt";
import type { OfflineRecognizerConfig } from "../../speech-io/src/engines/sherpa-api.ts";
import { bindingWith, seededModelRoot } from "../../speech-io/src/engines/test-support.ts";
import { TakeTranscription } from "./take-transcription.ts";
import { readBrowserVoiceOptions } from "./server-options.ts";
import { TRANSCRIBE_MODEL_ID, TranscriptionError } from "./transcriber.ts";

async function setup(decode?: () => Promise<void>, hotwordsPath?: string) {
  const modelRoot = await seededModelRoot(TRANSCRIBE_MODEL_ID);
  const source = join(modelRoot, "hotwords.txt");
  const pieces = ["<unk>", "<s>", "</s>", ...Array.from({ length: 271 }, (_, i) => `<control${i}>`),
    "en", "▁", "a", "b", "<blk>"];
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
  return { service, source, configs, constructedWords, sampleCounts, snapshots, reads: () => reads, created: () => created,
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
