import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { createSherpaEngine } from "./sherpa-factory.ts";
import { SherpaOfflineEngine } from "./sherpa-offline.ts";
import type { Hotwords, OfflineRecognizerConfig } from "./sherpa-api.ts";
import { resolveModelPaths, SHERPA_MODELS, type SherpaModelId } from "./sherpa-models.ts";
import { bindingWith, seededModelRoot } from "./test-support.ts";

const ULTRA = "en-parakeet-ultra-int8";
// Explicit scored fixture, independent of downloaded models.
const pieces = ["<unk>", "<s>", "</s>", ...Array.from({ length: 271 }, (_, i) => `<control${i}>`),
  "en", "▁", "a", "b", "▁a", "ab", "A", ".", "▁b", "▁A", "<blk>"];
const tokens = pieces.map((piece, id) => `${piece} ${id}\n`).join("");

async function setup(options: {
  phrases?: string[];
  score?: number;
  table?: string | null;
  model?: SherpaModelId;
  factory?: boolean;
  failConstruction?: boolean;
} = {}) {
  const model = options.model ?? ULTRA;
  const modelRoot = await seededModelRoot(model);
  const tokenFile = join(resolveModelPaths(SHERPA_MODELS[model], modelRoot).dir, "tokens.txt");
  if (options.table !== null) await writeFile(tokenFile, options.table ?? tokens);
  const configs: OfflineRecognizerConfig[] = [];
  const files: { words: string; vocab: string; dir: string }[] = [];
  let streams = 0;
  let fail = options.failConstruction;
  const hotwords: Hotwords | undefined = options.phrases === undefined ? undefined : {
    phrases: options.phrases, file: join(modelRoot, "untrusted-words.txt"), score: options.score ?? 2,
  };
  const binding = bindingWith({ OfflineRecognizer: class {
    constructor(config: OfflineRecognizerConfig) {
      configs.push(config);
      if (config.hotwordsFile) {
        files.push({ words: readFileSync(config.hotwordsFile, "utf8"),
          vocab: readFileSync(config.modelConfig.bpeVocab!, "utf8"), dir: dirname(config.hotwordsFile) });
      }
      if (fail) { fail = false; throw new Error("constructor failed"); }
    }
    createStream() { streams += 1; return { acceptWaveform() {} }; }
    async decodeAsync() { return { text: "a" }; }
  } });
  const engineOptions = { model, modelRoot, binding, hotwords };
  const engine = options.factory ? createSherpaEngine(engineOptions) : new SherpaOfflineEngine(engineOptions);
  return { engine, hotwords, configs, files, tokenFile, streams: () => streams };
}

test("offline absent and empty lists stay greedy without token data", async () => {
  for (const phrases of [undefined, []]) {
    const { engine, configs } = await setup({ phrases, table: null });
    await engine.prepare();
    await engine.open();
    assert.equal(configs[0]?.decodingMethod, "greedy_search");
    assert.equal(configs[0]?.modelConfig.modelingUnit, undefined);
    assert.equal(configs[0]?.modelConfig.bpeVocab, undefined);
    assert.equal(configs[0]?.hotwordsFile, undefined);
    assert.equal(configs[0]?.maxActivePaths, undefined);
  }
});

test("Ultra factory and direct construction initialize beam from validated private files", async (t) => {
  const warnings: unknown[][] = [];
  t.mock.method(console, "warn", (...args: unknown[]) => warnings.push(args));
  for (const factory of [false, true]) {
    const { engine, configs, files, streams } = await setup({ phrases: ["a", "A ab."], score: 3, factory });
    await engine.prepare();
    assert.equal(configs.length, 0);
    for (let i = 0; i < 2; i += 1) {
      const session = await engine.open();
      session.pushAudio(new Float32Array(160));
      await session.end();
    }
    assert.equal(configs.length, 1);
    assert.equal(streams(), 2);
    assert.equal(configs[0]?.decodingMethod, "modified_beam_search");
    assert.equal(configs[0]?.modelConfig.modelingUnit, "bpe");
    assert.equal(configs[0]?.modelConfig.provider, "cpu");
    assert.equal(configs[0]?.maxActivePaths, 4);
    assert.equal(configs[0]?.hotwordsScore, 3);
    assert.equal(files[0]?.words, "a\nA ab.\n");
    assert.ok(files[0]?.vocab.includes("en\t-0\n▁\t-1\n"));
    assert.ok(!files[0]?.vocab.includes("<blk>"));
    assert.equal(existsSync(files[0]!.dir), false);
  }
  assert.deepEqual(warnings, []);
});

test("offline snapshots phrases and score, and caches successful preparation", async () => {
  const { engine, hotwords, files, tokenFile } = await setup({ phrases: ["a"] });
  hotwords!.phrases[0] = "☃";
  hotwords!.score = NaN;
  await writeFile(hotwords!.file, "☃\n");
  await engine.prepare();
  await writeFile(tokenFile, "malformed");
  await engine.open();
  assert.equal(files[0]?.words, "a\n");
});

test("offline snapshots recreate an absent temp parent and retain private cleanup", async (t) => {
  const root = await mkdtemp("/tmp/speak-easy-missing-temp-");
  const parent = join(root, "removed", "nested");
  const first = await setup({ phrases: ["a"] });
  const second = await setup({ phrases: ["b"], failConstruction: true });
  const previous = process.env.TMPDIR;
  t.after(async () => {
    if (previous === undefined) delete process.env.TMPDIR;
    else process.env.TMPDIR = previous;
    await rm(root, { recursive: true, force: true });
  });
  process.env.TMPDIR = parent;
  assert.equal(existsSync(parent), false, "temp parent is absent at recognizer creation");
  await first.engine.open();
  await assert.rejects(second.engine.open(), /constructor failed/);
  await second.engine.open();
  const files = [...first.files, ...second.files];
  assert.deepEqual(files.map(({ words }) => words), ["a\n", "b\n", "b\n"]);
  assert.ok(files.every(({ vocab, dir }) => vocab.includes("en\t-0\n▁\t-1\n") &&
    dirname(dir) === parent && !existsSync(dir)));
  assert.equal(new Set(files.map(({ dir }) => dir)).size, 3);
  assert.deepEqual(await readdir(parent), []);
});

test("offline validation rejects the entire list before native construction", async () => {
  for (const [phrase, reason] of [
    ["ab☃", /unknown|unencodable|coverage/],
    ["<unk>", /reserved|control/],
    ["<blk>", /reserved|blank/],
    ["a :2", /score|syntax/],
    ["a\nb", /line|control|syntax/],
    ["▁a", /roundtrip/],
  ] as const) {
    const { engine, configs } = await setup({ phrases: ["a", phrase] });
    await assert.rejects(engine.prepare(), (error: Error) => {
      assert.ok(error.message.includes(ULTRA) || error.message.includes(SHERPA_MODELS[ULTRA].name));
      assert.ok(error.message.includes(JSON.stringify(phrase)));
      assert.match(error.message, reason);
      return true;
    });
    await assert.rejects(engine.open());
    assert.equal(configs.length, 0);
  }
});

test("offline preflight rejects missing and malformed tokens and nonfinite scores", async () => {
  for (const table of [null, "bad", tokens.replace("a 276", "a 275"), tokens.replace("b 277", "a 277")]) {
    const { engine, configs } = await setup({ phrases: ["a"], table });
    await assert.rejects(engine.open(), /tokens|token table/);
    assert.equal(configs.length, 0);
  }
  for (const score of [NaN, Infinity, -Infinity]) {
    const { engine, configs } = await setup({ phrases: ["a"], score });
    await assert.rejects(engine.prepare(), /finite.*score|score.*finite/);
    assert.equal(configs.length, 0);
  }
});

test("unproven offline tokenizers reject lists and preserve their baseline", async () => {
  for (const model of ["en-parakeet-tdt-0.6b-v2-int8", "en-moonshine-base-int8"] as const) {
    await assert.rejects(async () => {
      const { engine } = await setup({ model, phrases: ["a"], factory: true });
      await engine.open();
    }, /unsupported.*hotword|hotword.*unsupported/i);
    const { engine, configs } = await setup({ model, phrases: [], table: null });
    await engine.open();
    assert.equal(configs[0]?.decodingMethod, "greedy_search");
  }
});

test("independent preparations own their files; construction failure cleans up and retries", async () => {
  const first = await setup({ phrases: ["a"], failConstruction: true });
  const second = await setup({ phrases: ["b"] });
  await Promise.all([first.engine.prepare(), second.engine.prepare()]);
  await assert.rejects(first.engine.open(), /constructor failed/);
  await Promise.all([first.engine.open(), second.engine.open(), first.engine.open()]);
  assert.equal(first.configs.length, 2);
  assert.equal(second.configs.length, 1);
  const allFiles = [...first.files, ...second.files];
  assert.equal(new Set(allFiles.map(({ dir }) => dir)).size, 3);
  assert.deepEqual(first.files.map(({ words }) => words), ["a\n", "a\n"]);
  assert.equal(second.files[0]?.words, "b\n");
  assert.ok(allFiles.every(({ dir }) => !existsSync(dir)));
  assert.equal(await readFile(first.tokenFile, "utf8"), tokens);
});
