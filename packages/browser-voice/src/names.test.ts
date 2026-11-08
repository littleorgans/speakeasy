import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { DEFAULT_NAMES, MAX_NAME_BYTES, MAX_NAMES, NAMES_FILE, namesDigest, NamesError, NamesStore,
  parseNames } from "./names.ts";

async function folder(t: TestContext): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), "speakeasy-names-"));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

test("the defaults are the eight names, with the digest every runtime computes", () => {
  assert.deepEqual(DEFAULT_NAMES, ["Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen"]);
  assert.equal(namesDigest(DEFAULT_NAMES), "59d6626926c652e50c7e90fea5f95afdea75f6c27e11aa07c2996b89d51fc068");
  assert.equal(namesDigest(["Zoë"]), namesDigest(JSON.parse('["Zo\\u00eb"]')), "non ASCII stays unescaped");
  assert.equal(NAMES_FILE, "models/road-test-names.json");
});

test("parseNames trims, drops blanks and repeats keeping the first, and refuses each broken rule by name", () => {
  assert.deepEqual(parseNames([" Grok ", "", "  ", "Claude", "Grok", "grok"]), ["Grok", "Claude", "grok"]);
  assert.deepEqual(parseNames([]), []);
  assert.ok(Object.isFrozen(parseNames(["Grok"])));
  const refused = (value: unknown, pattern: RegExp) =>
    assert.throws(() => parseNames(value), (error) => error instanceof NamesError && pattern.test(error.message), JSON.stringify(value));
  refused("Grok, Claude", /array of strings/);
  refused({ names: [] }, /array of strings/);
  refused(["Grok", 3], /array of strings/);
  refused(["Grok, Claude"], /"Grok, Claude" contains a comma/);
  for (const control of ["\u0000", "\t", "\n", "\u001f", "\u007f", "\u0085", "\u009f"]) refused([`Gr${control}ok`], /control character/);
  refused(["é".repeat(65)], new RegExp(`exceeds ${MAX_NAME_BYTES} UTF8 bytes`));
  assert.equal(parseNames(["é".repeat(64)])[0], "é".repeat(64), "the byte bound, not characters");
  refused(Array.from({ length: MAX_NAMES + 1 }, (_, index) => `n${index}`), /at most 64 names/);
  assert.equal(parseNames([...Array.from({ length: MAX_NAMES }, (_, index) => `n${index}`), "n0", " "]).length, MAX_NAMES,
    "repeats and blanks do not count");
});

test("a store without a file holds the defaults in memory and never writes", async () => {
  const store = await NamesStore.open();
  assert.equal(store.file, undefined);
  assert.deepEqual(store.current().names, DEFAULT_NAMES);
  assert.deepEqual((await store.replace(["Qwen"])).names, ["Qwen"]);
  assert.deepEqual(store.current(), { names: ["Qwen"], digest: namesDigest(["Qwen"]) });
});

test("a missing file seeds the defaults; a save persists atomically and a reopen reads it", async (t) => {
  const root = await folder(t);
  const file = join(root, "models", "names.json");
  const store = await NamesStore.open(file);
  assert.deepEqual(store.current().names, DEFAULT_NAMES);
  await assert.rejects(readFile(file), { code: "ENOENT" }, "seeding writes nothing");
  const saved = await store.replace([" Qwen", "GPT-SOL", "Qwen"]);
  assert.deepEqual(saved, { names: ["Qwen", "GPT-SOL"], digest: namesDigest(["Qwen", "GPT-SOL"]) });
  assert.deepEqual(JSON.parse(await readFile(file, "utf8")), { names: ["Qwen", "GPT-SOL"] });
  assert.deepEqual(await readdir(join(root, "models")), ["names.json"], "no temp file stays");
  assert.deepEqual((await NamesStore.open(file)).current(), saved, "a restart keeps the list");
  await assert.rejects(store.replace(["a,b"]), NamesError);
  assert.deepEqual(store.current(), saved, "a refused edit changes nothing");
  assert.deepEqual(JSON.parse(await readFile(file, "utf8")), { names: ["Qwen", "GPT-SOL"] });
});

test("an invalid or oversized file fails open with its path and reason", async (t) => {
  const root = await folder(t);
  for (const [content, reason] of [["not json", /invalid/], ['["Grok"]', /expected/], ['{"names":["a,b"]}', /comma/],
    ['{"names":[],"extra":1}', /unsupported fields/], [`{"names":["${"x".repeat(70_000)}"]}`, /exceeds limit/]] as const) {
    const file = join(root, "names.json");
    await writeFile(file, content);
    await assert.rejects(NamesStore.open(file), (error: Error) => error.message.startsWith(`names file ${file} is invalid`)
      && reason.test(error.message), content.slice(0, 40));
  }
});

test("saves apply in order, last writer wins, and a failed write leaves the list unchanged", async (t) => {
  const root = await folder(t);
  const file = join(root, "names.json");
  const store = await NamesStore.open(file);
  const saves = await Promise.all([store.replace(["A"]), store.replace(["B"]), store.replace(["C"])]);
  assert.deepEqual(saves.map((saved) => saved.names), [["A"], ["B"], ["C"]]);
  assert.deepEqual(store.current().names, ["C"]);
  assert.deepEqual(JSON.parse(await readFile(file, "utf8")), { names: ["C"] });
  const blocked = await NamesStore.open(join(root, "names.json", "child.json")).catch((error: NodeJS.ErrnoException) => error);
  assert.equal((blocked as NodeJS.ErrnoException).code, "ENOTDIR");
  const unwritable = await NamesStore.open(join(root, "folder", "names.json"));
  await writeFile(join(root, "folder"), "a file where the folder should be");
  await assert.rejects(unwritable.replace(["D"]));
  assert.deepEqual(unwritable.current().names, DEFAULT_NAMES);
  await rm(join(root, "folder"));
  assert.deepEqual((await unwritable.replace([])).names, [], "a later save still applies");
});
