import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { loadHotwords } from "./hotwords.ts";

test("file snapshots reload exact content with existing comment and blank handling", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "reader-hotwords-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "hotwords.txt");
  const content = "# owner list\r\n  alpha  \r\n\r\nalpha beta\n";
  await writeFile(source, content);
  const snapshot = await loadHotwords({ snapshot: true, source });
  assert.equal(snapshot.source, source);
  assert.equal(snapshot.content, content);
  assert.deepEqual(snapshot.hotwords?.phrases, ["alpha", "alpha beta"]);
  assert.equal(snapshot.hotwords?.file, source);
  assert.equal(await readFile(source, "utf8"), content, "the owner file is never rewritten");
  await writeFile(source, "beta\n");
  assert.deepEqual((await loadHotwords({ snapshot: true, source })).hotwords?.phrases, ["beta"]);
});

test("empty, comments only and missing snapshots carry no list", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "reader-empty-hotwords-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "hotwords.txt");
  const missing = await loadHotwords({ snapshot: true, source });
  assert.equal(missing.hotwords, undefined);
  for (const content of ["", "# empty\n \n"]) {
    await writeFile(source, content);
    const snapshot = await loadHotwords({ snapshot: true, source });
    assert.equal(snapshot.content, content);
    assert.equal(snapshot.hotwords, undefined);
  }
});

test("an unreadable snapshot rejects instead of silently becoming no list", async (t) => {
  const source = await mkdtemp(join(tmpdir(), "reader-invalid-source-"));
  t.after(() => rm(source, { recursive: true, force: true }));
  await assert.rejects(loadHotwords({ snapshot: true, source }), { code: "EISDIR" });
});
