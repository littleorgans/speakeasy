import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import {
  CORPUS_SCHEMA_VERSION,
  readCorpusEntries,
  saveCorpusPair,
  type CorpusSidecarMeta,
} from "./store.ts";

const META: CorpusSidecarMeta = {
  recordedAt: "2026-10-31T10:00:00.000Z",
  hypothesis: "",
  expected: "Open .lilo/docs/index.md first.",
  engineLabel: "none",
  endpoint: "hold",
  flushToFinalMs: null,
  device: "test mic",
  peakLevel: 0.5,
};

async function withCorpus(run: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "speakeasy-store-"));
  try {
    await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test("schema 2 pairs carry the prompt id and stratum through a round trip", async () => {
  await withCorpus(async (dir) => {
    const saved = await saveCorpusPair(dir, [new Float32Array(320)], 16_000, {
      ...META,
      promptId: "p042",
      stratum: "medium",
    });
    assert.equal(basename(saved.wavPath), "utt-p042-2026-10-31T10-00-00.000Z.wav");
    const written = JSON.parse(await readFile(saved.sidecarPath, "utf8"));
    assert.equal(written.schema, 2);
    assert.equal(written.promptId, "p042");
    assert.equal(written.stratum, "medium");
    assert.equal(written.flushToFinalMs, null);
    assert.equal("preRollMs" in written, false, "an unset optional field is not written");

    const [entry] = await readCorpusEntries(dir);
    assert.equal(entry?.sidecar.promptId, "p042");
    assert.equal(entry?.sidecar.stratum, "medium");
    assert.equal(entry?.wavPath, saved.wavPath);
  });
});

test("pairs saved without a prompt keep the demo stem and null prompt fields", async () => {
  await withCorpus(async (dir) => {
    const saved = await saveCorpusPair(dir, [new Float32Array(320)], 16_000, {
      ...META,
      flushToFinalMs: 120,
    });
    assert.equal(basename(saved.wavPath), "utt-2026-10-31T10-00-00.000Z.wav");
    const [entry] = await readCorpusEntries(dir);
    assert.equal(entry?.sidecar.promptId, null);
    assert.equal(entry?.sidecar.stratum, null);
  });
});

test("schema 1 sidecars still read, as unprompted schema 2", async () => {
  await withCorpus(async (dir) => {
    await writeFile(
      join(dir, "utt-old.json"),
      JSON.stringify({
        schema: 1,
        recordedAt: "2026-07-01T00:00:00.000Z",
        audio: "utt-old.wav",
        hypothesis: "and so my fellow americans",
        expected: "and so my fellow americans",
        engineLabel: "sherpa",
        endpoint: "manual",
        flushToFinalMs: 80,
        device: ":default",
        peakLevel: 0.3,
      }),
    );
    const [entry] = await readCorpusEntries(dir);
    assert.equal(entry?.sidecar.schema, CORPUS_SCHEMA_VERSION);
    assert.equal(entry?.sidecar.promptId, null);
    assert.equal(entry?.sidecar.stratum, null);
    assert.equal(entry?.sidecar.expected, "and so my fellow americans");
  });
});

test("unknown schemas and malformed prompt fields are refused", async () => {
  await withCorpus(async (dir) => {
    await writeFile(join(dir, "a.json"), JSON.stringify({ schema: 3, audio: "a.wav" }));
    await assert.rejects(readCorpusEntries(dir), /unsupported schema 3; expected 1 or 2/);
  });
  await withCorpus(async (dir) => {
    await writeFile(
      join(dir, "b.json"),
      JSON.stringify({ schema: 2, audio: "b.wav", hypothesis: "", expected: null, promptId: 7, stratum: null }),
    );
    await assert.rejects(readCorpusEntries(dir), /promptId must be a string or null/);
  });
});

test("prompt ids that are not path safe never reach the filesystem", async () => {
  await withCorpus(async (dir) => {
    await assert.rejects(
      saveCorpusPair(dir, [new Float32Array(320)], 16_000, { ...META, promptId: "../escape" }),
      /promptId must match/,
    );
  });
});

test("concurrent same timestamp retakes keep every pair", async () => {
  await withCorpus(async (dir) => {
    const saved = await Promise.all([0.1, 0.2, 0.3].map((value) =>
      saveCorpusPair(dir, [new Float32Array(320).fill(value)], 16_000, { ...META, promptId: "p042" }),
    ));
    assert.equal(new Set(saved.map((pair) => pair.wavPath)).size, 3);
    assert.equal((await readCorpusEntries(dir)).length, 3);
  });
});

test("a crash during sidecar write keeps prior takes readable and retry never overwrites", async () => {
  await withCorpus(async (dir) => {
    const first = await saveCorpusPair(dir, [new Float32Array(320).fill(0.1)], 16_000, META);
    const original = await readFile(first.wavPath);
    const script = `
      import fs from "node:fs/promises";
      import { syncBuiltinESMExports } from "node:module";
      const write = fs.writeFile;
      fs.writeFile = async (path, data, options) => {
        if (/\\.json(?:\\.part)?$/.test(String(path))) {
          await write(path, String(data).slice(0, 20), options);
          process.kill(process.pid, "SIGKILL");
        }
        return write(path, data, options);
      };
      syncBuiltinESMExports();
      const { saveCorpusPair } = await import(${JSON.stringify(new URL("./store.ts", import.meta.url).href)});
      await saveCorpusPair(${JSON.stringify(dir)}, [new Float32Array(320).fill(0.2)], 16000, ${JSON.stringify(META)});
    `;
    const crash = spawnSync(process.execPath, ["--input-type=module", "-"], { input: script, encoding: "utf8" });
    assert.equal(crash.signal, "SIGKILL", crash.stderr);
    assert.equal((await readCorpusEntries(dir)).length, 1);
    const retry = await saveCorpusPair(dir, [new Float32Array(320).fill(0.3)], 16_000, META);
    assert.notEqual(retry.wavPath, first.wavPath);
    assert.deepEqual(await readFile(first.wavPath), original);
    assert.equal((await readCorpusEntries(dir)).length, 2);
  });
});

test("an existing sidecar without its wav is never overwritten by a retake", async () => {
  await withCorpus(async (dir) => {
    const first = await saveCorpusPair(dir, [new Float32Array(320)], 16_000, META);
    const original = await readFile(first.sidecarPath, "utf8");
    await rm(first.wavPath);
    const retry = await saveCorpusPair(dir, [new Float32Array(320)], 16_000, { ...META, expected: "new label" });
    assert.notEqual(retry.sidecarPath, first.sidecarPath);
    assert.equal(await readFile(first.sidecarPath, "utf8"), original);
  });
});
