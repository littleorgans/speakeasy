import assert from "node:assert/strict";
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
