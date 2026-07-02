import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { encodeWavPcm16 } from "../bench/wav.ts";

/**
 * Labeled-corpus storage: wav + json sidecar pairs written by the demo and
 * scored by the bench. Sidecars are plain, stable, hand-editable ground
 * truth: `expected` starts null (or the accepted/corrected transcript when
 * labeled interactively) and can be filled in by hand later.
 */

export const CORPUS_SCHEMA_VERSION = 1;
export const DEFAULT_CORPUS_DIR = "corpus";

export type CorpusSidecar = {
  schema: typeof CORPUS_SCHEMA_VERSION;
  recordedAt: string;
  /** Wav filename next to this sidecar (same stem). */
  audio: string;
  /** What the engine transcribed at record time. */
  hypothesis: string;
  /** Ground-truth transcript; null = unlabeled (skipped by the scorer). */
  expected: string | null;
  engineLabel: string;
  endpoint: string;
  flushToFinalMs: number;
  device: string;
  peakLevel: number;
};

export type CorpusSidecarMeta = Omit<CorpusSidecar, "schema" | "audio">;

export type CorpusEntry = {
  sidecarPath: string;
  wavPath: string;
  sidecar: CorpusSidecar;
};

/** Write a wav + sidecar pair; the stem is derived from recordedAt. */
export async function saveCorpusPair(
  dir: string,
  frames: Float32Array[],
  sampleRate: number,
  meta: CorpusSidecarMeta,
): Promise<{ wavPath: string; sidecarPath: string }> {
  await mkdir(dir, { recursive: true });
  const wavBytes = encodeWavPcm16(concatFrames(frames), sampleRate);
  const base = `utt-${meta.recordedAt.replace(/:/g, "-")}`;

  for (let attempt = 1; ; attempt += 1) {
    const stem = attempt === 1 ? base : `${base}-${attempt}`;
    const wavPath = join(dir, `${stem}.wav`);
    try {
      // wx claims the stem atomically; a same-millisecond collision retries.
      await writeFile(wavPath, wavBytes, { flag: "wx" });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        continue;
      }
      throw error;
    }
    const sidecar: CorpusSidecar = {
      schema: CORPUS_SCHEMA_VERSION,
      recordedAt: meta.recordedAt,
      audio: `${stem}.wav`,
      hypothesis: meta.hypothesis,
      expected: meta.expected,
      engineLabel: meta.engineLabel,
      endpoint: meta.endpoint,
      flushToFinalMs: meta.flushToFinalMs,
      device: meta.device,
      peakLevel: meta.peakLevel,
    };
    const sidecarPath = join(dir, `${stem}.json`);
    await writeFile(sidecarPath, `${JSON.stringify(sidecar, null, 2)}\n`);
    return { wavPath, sidecarPath };
  }
}

/** Read every sidecar in a corpus directory, sorted by filename. */
export async function readCorpusEntries(dir: string): Promise<CorpusEntry[]> {
  const names = (await readdir(dir))
    .filter((name) => name.endsWith(".json"))
    .sort();
  const entries: CorpusEntry[] = [];
  for (const name of names) {
    const sidecarPath = join(dir, name);
    const sidecar = parseSidecar(
      await readFile(sidecarPath, "utf8"),
      sidecarPath,
    );
    entries.push({
      sidecarPath,
      wavPath: join(dir, sidecar.audio),
      sidecar,
    });
  }
  return entries;
}

/** Validate the fields the scorer depends on; sidecars are hand-edited. */
function parseSidecar(raw: string, path: string): CorpusSidecar {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `${path}: invalid JSON (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`${path}: sidecar must be a JSON object`);
  }
  const record = parsed as Record<string, unknown>;
  if (record.schema !== CORPUS_SCHEMA_VERSION) {
    throw new Error(
      `${path}: unsupported schema ${JSON.stringify(record.schema)}; expected ${CORPUS_SCHEMA_VERSION}`,
    );
  }
  if (typeof record.audio !== "string" || record.audio.length === 0) {
    throw new Error(`${path}: audio must be a non-empty wav filename`);
  }
  if (typeof record.hypothesis !== "string") {
    throw new Error(`${path}: hypothesis must be a string`);
  }
  if (record.expected !== null && typeof record.expected !== "string") {
    throw new Error(`${path}: expected must be a string or null`);
  }
  // A hand-edited empty expected means unlabeled, not "expect silence".
  if (typeof record.expected === "string" && record.expected.trim() === "") {
    record.expected = null;
  }
  return record as CorpusSidecar;
}

function concatFrames(frames: Float32Array[]): Float32Array {
  const total = frames.reduce((sum, frame) => sum + frame.length, 0);
  const samples = new Float32Array(total);
  let offset = 0;
  for (const frame of frames) {
    samples.set(frame, offset);
    offset += frame.length;
  }
  return samples;
}
