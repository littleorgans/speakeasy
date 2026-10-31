import { access, link, mkdir, readdir, readFile, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { encodeWavFloat32 } from "../bench/wav.ts";

/**
 * Labeled-corpus storage: wav + json sidecar pairs written by the demo and
 * the browser prompt reader, scored by the bench. Sidecars are plain,
 * stable, hand-editable ground truth: `expected` starts null (or the
 * accepted/corrected transcript when labeled interactively) and can be
 * filled in by hand later.
 *
 * Schema 2 adds `promptId` and `stratum` for takes read from a prompt set,
 * and allows a null `flushToFinalMs` when no engine ran at record time.
 * Schema 1 sidecars still read, as free speech (both fields null).
 */

export const CORPUS_SCHEMA_VERSION = 2;
export const DEFAULT_CORPUS_DIR = "corpus";

/** Prompt ids become filename stems, so they stay path safe. */
const PROMPT_ID = /^[A-Za-z0-9_-]+$/;

export type CorpusSidecar = {
  schema: typeof CORPUS_SCHEMA_VERSION;
  recordedAt: string;
  /** Wav filename next to this sidecar (same stem). */
  audio: string;
  /** What the engine transcribed at record time; "" when none ran. */
  hypothesis: string;
  /** Ground-truth transcript; null = unlabeled (skipped by the scorer). */
  expected: string | null;
  /** Prompt set id this take reads; null for free speech. */
  promptId: string | null;
  /** Prompt stratum (e.g. short, medium, long); null when not prompted. */
  stratum: string | null;
  engineLabel: string;
  endpoint: string;
  /** Release to final at record time; null when no engine ran. */
  flushToFinalMs: number | null;
  device: string;
  peakLevel: number;
};

export type CorpusSidecarMeta = Omit<
  CorpusSidecar,
  "schema" | "audio" | "promptId" | "stratum"
> &
  Partial<Pick<CorpusSidecar, "promptId" | "stratum">>;

export type CorpusEntry = {
  sidecarPath: string;
  wavPath: string;
  sidecar: CorpusSidecar;
};

/**
 * Write a wav + sidecar pair; the stem is derived from recordedAt, prefixed
 * by the prompt id when there is one.
 */
export async function saveCorpusPair(
  dir: string,
  frames: Float32Array[],
  sampleRate: number,
  meta: CorpusSidecarMeta,
): Promise<{ wavPath: string; sidecarPath: string }> {
  const promptId = meta.promptId ?? null;
  if (promptId !== null && !PROMPT_ID.test(promptId)) {
    throw new Error(`promptId must match ${PROMPT_ID}; received ${JSON.stringify(promptId)}`);
  }
  await mkdir(dir, { recursive: true });
  const wavBytes = encodeWavFloat32(concatFrames(frames), sampleRate);
  const stamp = meta.recordedAt.replace(/:/g, "-");
  const base = promptId === null ? `utt-${stamp}` : `utt-${promptId}-${stamp}`;

  for (let attempt = 1; ; attempt += 1) {
    const stem = attempt === 1 ? base : `${base}-${attempt}`;
    const wavPath = join(dir, `${stem}.wav`);
    const sidecarPath = join(dir, `${stem}.json`);
    try {
      await access(sidecarPath);
      continue;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    try {
      // wx claims the stem atomically; a same-millisecond collision retries.
      await writeFile(wavPath, wavBytes, { flag: "wx", flush: true });
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
      promptId,
      stratum: meta.stratum ?? null,
      engineLabel: meta.engineLabel,
      endpoint: meta.endpoint,
      flushToFinalMs: meta.flushToFinalMs,
      device: meta.device,
      peakLevel: meta.peakLevel,
    };
    // The final sidecar is the commit marker. A crash leaves an unlisted
    // wav or .part, preserving earlier pairs and reserving this stem on retry.
    const pendingPath = `${sidecarPath}.part`;
    await writeFile(pendingPath, `${JSON.stringify(sidecar, null, 2)}\n`, { flag: "wx", flush: true });
    try {
      await link(pendingPath, sidecarPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") continue;
      throw error;
    }
    await unlink(pendingPath);
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

/**
 * Validate the fields the scorer depends on; sidecars are hand-edited.
 * Schema 1 upgrades in memory to schema 2 with no prompt.
 */
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
  if (record.schema === 1) {
    record.schema = CORPUS_SCHEMA_VERSION;
    record.promptId = null;
    record.stratum = null;
  } else if (record.schema !== CORPUS_SCHEMA_VERSION) {
    throw new Error(
      `${path}: unsupported schema ${JSON.stringify(record.schema)}; expected 1 or ${CORPUS_SCHEMA_VERSION}`,
    );
  }
  for (const field of ["promptId", "stratum"] as const) {
    if (record[field] !== null && typeof record[field] !== "string") {
      throw new Error(`${path}: ${field} must be a string or null`);
    }
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
