import { readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hasNonEmptyFile } from "./assets.ts";
import type { Hotwords } from "./sherpa-api.ts";
export type { Hotwords } from "./sherpa-api.ts";

/**
 * Optional contextual biasing hotwords for local sherpa callers (experimental).
 *
 * If a `hotwords.txt` sits in the project root, its terms bias the decode
 * toward those phrases. One term/phrase per line; blank lines and `#` comments
 * are ignored. Absent or empty file => baseline behavior (greedy, no bias).
 *
 * Caveats (see MODEL-SWEEP.md hotwords section):
 *   - sherpa applies hotwords ONLY under modified_beam_search, so enabling
 *     them switches the decoder off greedy_search.
 *   - BPE bias needs matching tokenizer data. Offline Ultra reconstructs its
 *     proven vocabulary and validates every phrase; unproven offline models
 *     reject lists. Streaming models without BPE data (e.g. kroko) cannot bias.
 */

/** Default contextual bias weight; higher = stronger pull toward hotwords. */
export const HOTWORDS_SCORE = 2.0;

export const HOTWORDS_FILENAME = "hotwords.txt";

export type HotwordFileSnapshot = {
  source: string;
  /** Exact source text, including comments, for recognizer cache identity. */
  content: string;
  hotwords: Hotwords | undefined;
};

/**
 * Load `hotwords.txt` from the project root if it has usable terms, else
 * undefined. Writes a cleaned copy to a temp path because sherpa's hotwords
 * reader does not skip comments or blank lines.
 * Snapshot mode reads once, returns exact content and never writes a temp
 * file. Only a missing source becomes empty; other read failures reject.
 */
export function loadHotwords(): Promise<Hotwords | undefined>;
export function loadHotwords(options: { snapshot: true; source?: string }): Promise<HotwordFileSnapshot>;
export async function loadHotwords(options?: { snapshot: true; source?: string }): Promise<Hotwords | HotwordFileSnapshot | undefined> {
  const source = options?.source ?? join(process.cwd(), HOTWORDS_FILENAME);
  const content = options?.snapshot
    ? await readFile(source, "utf8").catch((error: unknown) => {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return "";
      throw error;
    })
    : await hasNonEmptyFile(source) ? await readFile(source, "utf8") : "";
  const phrases = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
  const hotwords = phrases.length ? { phrases, file: source, score: HOTWORDS_SCORE } : undefined;
  // Offline engines own their native files. Do not rewrite the shared legacy
  // streaming file when taking a snapshot for an offline recognizer.
  if (options?.snapshot) return { source, content, hotwords };
  if (!hotwords) return undefined;
  const file = join(tmpdir(), "speak-easy-hotwords.txt");
  await writeFile(file, `${phrases.join("\n")}\n`, "utf8");
  return { ...hotwords, file };
}
