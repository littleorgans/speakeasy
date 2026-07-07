import { readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hasNonEmptyFile } from "./assets.ts";

/**
 * Optional contextual-biasing hotwords for the SherpaEngine (experimental).
 *
 * If a `hotwords.txt` sits in the project root, its terms bias the decode
 * toward those phrases. One term/phrase per line; blank lines and `#` comments
 * are ignored. Absent or empty file => baseline behavior (greedy, no bias).
 *
 * Caveats (see MODEL-SWEEP.md hotwords section):
 *   - sherpa applies hotwords ONLY under modified_beam_search, so enabling
 *     them switches the decoder off greedy_search.
 *   - BPE models tokenize each hotword via the SentencePiece bpe model; a
 *     model shipped without one (e.g. kroko) cannot tokenize hotwords.
 */

/** Default contextual bias weight; higher = stronger pull toward hotwords. */
export const HOTWORDS_SCORE = 2.0;

const HOTWORDS_FILENAME = "hotwords.txt";

export type Hotwords = {
  /** Cleaned phrases, one per hotword. */
  phrases: string[];
  /** Path to the cleaned file handed to sherpa (comments/blanks stripped). */
  file: string;
  score: number;
};

/**
 * Load `hotwords.txt` from the project root if it has usable terms, else
 * undefined. Writes a cleaned copy to a temp path because sherpa's hotwords
 * reader does not skip comments or blank lines.
 */
export async function loadHotwords(): Promise<Hotwords | undefined> {
  const source = join(process.cwd(), HOTWORDS_FILENAME);
  if (!(await hasNonEmptyFile(source))) {
    return undefined;
  }
  const phrases = (await readFile(source, "utf8"))
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
  if (phrases.length === 0) {
    return undefined;
  }
  const file = join(tmpdir(), "speak-easy-hotwords.txt");
  await writeFile(file, `${phrases.join("\n")}\n`, "utf8");
  return { phrases, file, score: HOTWORDS_SCORE };
}
