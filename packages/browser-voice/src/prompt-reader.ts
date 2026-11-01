import { readFile, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { PROMPT_ID, readCorpusEntries, saveCorpusPair } from "@speakeasy/speech-io";

/**
 * Prompt reader: the owner reads a prompt set aloud in the browser, one take
 * per prompt, and each take lands as a schema 2 corpus pair (wav + sidecar
 * with `promptId`, `stratum` and the prompt text as `expected`) in an output
 * folder outside the repo. `pnpm bench --corpus <dir>` scores the folder.
 *
 * The browser sends 16 kHz mono Float32 samples from the capture worklet;
 * prompt text and stratum come from the prompt file, never the client.
 */

export const PROMPT_SAMPLE_RATE = 16_000;
const MAX_TAKE_SECONDS = 120;
/** Longest take the save route accepts, in Float32 bytes. */
export const PROMPT_MAX_TAKE_BYTES =
  PROMPT_SAMPLE_RATE * Float32Array.BYTES_PER_ELEMENT * MAX_TAKE_SECONDS;
const MAX_DEVICE_LABEL = 200;
const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

export type PromptReaderConfig = {
  /** Prompt set JSON: `{ set?, items: [{ id, stratum, text }] }`. */
  promptsPath: string;
  /** Corpus folder for the takes; must sit outside the repo. */
  outputDir: string;
};

export type Prompt = { id: string; stratum: string; text: string };

export type PromptListing = {
  set: string | null;
  outputDir: string;
  /** `takes` lists the prompt's saved audio file names, oldest first. */
  prompts: Array<Prompt & { recorded: boolean; takes: string[] }>;
};

export class UnknownPromptError extends Error {}

export class PromptReader {
  readonly #set: string | null;
  readonly #prompts: Prompt[];
  readonly #outputDir: string;

  private constructor(set: string | null, prompts: Prompt[], outputDir: string) {
    this.#set = set;
    this.#prompts = prompts;
    this.#outputDir = outputDir;
  }

  /** Read and validate the prompt set; refuse an output folder inside the repo. */
  static async load(
    config: PromptReaderConfig,
    repoRoot: string = REPO_ROOT,
  ): Promise<PromptReader> {
    const outputDir = await canonicalPath(resolve(config.outputDir));
    const inside = relative(await realpath(repoRoot), outputDir);
    if (inside === "" || (inside !== ".." && !inside.startsWith(`..${sep}`) && !isAbsolute(inside))) {
      throw new Error(
        `prompt reader output folder must be outside the repo (${repoRoot}); received ${outputDir}`,
      );
    }
    const { set, prompts } = parsePromptSet(
      await readFile(config.promptsPath, "utf8"),
      config.promptsPath,
    );
    return new PromptReader(set, prompts, outputDir);
  }

  /** Prompts in id order, each with the takes the output folder holds for it. */
  async listing(): Promise<PromptListing> {
    const takes = await this.#takes();
    return {
      set: this.#set,
      outputDir: this.#outputDir,
      prompts: this.#prompts.map((prompt) => {
        const audio = takes.filter((take) => take.promptId === prompt.id).map((take) => take.audio);
        return { ...prompt, recorded: audio.length > 0, takes: audio };
      }),
    };
  }

  /** Stable server corpus identity for narrow trusted saved replay grants. */
  get corpusId(): string {
    return createHash("sha256").update(this.#outputDir).digest("hex");
  }

  /**
   * The path of a saved take, only when `audio` is exactly the file a
   * sidecar for one of this set's prompts names; anything else is undefined.
   */
  async takePath(audio: string): Promise<string | undefined> {
    const take = (await this.#takes()).find((candidate) => candidate.audio === audio);
    return take && basename(take.wavPath) === audio ? take.wavPath : undefined;
  }

  /** Saved takes for this set's prompts, sorted by file name (oldest first). */
  async #takes(): Promise<Array<{ promptId: string; audio: string; wavPath: string }>> {
    const ids = new Set(this.#prompts.map((prompt) => prompt.id));
    return (await readEntriesIfPresent(this.#outputDir)).flatMap(({ sidecar, wavPath }) =>
      sidecar.promptId !== null && ids.has(sidecar.promptId)
        ? [{ promptId: sidecar.promptId, audio: sidecar.audio, wavPath }]
        : [],
    );
  }

  async save(
    promptId: string,
    samples: Float32Array,
    device: string,
    preRollMs?: number,
  ): Promise<{ audio: string }> {
    const prompt = this.#prompts.find((candidate) => candidate.id === promptId);
    if (!prompt) {
      throw new UnknownPromptError(`unknown prompt ${JSON.stringify(promptId)}`);
    }
    const saved = await saveCorpusPair(this.#outputDir, [samples], PROMPT_SAMPLE_RATE, {
      recordedAt: new Date().toISOString(),
      hypothesis: "",
      expected: prompt.text,
      promptId: prompt.id,
      stratum: prompt.stratum,
      engineLabel: "none",
      endpoint: "hold-release",
      flushToFinalMs: null,
      device: device.slice(0, MAX_DEVICE_LABEL),
      peakLevel: Number(peakLevel(samples).toFixed(4)),
      preRollMs,
    });
    return { audio: basename(saved.wavPath) };
  }
}

function parsePromptSet(
  raw: string,
  path: string,
): { set: string | null; prompts: Prompt[] } {
  const parsed: unknown = JSON.parse(raw);
  const record = parsed as { set?: unknown; items?: unknown };
  if (typeof parsed !== "object" || parsed === null || !Array.isArray(record.items)) {
    throw new Error(`${path}: prompt set must be an object with an items array`);
  }
  const prompts = record.items.map((item: unknown, index: number): Prompt => {
    const { id, stratum, text } = (item ?? {}) as Record<string, unknown>;
    if (typeof id !== "string" || typeof stratum !== "string" || typeof text !== "string" || !text.trim()) {
      throw new Error(`${path}: items[${index}] needs string id, stratum and non-empty text`);
    }
    if (!PROMPT_ID.test(id)) {
      throw new Error(`${path}: items[${index}] id ${JSON.stringify(id)} must match ${PROMPT_ID} so its take can be saved`);
    }
    return { id, stratum, text };
  });
  if (new Set(prompts.map((prompt) => prompt.id)).size !== prompts.length) {
    throw new Error(`${path}: prompt ids must be unique`);
  }
  prompts.sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  return { set: typeof record.set === "string" ? record.set : null, prompts };
}

async function readEntriesIfPresent(dir: string) {
  try {
    return await readCorpusEntries(dir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

function peakLevel(samples: Float32Array): number {
  let peak = 0;
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
  return peak;
}

/** Resolve existing ancestors too, since the output directory may be new. */
async function canonicalPath(path: string): Promise<string> {
  try {
    return await realpath(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return resolve(await canonicalPath(dirname(path)), basename(path));
  }
}
