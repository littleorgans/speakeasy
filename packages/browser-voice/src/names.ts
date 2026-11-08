import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { dirname } from "node:path";
import { exactFields, HttpError, readJson, requireLoopback, sendHttpError, sendJson } from "./http-guards.ts";
import { sha256 } from "./setup-protocol.ts";

/**
 * The one names list: every names consumer on the road test server takes
 * the store's current list, edited from the page's Names control. The host
 * captures one snapshot per take; workers apply exactly that snapshot.
 */
export const DEFAULT_NAMES = Object.freeze(["Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen"]);
/** Gitignored and host owned; resolved from the repo root, where every command runs. */
export const NAMES_FILE = "models/road-test-names.json";
export const MAX_NAMES = 64;
export const MAX_NAME_BYTES = 128;
const MAX_FILE_BYTES = 65_536;
// C0, DEL and C1 controls; rejecting them keeps every runtime's trim and NUL rules idempotent on this output.
const CONTROL = /[\u0000-\u001f\u007f-\u009f]/;

export type NamesSnapshot = Readonly<{ names: readonly string[]; digest: string }>;

/** A list that breaks a rule; its message names the rule and the name. */
export class NamesError extends Error {
  constructor(message: string) { super(message); this.name = "NamesError"; }
}

/** The canonical list: trimmed, blanks and exact duplicates dropped, first kept. Throws a message naming the bad name. */
export function parseNames(value: unknown): readonly string[] {
  if (!Array.isArray(value) || value.some((name) => typeof name !== "string")) {
    throw new NamesError("names must be an array of strings");
  }
  const names: string[] = [];
  for (const raw of value as string[]) {
    const name = raw.trim();
    if (!name || names.includes(name)) continue;
    if (CONTROL.test(name)) throw new NamesError(`name ${JSON.stringify(name)} contains a control character`);
    if (name.includes(",")) throw new NamesError(`name ${JSON.stringify(name)} contains a comma; give one name per entry`);
    if (Buffer.byteLength(name, "utf8") > MAX_NAME_BYTES) {
      throw new NamesError(`name ${JSON.stringify(name)} exceeds ${MAX_NAME_BYTES} UTF8 bytes`);
    }
    names.push(name);
  }
  if (names.length > MAX_NAMES) throw new NamesError(`at most ${MAX_NAMES} names`);
  return Object.freeze(names);
}

/** sha256 of the compact JSON array, as every runtime serializes it. */
export function namesDigest(names: readonly string[]): string {
  return sha256(JSON.stringify(names));
}

export function namesSnapshot(names: readonly string[]): NamesSnapshot {
  return Object.freeze({ names, digest: namesDigest(names) });
}

export const DEFAULT_NAMES_SNAPSHOT = namesSnapshot(DEFAULT_NAMES);

/** The current list, persisted atomically to one file, or held in memory without one. */
export class NamesStore {
  /** The file the list persists to; undefined for an in memory store. */
  readonly file: string | undefined;
  #current: NamesSnapshot;
  #writes: Promise<unknown> = Promise.resolve();

  private constructor(file: string | undefined, current: NamesSnapshot) {
    this.file = file;
    this.#current = current;
  }

  /** A missing file seeds the defaults in memory; an invalid or oversized one fails clearly. */
  static async open(file?: string): Promise<NamesStore> {
    if (file === undefined) return new NamesStore(undefined, DEFAULT_NAMES_SNAPSHOT);
    const info = await stat(file).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined;
      throw error;
    });
    if (!info) return new NamesStore(file, DEFAULT_NAMES_SNAPSHOT);
    try {
      if (info.size > MAX_FILE_BYTES) throw new Error("file exceeds limit");
      const value: unknown = JSON.parse(await readFile(file, "utf8"));
      if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("expected {\"names\": [...]}");
      if (Object.keys(value).some((key) => key !== "names")) throw new Error("unsupported fields");
      return new NamesStore(file, namesSnapshot(parseNames((value as { names?: unknown }).names)));
    } catch (error) {
      throw new Error(`names file ${file} is invalid: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  current(): NamesSnapshot { return this.#current; }

  /** Validates, then writes in order through a temp file and rename; the list changes only after its write. */
  replace(value: unknown): Promise<NamesSnapshot> {
    let snapshot: NamesSnapshot;
    try { snapshot = namesSnapshot(parseNames(value)); } catch (error) { return Promise.reject(error); }
    const write = this.#writes.then(async () => {
      if (this.file !== undefined) await persist(this.file, snapshot.names);
      this.#current = snapshot;
      return snapshot;
    });
    this.#writes = write.catch(() => {});
    return write;
  }
}

async function persist(file: string, names: readonly string[]): Promise<void> {
  await mkdir(dirname(file), { recursive: true });
  const temp = `${file}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, `${JSON.stringify({ names }, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temp, file);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

const ROUTE = "/road-test/names";

/** GET answers the current list and its limits; PUT replaces it, last writer wins. */
export class NamesRoutes {
  readonly #store: NamesStore;

  constructor(store: NamesStore) { this.#store = store; }

  handles(path: string): boolean { return path === ROUTE; }

  serve(request: IncomingMessage, response: ServerResponse): void {
    void this.#serve(request, response).catch((error) => sendHttpError(response, error));
  }

  async #serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    requireLoopback(request, request.method !== "GET");
    if (request.method === "GET") {
      sendJson(response, 200, this.#reply(this.#store.current()));
      return;
    }
    if (request.method !== "PUT") throw new HttpError(404, "Not found");
    const value = await readJson(request);
    exactFields(value, ["names"]);
    let snapshot: NamesSnapshot;
    try { snapshot = await this.#store.replace(value.names); }
    catch (error) { throw error instanceof NamesError ? new HttpError(400, error.message) : error; }
    sendJson(response, 200, this.#reply(snapshot));
  }

  #reply({ names, digest }: NamesSnapshot) {
    return { names, digest, limits: { maxNames: MAX_NAMES, maxNameBytes: MAX_NAME_BYTES } };
  }
}
