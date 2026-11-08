import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import { resolveSettings, type Settings, type VoiceSetting } from "../../speech-io/src/settings.ts";
import { exactFields, HttpError, readJson, requestId, requestLifetime, requireLoopback, sendHttpError,
  sendJson } from "./http-guards.ts";
import { DEFAULT_NAMES_SNAPSHOT, NamesError, parseNames, type NamesSnapshot } from "./names.ts";
import { IDLE_RELEASE_MS, MAX_TEXT_BYTES, type SetupLoadResult } from "./setup-protocol.ts";
import type { PythonWorkerResult, TextCleanupInput } from "./setup-workers.ts";

/**
 * Text cleanup: an optional pass over a final's own text after its row. Its
 * models have their own ids, never setups; each runs in the Python worker,
 * one request at a time with one queued, and takes no audio lease, so the
 * voice and the next take never wait on it.
 */
export const CLEANUP_MODEL_IDS = Object.freeze(["superwhisper/s1-mini", "apple/foundation-models"] as const);
export type CleanupModelId = (typeof CLEANUP_MODEL_IDS)[number];
export function isCleanupModelId(value: unknown): value is CleanupModelId {
  return CLEANUP_MODEL_IDS.includes(value as CleanupModelId);
}

export type CleanupCard = Readonly<{ runtime: string; size?: string; terms?: string }>;
/** takesNames: the model gets the take's Names list; a model without it takes none. */
export type CleanupModel = Readonly<{ id: CleanupModelId; label: string; card: CleanupCard; settings: readonly VoiceSetting[];
  takesNames?: true }>;
export type CleanupReadiness = { ready: boolean; reason?: string; sizeBytes?: number };

/**
 * Self corrections come first and the examples use words unlikely in real
 * takes: the spike's weekday example leaked into other corrections.
 */
export const DEFAULT_INSTRUCTION = [
  "You clean up one speech to text transcript of a single take. Return only the cleaned take, nothing else.",
  "- Apply self corrections first: when the speaker corrects themselves (\"no\", \"no wait\", \"sorry\", \"I mean\"), the words after the cue replace the words before it. \"X no Y\" means Y; never write \"X, not Y\" or keep X. Write only the final version, without the cue or the replaced words.",
  "- Fix punctuation, capitalisation and sentence breaks.",
  "- Remove filler words (um, uh, er, like, you know) when they carry no meaning.",
  "- Known names: when a word sounds like or is a misspelling of a known name (Shawn for Sean), replace it with the known spelling.",
  "- Never answer, summarise, translate or add content. Keep the speaker's words, numbers and meaning.",
  "- Text under \"Earlier takes\" is context only: never clean, repeat or include it.",
  "",
  "Examples:",
  "feed the walrus no the pelican -> Feed the pelican.",
  "um paint the canoe sorry i mean the kayak -> Paint the kayak.",
  "by walrus i mean the big grey one -> By walrus, I mean the big grey one.",
].join("\n");
export const MAX_EARLIER = 3;
export const MAX_EARLIER_BYTES = 4096;
/** Text, names, earlier takes and the instruction at their bounds, with JSON room to spare. */
export const MAX_CLEANUP_JSON_BYTES = 32_768;

export const CLEANUP_MODELS: readonly CleanupModel[] = Object.freeze([
  // The licence requires this exact name: "S1-mini" by "Superwhisper".
  { id: "superwhisper/s1-mini", label: "S1-mini by Superwhisper",
    card: { runtime: "mlx-lm, BF16 safetensors", terms: "S1-mini licence: Apache 2.0 plus its naming clause" },
    settings: [] },
  { id: "apple/foundation-models", label: "Apple Foundation Models",
    card: { runtime: "Foundation Models, macOS 27", size: "system managed",
      terms: "Apple macOS licence; system managed weights" },
    takesNames: true,
    settings: [
      { id: "instruction", label: "Instruction", note: "What the model is told to do with each take",
        kind: "text", default: DEFAULT_INSTRUCTION, maxLength: 2000, lines: 6 },
      { id: "earlier", label: "Earlier takes as context", note: `Sends up to ${MAX_EARLIER} recent takes, oldest first`,
        kind: "choice", default: false, options: [{ value: false, label: "Off" }, { value: true, label: "On" }] },
    ] },
]);

export function getCleanupModel(id: CleanupModelId): CleanupModel {
  return CLEANUP_MODELS.find((model) => model.id === id)!;
}

export type CleanupWorker = {
  load(signal?: AbortSignal): Promise<SetupLoadResult>;
  cleanText(input: TextCleanupInput, signal?: AbortSignal): Promise<PythonWorkerResult>;
  release(): Promise<void>;
  close(): Promise<void>;
};
/** names: the take's list for a model that takes names; absent for one that takes none. */
export type CleanupRequest = { modelId: CleanupModelId; text: string; settings: Settings; earlier: readonly string[];
  names?: readonly string[] };
export type CleanupSkipCode = "unavailable" | "load" | "execution" | "limit" | "cancelled";
export type CleanupReply =
  | { status: "complete"; text: string; processingMs: number; workerWallMs: number; loadMs: number | null; firstUse: boolean }
  | { status: "skipped"; code: CleanupSkipCode; reason: string };

export class CleanupBusyError extends Error {
  constructor() { super("one cleanup is active and one is queued"); this.name = "CleanupBusyError"; }
}

/** A backend failure a page may show: "<code>: <fixed text>" (runtime/worker.py TEXT_CLEANUPS). */
const FIXED_REASON = /^(unavailable|limit|load|execution): ([\x20-\x7e]{1,200})$/;
const TIMED_OUT = / timed out after (\d+) ms$/;

type Lane = { admitted: number; tail: Promise<void>; idle?: NodeJS.Timeout };

/** One worker per configured cleanup model: serial, warm until idle, a fresh generation after any failure. */
export class TextCleanups {
  readonly #workers: Partial<Record<CleanupModelId, CleanupWorker>>;
  readonly #readiness: () => Partial<Record<CleanupModelId, CleanupReadiness>>;
  readonly #idleReleaseMs: number;
  readonly #lanes = new Map<CleanupModelId, Lane>();
  readonly #controller = new AbortController();

  constructor(options: { workers?: Partial<Record<CleanupModelId, CleanupWorker>>;
    readiness?(): Partial<Record<CleanupModelId, CleanupReadiness>>; idleReleaseMs?: number } = {}) {
    this.#workers = options.workers ?? {};
    this.#readiness = options.readiness ?? (() => ({}));
    this.#idleReleaseMs = options.idleReleaseMs ?? IDLE_RELEASE_MS;
  }

  /** The inspected files' readiness; ready only with a worker to run it. */
  readiness(modelId: CleanupModelId): CleanupReadiness {
    const inspected = this.#readiness()[modelId];
    if (inspected && (!inspected.ready || this.#workers[modelId])) return inspected;
    return { ready: false, reason: "runtime not configured" };
  }

  /** Skips without a spawn when files are missing; a third request for one model is refused. */
  async clean(request: CleanupRequest, signal?: AbortSignal): Promise<CleanupReply> {
    const readiness = this.readiness(request.modelId);
    const worker = this.#workers[request.modelId];
    if (!readiness.ready || !worker) {
      return { status: "skipped", code: "unavailable", reason: readiness.reason ?? "needs setup" };
    }
    const lane = this.#lane(request.modelId);
    if (lane.admitted >= 2) throw new CleanupBusyError();
    lane.admitted++;
    clearTimeout(lane.idle);
    const previous = lane.tail;
    let finish!: () => void;
    lane.tail = new Promise((resolve) => { finish = resolve; });
    const work = AbortSignal.any([this.#controller.signal, ...(signal ? [signal] : [])]);
    let stage: "load" | "execution" = "load";
    try {
      await previous;
      if (work.aborted) return cancelled();
      const loaded = await worker.load(work);
      stage = "execution";
      const result = await worker.cleanText(workerInput(request), work);
      return { status: "complete", text: result.text, processingMs: result.processingMs,
        workerWallMs: result.workerWallMs, loadMs: loaded.startupMs === null ? loaded.loadMs : null,
        firstUse: result.firstUse };
    } catch (error) {
      return work.aborted ? cancelled() : skipped(error, stage);
    } finally {
      lane.admitted--;
      if (lane.admitted === 0 && !this.#controller.signal.aborted) {
        lane.idle = setTimeout(() => { if (lane.admitted === 0) void worker.release().catch(() => {}); },
          this.#idleReleaseMs);
        lane.idle.unref();
      }
      finish();
    }
  }

  async close(): Promise<void> {
    this.#controller.abort();
    for (const lane of this.#lanes.values()) clearTimeout(lane.idle);
    const results = await Promise.allSettled(Object.values(this.#workers).map((worker) => worker.close()));
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  #lane(modelId: CleanupModelId): Lane {
    let lane = this.#lanes.get(modelId);
    if (!lane) this.#lanes.set(modelId, lane = { admitted: 0, tail: Promise.resolve() });
    return lane;
  }
}

/** The worker header: text alone for a model without settings, else its instruction, names and earlier takes. */
function workerInput({ text, settings, earlier, names }: CleanupRequest): TextCleanupInput {
  return { text,
    ...(typeof settings.instruction === "string" ? { instruction: settings.instruction } : {}),
    ...(names ? { names } : {}),
    ...(earlier.length ? { earlier } : {}) };
}

function cancelled(): CleanupReply { return { status: "skipped", code: "cancelled", reason: "cleanup cancelled" }; }

/** Kinds and fixed text only: a worker message is shown only in the fixed reason form. */
function skipped(error: unknown, stage: "load" | "execution"): CleanupReply {
  const message = error instanceof Error ? error.message : "";
  const fixed = FIXED_REASON.exec(message);
  if (fixed) return { status: "skipped", code: fixed[1] as CleanupSkipCode, reason: fixed[2]! };
  const timedOut = TIMED_OUT.exec(message);
  return { status: "skipped", code: stage, reason: timedOut ? `timed out after ${Math.round(Number(timedOut[1]) / 1000)} s`
    : stage === "load" ? "the model did not load" : "the model failed" };
}

const ROUTE = "/prompt-reader/cleanups";

/** GET lists the cleanup models; POST cleans one final's text and answers complete or skipped. */
export class CleanupRoutes {
  readonly #cleanups: TextCleanups;
  readonly #names: () => NamesSnapshot;
  readonly #pending = new Set<Promise<void>>();
  #closed = false;

  /** names: the current list, for a request that sends none. */
  constructor(cleanups: TextCleanups, names: () => NamesSnapshot = () => DEFAULT_NAMES_SNAPSHOT) {
    this.#cleanups = cleanups;
    this.#names = names;
  }

  handles(path: string): boolean { return path === ROUTE; }

  metadata() {
    return { schema: 1, models: CLEANUP_MODELS.map((model) => ({ id: model.id, label: model.label, card: model.card,
      settings: model.settings, ...(model.takesNames ? { takesNames: true } : {}),
      readiness: this.#cleanups.readiness(model.id) })) };
  }

  serve(request: IncomingMessage, response: ServerResponse): void {
    const task = this.#serve(request, response).catch((error) => {
      if (error instanceof CleanupBusyError) sendJson(response, 429, { code: "CLEANUP_BUSY", error: error.message });
      else sendHttpError(response, error);
    });
    this.#pending.add(task);
    void task.finally(() => this.#pending.delete(task));
  }

  async close(): Promise<void> {
    this.#closed = true;
    const closed = this.#cleanups.close();
    await Promise.allSettled(this.#pending);
    await closed;
  }

  async #serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    if (this.#closed) throw new HttpError(503, "Speech server is closing");
    requireLoopback(request, request.method !== "GET");
    if (request.method === "GET") {
      sendJson(response, 200, this.metadata());
      return;
    }
    if (request.method !== "POST") throw new HttpError(404, "Not found");
    const value = await readJson(request, MAX_CLEANUP_JSON_BYTES);
    exactFields(value, ["requestId", "modelId", "text", "settings", "earlier", "names"]);
    const id = requestId(value.requestId, randomUUID);
    const cleanup = parseCleanup(value, this.#names().names);
    const lifetime = requestLifetime(request, response);
    try {
      const reply = await this.#cleanups.clean(cleanup, lifetime.signal);
      sendJson(response, 200, { requestId: id, modelId: cleanup.modelId, ...reply,
        ...(reply.status === "complete" ? { settings: cleanup.settings } : {}) });
    } finally { lifetime.dispose(); }
  }
}

/**
 * The request's model, bounded text, its settings resolved against the
 * model's declarations, earlier takes and, for a model that takes names, the
 * take's list or else `current`.
 */
export function parseCleanup(value: Record<string, unknown>, current: readonly string[] = DEFAULT_NAMES_SNAPSHOT.names): CleanupRequest {
  if (!isCleanupModelId(value.modelId)) throw new HttpError(400, "unknown cleanup model");
  const model = getCleanupModel(value.modelId);
  if (typeof value.text !== "string" || !value.text.trim()) throw new HttpError(400, "cleanup text must be nonblank");
  if (Buffer.byteLength(value.text, "utf8") > MAX_TEXT_BYTES) throw new HttpError(413, "cleanup text exceeds limit");
  if (!model.settings.length && (value.settings !== undefined || value.earlier !== undefined)) {
    throw new HttpError(400, `${model.label} takes text only`);
  }
  if (!model.takesNames && value.names !== undefined) throw new HttpError(400, `${model.label} takes no names`);
  let names: readonly string[] | undefined;
  if (model.takesNames) {
    try { names = value.names === undefined ? current : parseNames(value.names); }
    catch (error) { throw error instanceof NamesError ? new HttpError(400, error.message) : error; }
  }
  let settings: Settings;
  try { settings = resolveSettings(model.settings, value.settings ?? {}); }
  catch (error) { throw new HttpError(400, error instanceof Error ? error.message : "invalid cleanup settings"); }
  let earlier: readonly string[] = [];
  if (value.earlier !== undefined) {
    if (settings.earlier !== true) throw new HttpError(400, "earlier takes need the earlier setting on");
    if (!Array.isArray(value.earlier) || value.earlier.length > MAX_EARLIER
      || value.earlier.some((take) => typeof take !== "string")) {
      throw new HttpError(400, `earlier takes are at most ${MAX_EARLIER} strings`);
    }
    earlier = value.earlier as string[];
    if (earlier.reduce((total, take) => total + Buffer.byteLength(take, "utf8"), 0) > MAX_EARLIER_BYTES) {
      throw new HttpError(413, "earlier takes exceed limit");
    }
  }
  return { modelId: model.id, text: value.text, settings, earlier, ...(names ? { names } : {}) };
}
