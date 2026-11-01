import { randomUUID } from "node:crypto";
import { getSetup, parseSetupSelection } from "./setup-registry.ts";
import {
  assertBoundedText, assertSetupExecutionResult, assertSetupLoadResult, CLEANUP_SETUP_ID,
  createCanonicalPcm, IDLE_RELEASE_MS, MAX_TEXT_BYTES, pcmIdentity, SETUP_IDS, sha256, ULTRA_SETUP_ID,
  type CanonicalPcm, type ComparisonEvent, type ComparisonHandle, type ComparisonResult,
  type SetupExecution, type SetupExecutionResult, type SetupExecutor, type SetupFailure,
  type SetupFailureCode, type SetupId, type SetupRow, type SetupRowStatus, type SetupTimings,
} from "./setup-protocol.ts";

export class ComparisonBusyError extends Error {
  constructor() { super("one comparison is active and one is queued"); this.name = "ComparisonBusyError"; }
}

export type ComparisonRequest = {
  samples: Float32Array | Uint8Array; setupIds?: readonly SetupId[]; comparisonId?: string;
  signal?: AbortSignal; onEvent?(event: ComparisonEvent): void;
};
type Job = {
  id: string; pcm: CanonicalPcm; ids: readonly SetupId[]; admitted: number;
  controller: AbortController; rows: Map<SetupId, SetupRow>; onEvent?: ComparisonRequest["onEvent"];
  resolve(result: ComparisonResult): void; done: boolean; detach(): void;
};
type Outcome =
  | { ok: true; result: SetupExecutionResult; timings: SetupTimings }
  | { ok: false; error: SetupFailure; timings: SetupTimings };

function blankTimings(): SetupTimings {
  return { queueMs: null, startupMs: null, loadMs: null, firstUse: null, processingMs: null,
    workerWallMs: null, requestWallMs: null, serviceWallMs: null, dependencyProcessingMs: null, deliveryMs: null };
}

function bounded(value: string, limit = MAX_TEXT_BYTES): string {
  const bytes = Buffer.from(value, "utf8");
  if (bytes.length <= limit) return value;
  let end = limit;
  while ((bytes[end] & 0xc0) === 0x80) end -= 1;
  return bytes.subarray(0, end).toString("utf8");
}

function failure(error: unknown, code: SetupFailureCode): SetupFailure {
  const rawText = error && typeof error === "object" && "rawText" in error ? error.rawText : undefined;
  return { code, message: bounded(error instanceof Error ? error.message : String(error), 2048),
    ...(typeof rawText === "string" ? { rawText: bounded(rawText),
      ...(Buffer.byteLength(rawText, "utf8") > MAX_TEXT_BYTES ? { rawTextTruncated: true } : {}) } : {}) };
}

/** Owns admission, one canonical take, serial model work and a per comparison dependency memo. */
export class SetupComparison {
  readonly #executor: SetupExecutor;
  readonly #now: () => number;
  readonly #idleReleaseMs: number;
  readonly #resident = new Map<SetupId, number>();
  #active: Job | undefined;
  #queued: Job | undefined;
  #pump: Promise<void> | undefined;
  #idleTimer: NodeJS.Timeout | undefined;
  #closed = false;
  #closing: Promise<void> | undefined;

  constructor(options: { executor: SetupExecutor; now?(): number; idleReleaseMs?: number }) {
    this.#executor = options.executor;
    this.#now = options.now ?? (() => performance.now());
    this.#idleReleaseMs = options.idleReleaseMs ?? IDLE_RELEASE_MS;
    if (!Number.isFinite(this.#idleReleaseMs) || this.#idleReleaseMs <= 0) throw new Error("invalid idle release duration");
  }

  compare(request: ComparisonRequest): ComparisonHandle {
    if (this.#closed) throw new Error("comparison service is closed");
    if (this.#active && this.#queued) throw new ComparisonBusyError();
    const ids = parseSetupSelection(request.setupIds);
    const id = request.comparisonId ?? randomUUID();
    if (typeof id !== "string" || !/^[a-zA-Z0-9_.:-]{1,128}$/.test(id)) throw new Error("invalid comparison id");
    if (this.#active?.id === id || this.#queued?.id === id) throw new Error("comparison id already admitted");
    const pcm = createCanonicalPcm(request.samples);
    let resolve!: Job["resolve"];
    const done = new Promise<ComparisonResult>((r) => { resolve = r; });
    const job: Job = { id, pcm, ids, admitted: this.#now(), controller: new AbortController(),
      rows: new Map(), onEvent: request.onEvent, resolve, done: false, detach() {} };
    const queued = this.#active !== undefined;
    if (queued) this.#queued = job;
    else this.#active = job;
    clearTimeout(this.#idleTimer);
    const cancel = () => this.cancel(id);
    request.signal?.addEventListener("abort", cancel, { once: true });
    job.detach = () => request.signal?.removeEventListener("abort", cancel);
    for (const setupId of ids) job.rows.set(setupId, {
      setupId, status: "queued", text: null, pcm: pcmIdentity(pcm), timings: blankTimings(),
      ...(setupId === CLEANUP_SETUP_ID ? { dependency: ULTRA_SETUP_ID } : {}),
    });
    this.#emit(job, { type: "accepted", comparisonId: id, queued, setupIds: ids, pcm: pcmIdentity(pcm) });
    for (const row of job.rows.values()) this.#publish(job, row);
    if (request.signal?.aborted || job.controller.signal.aborted) cancel();
    this.#schedule();
    return { comparisonId: id, done, cancel };
  }

  cancel(comparisonId: string): void {
    const job = this.#active?.id === comparisonId ? this.#active
      : this.#queued?.id === comparisonId ? this.#queued : undefined;
    if (!job || job.done) return;
    job.controller.abort();
    if (job === this.#queued) {
      this.#queued = undefined;
      this.#finish(job);
    }
  }

  close(): Promise<void> {
    if (this.#closing) return this.#closing;
    this.#closed = true;
    clearTimeout(this.#idleTimer);
    if (this.#queued) this.cancel(this.#queued.id);
    if (this.#active) this.cancel(this.#active.id);
    this.#closing = (async () => {
      await this.#pump;
      await this.#executor.close();
      this.#resident.clear();
    })();
    return this.#closing;
  }

  #emit(job: Job, event: ComparisonEvent): void {
    try { job.onEvent?.(Object.freeze(event)); }
    catch { job.controller.abort(); }
  }

  #publish(job: Job, row: SetupRow): void {
    const snapshot = Object.freeze({ ...row, pcm: pcmIdentity(row.pcm),
      timings: Object.freeze({ ...row.timings, serviceWallMs: this.#now() - job.admitted }),
      ...(row.error ? { error: Object.freeze({ ...row.error }) } : {}),
    });
    job.rows.set(row.setupId, snapshot);
    this.#emit(job, { type: "row", comparisonId: job.id, row: snapshot });
  }

  #schedule(): void {
    if (this.#pump) return;
    // Admission is synchronous; all executor calls start on this one pump.
    this.#pump = Promise.resolve().then(async () => {
      await this.#releaseExpired();
      while (this.#active) {
        const job = this.#active;
        try { await this.#run(job); }
        finally { this.#finish(job); }
        this.#active = this.#queued;
        this.#queued = undefined;
      }
    }).finally(() => {
      this.#pump = undefined;
      if (this.#active) this.#schedule();
      else this.#armIdle();
    });
  }

  async #run(job: Job): Promise<void> {
    if (job.controller.signal.aborted) return;
    const queueMs = this.#now() - job.admitted;
    this.#emit(job, { type: "started", comparisonId: job.id, queueMs });
    for (const row of job.rows.values()) job.rows.set(row.setupId, { ...row, timings: { ...row.timings, queueMs } });
    const needed = new Set(job.ids);
    if (needed.has(CLEANUP_SETUP_ID)) needed.add(ULTRA_SETUP_ID);
    // The memo exists only on this stack and is discarded after completion/cancellation.
    const outcomes = new Map<SetupId, Outcome>();
    for (const setupId of SETUP_IDS) {
      if (job.controller.signal.aborted) break;
      if (!needed.has(setupId)) continue;
      const base = outcomes.get(ULTRA_SETUP_ID);
      if (setupId === CLEANUP_SETUP_ID && !base?.ok) {
        this.#terminal(job, setupId, "skipped", blankTimings(), {
          code: "execution", message: "Ultra dependency failed", ...(base && !base.ok ? { rawText: base.error.rawText } : {}),
        });
        continue;
      }
      const outcome = await this.#execute(job, setupId, queueMs, base);
      outcomes.set(setupId, outcome);
      if (job.ids.includes(setupId)) {
        if (outcome.ok) this.#terminal(job, setupId, "complete", outcome.timings, undefined, outcome.result);
        else this.#terminal(job, setupId, job.controller.signal.aborted ? "cancelled" : "error", outcome.timings, outcome.error);
      }
    }
    outcomes.clear();
  }

  async #execute(job: Job, setupId: SetupId, queueMs: number, base?: Outcome): Promise<Outcome> {
    const timings = { ...blankTimings(), queueMs,
      dependencyProcessingMs: setupId === CLEANUP_SETUP_ID && base?.ok ? base.result.processingMs : null };
    if (job.pcm.sampleCount > getSetup(setupId).maxSamples) {
      return { ok: false, timings, error: { code: "limit", message: `${setupId} exceeds its sample limit` } };
    }
    const started = this.#now();
    const signal = job.controller.signal;
    let code: SetupFailureCode = "load";
    try {
      let request: SetupExecution;
      if (setupId === CLEANUP_SETUP_ID) {
        if (!base?.ok) throw new Error("missing Ultra dependency");
        code = "protocol";
        assertBoundedText(base.result.text, "Ultra text");
        request = { operation: "cleanup", requestId: job.id, setupId, base: pcmIdentity(job.pcm),
          baseTextSha256: sha256(base.result.text), A: base.result.text, B: null };
      } else request = { operation: "transcribe", requestId: job.id, setupId, pcm: job.pcm };
      code = "load";
      this.#phase(job, setupId, "loading", timings);
      if (signal.aborted) throw new Error("comparison cancelled");
      const loaded = await this.#executor.load(setupId, { signal });
      code = "protocol";
      assertSetupLoadResult(loaded);
      Object.assign(timings, loaded);
      this.#resident.set(setupId, this.#now());
      if (signal.aborted) throw new Error("comparison cancelled");
      this.#phase(job, setupId, "processing", timings);
      if (signal.aborted) throw new Error("comparison cancelled");
      code = "execution";
      const result = await this.#executor.execute(request, { signal });
      code = "protocol";
      assertSetupExecutionResult(result);
      if (signal.aborted) throw new Error("comparison cancelled");
      Object.assign(timings, { processingMs: result.processingMs, workerWallMs: result.workerWallMs,
        requestWallMs: this.#now() - started });
      this.#resident.set(setupId, this.#now());
      return { ok: true, result, timings };
    } catch (error) {
      this.#resident.delete(setupId);
      // Ensure a corrupt adapter result cannot leave an old model generation reusable.
      const problem = failure(error, signal.aborted ? "cancelled" : code);
      try { await this.#executor.release(setupId); }
      catch (releaseError) {
        // A rejecting release has also awaited exit, as required by SetupExecutor.
        problem.message = bounded(`${problem.message}; release: ${failure(releaseError, "execution").message}`, 2048);
      }
      timings.requestWallMs = this.#now() - started;
      return { ok: false, timings, error: problem };
    }
  }

  #phase(job: Job, setupId: SetupId, status: "loading" | "processing", timings: SetupTimings): void {
    const rowId = job.ids.includes(setupId) ? setupId : CLEANUP_SETUP_ID;
    const row = job.rows.get(rowId)!;
    this.#publish(job, { ...row, status,
      timings: rowId === setupId ? timings : row.timings, activeSetupId: setupId });
  }

  #terminal(job: Job, setupId: SetupId, status: SetupRowStatus, timings: SetupTimings,
    error?: SetupFailure, result?: SetupExecutionResult): void {
    const row = job.rows.get(setupId)!;
    this.#publish(job, { ...row, status, activeSetupId: undefined,
      text: result?.text ?? null, ...(result?.rawText !== undefined ? { rawText: result.rawText } : {}),
      ...(result?.decodeMs !== undefined ? { decodeMs: result.decodeMs } : {}),
      ...(error?.rawText !== undefined ? { rawText: error.rawText } : {}), error,
      timings: { ...timings, queueMs: timings.queueMs ?? row.timings.queueMs } });
  }

  #finish(job: Job): void {
    if (job.done) return;
    const cancelled = job.controller.signal.aborted;
    for (const row of job.rows.values()) {
      if (["queued", "loading", "processing"].includes(row.status)) {
        this.#terminal(job, row.setupId, "cancelled", row.timings, { code: "cancelled", message: "comparison cancelled" });
      }
    }
    job.done = true;
    job.detach();
    const result: ComparisonResult = Object.freeze({ comparisonId: job.id,
      status: cancelled ? "cancelled" : "complete", rows: Object.freeze([...job.rows.values()]),
      wallMs: this.#now() - job.admitted });
    this.#emit(job, { type: "done", comparisonId: job.id, status: result.status, wallMs: result.wallMs });
    job.resolve(result);
  }

  async #releaseExpired(): Promise<void> {
    for (const [setupId, used] of this.#resident) {
      if (this.#now() - used < this.#idleReleaseMs) continue;
      this.#resident.delete(setupId);
      // Even rejection must mean the retired generation has exited. A released
      // idle model has no row to fail and cannot poison later queue admission.
      try { await this.#executor.release(setupId); } catch { /* already stopped */ }
    }
  }

  #armIdle(): void {
    clearTimeout(this.#idleTimer);
    if (this.#closed || !this.#resident.size) return;
    const remaining = Math.min(...[...this.#resident.values()].map((used) => used + this.#idleReleaseMs - this.#now()));
    this.#idleTimer = setTimeout(() => this.#schedule(), Math.max(1, remaining));
    this.#idleTimer.unref();
  }
}
