import { fork, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Display only transcription for the prompt reader: a take's samples go to a
 * child process that holds the recognizer, so a native sherpa failure kills
 * the child and never the host. The child loads its model on the first
 * request and stays warm while list content is unchanged. A child that exits
 * fails its pending requests and is forked again on the next one.
 */

export type TakeHotwords = { path: string; count: number; revision: string };
export type Transcript = { text: string; decodeMs: number; model: string; hotwords?: TakeHotwords };

export class TranscriptionError extends Error {
  readonly model: string;
  readonly hotwords: TakeHotwords | undefined;

  constructor(message: string, model: string, hotwords?: TakeHotwords) {
    super(message);
    this.model = model;
    this.hotwords = hotwords;
  }
}

export type Transcriber = {
  transcribe(samples: Float32Array): Promise<Transcript>;
  close(): void;
};

/** Worker protocol: one request per take, answered by id. */
export type TranscribeRequest = { id: number; samples: Float32Array };
export type TranscribeReply =
  | { id: number; text: string; decodeMs: number; hotwords?: TakeHotwords }
  | { id: number; error: string; hotwords?: TakeHotwords };

export const TRANSCRIBE_MODEL_ID = "en-parakeet-ultra-int8";
const WORKER_PATH = fileURLToPath(new URL("./transcribe-worker.ts", import.meta.url));
/**
 * Covers the first request's model load. A slower take is reported failed
 * and its child killed, so a stuck decode cannot hold later takes.
 */
const TRANSCRIBE_TIMEOUT_MS = 30_000;

type Pending = {
  resolve(transcript: Transcript): void;
  reject(error: Error): void;
  timer: NodeJS.Timeout;
};

export class ChildTranscriber implements Transcriber {
  readonly #workerPath: string;
  readonly #label: string;
  readonly #timeoutMs: number;
  readonly #hotwordsPath: string | undefined;
  readonly #pending = new Map<number, Pending>();
  #child: ChildProcess | undefined;
  #nextId = 0;

  constructor(options: { workerPath?: string; label?: string; timeoutMs?: number; hotwordsPath?: string } = {}) {
    this.#workerPath = options.workerPath ?? WORKER_PATH;
    this.#label = options.label ?? TRANSCRIBE_MODEL_ID;
    this.#timeoutMs = options.timeoutMs ?? TRANSCRIBE_TIMEOUT_MS;
    this.#hotwordsPath = options.hotwordsPath === undefined ? undefined : resolve(options.hotwordsPath);
  }

  transcribe(samples: Float32Array): Promise<Transcript> {
    const child = this.#spawn();
    const id = ++this.#nextId;
    return new Promise<Transcript>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#settle(id, new Error(`transcription timed out after ${this.#timeoutMs} ms`));
        child.kill();
      }, this.#timeoutMs);
      this.#pending.set(id, { resolve, reject, timer });
      child.send({ id, samples } satisfies TranscribeRequest, (error) => {
        if (error) this.#settle(id, error);
      });
    });
  }

  close(): void {
    const child = this.#child;
    this.#child = undefined;
    child?.kill();
    this.#failAll(new Error("transcriber closed"));
  }

  #spawn(): ChildProcess {
    if (this.#child) return this.#child;
    // Advanced serialization carries the Float32Array without a JSON copy.
    const child = fork(this.#workerPath, this.#hotwordsPath === undefined ? [] : [this.#hotwordsPath], {
      serialization: "advanced",
      stdio: ["ignore", "inherit", "inherit", "ipc"],
    });
    child.on("message", (reply: TranscribeReply) => {
      this.#settle(reply.id, "error" in reply
        ? new TranscriptionError(reply.error, this.#label, reply.hotwords)
        : reply);
    });
    child.on("error", (error) => this.#exited(child, error));
    child.on("exit", (code, signal) => {
      this.#exited(child, new Error(`transcriber exited (${signal ?? code})`));
    });
    this.#child = child;
    return child;
  }

  #exited(child: ChildProcess, error: Error): void {
    if (this.#child !== child) return;
    this.#child = undefined;
    this.#failAll(error);
  }

  #failAll(error: Error): void {
    for (const id of [...this.#pending.keys()]) this.#settle(id, error);
  }

  #settle(id: number, outcome: Error | { text: string; decodeMs: number; hotwords?: TakeHotwords }): void {
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    clearTimeout(pending.timer);
    if (outcome instanceof Error) pending.reject(outcome instanceof TranscriptionError
      ? outcome : new TranscriptionError(outcome.message, this.#label));
    else pending.resolve({ text: outcome.text, decodeMs: outcome.decodeMs, model: this.#label,
      ...(outcome.hotwords ? { hotwords: outcome.hotwords } : {}) });
  }
}
