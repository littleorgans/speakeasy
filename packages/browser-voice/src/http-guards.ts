import type { IncomingMessage, ServerResponse } from "node:http";
import { decodePcmSamples } from "./protocol.ts";
import { MAX_PCM_BYTES } from "./setup-protocol.ts";

const LOOPBACK_HOSTNAMES = new Set(["127.0.0.1", "localhost"]);
export const MAX_JSON_BYTES = 16_384;

export class HttpError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, message: string, code = "INVALID_REQUEST") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export function hostOrigin(request: IncomingMessage): string | undefined {
  return request.headers.host ? `http://${request.headers.host}` : undefined;
}

export function isLoopbackHost(request: IncomingMessage): boolean {
  try {
    const host = request.headers.host;
    return !!host && LOOPBACK_HOSTNAMES.has(new URL(`http://${host}`).hostname);
  } catch { return false; }
}

/** Reads with an Origin are protected too; non browser loopback GETs remain usable. */
export function requireLoopback(request: IncomingMessage, originRequired = false): void {
  const origin = request.headers.origin;
  if (!isLoopbackHost(request) || (originRequired || origin !== undefined) && origin !== hostOrigin(request)) {
    throw new HttpError(403, "same origin loopback pages only", "ORIGIN_FORBIDDEN");
  }
}

/** The request body, or null once it passes the bound. */
export async function readBody(request: IncomingMessage, limit: number): Promise<Buffer | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request as AsyncIterable<Buffer>) {
    size += chunk.length;
    if (size > limit) return null;
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

export async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  requireLoopback(request, true);
  if (request.headers["content-type"] !== "application/json") {
    throw new HttpError(415, "body must be application/json");
  }
  const body = await readBody(request, MAX_JSON_BYTES);
  if (body === null) throw new HttpError(413, "JSON body exceeds limit");
  try {
    const value: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
    if (!isRecord(value)) throw new Error();
    return value;
  } catch { throw new HttpError(400, "body must be a JSON object"); }
}

export async function readTakeSamples(request: IncomingMessage): Promise<Float32Array> {
  requireLoopback(request, true);
  if (request.headers["content-type"] !== "application/octet-stream") {
    throw new HttpError(415, "body must be application/octet-stream Float32 samples");
  }
  const body = await readBody(request, MAX_PCM_BYTES);
  if (body === null) throw new HttpError(413, `take exceeds ${MAX_PCM_BYTES} bytes`);
  try { return decodePcmSamples(body); }
  catch (error) { throw new HttpError(400, (error as Error).message); }
}

export function sendJson(response: ServerResponse, status: number, body: unknown): void {
  if (response.destroyed || response.writableEnded) return;
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...(status === 413 ? { connection: "close" } : {}),
  });
  response.end(JSON.stringify(body));
}

export function sendHttpError(response: ServerResponse, error: unknown): void {
  if (response.headersSent) { response.destroy(); return; }
  const known = error instanceof HttpError;
  sendJson(response, known ? error.status : 500, {
    code: known ? error.code : "INTERNAL_ERROR",
    error: known ? error.message : "Local speech request failed",
  });
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function exactFields(value: Record<string, unknown>, fields: readonly string[]): void {
  if (Object.keys(value).some((key) => !fields.includes(key))) {
    throw new HttpError(400, "unsupported request fields");
  }
}

export function requestId(value: unknown, fallback: () => string): string {
  if (value === undefined || value === null) return fallback();
  if (typeof value !== "string" || !/^[A-Za-z0-9_.:-]{1,128}$/.test(value)) {
    throw new HttpError(400, "requestId must be a bounded identifier");
  }
  return value;
}

/** Use response closure, since request close also fires after a normal upload. */
export function requestLifetime(request: IncomingMessage, response: ServerResponse) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  request.once("aborted", abort);
  response.once("close", abort);
  if (request.aborted || response.destroyed) abort();
  return {
    signal: controller.signal,
    abort,
    close() { abort(); response.destroy(); request.destroy(); },
    dispose() {
      request.off("aborted", abort);
      response.off("close", abort);
    },
  };
}
