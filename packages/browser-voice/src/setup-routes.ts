import { createHash, randomUUID } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { decodeWavSamples } from "@speakeasy/speech-io";
import { ProviderError, safeProviderError, type SpeechCatalogue, type SpeechConfigRegistries,
  type SpeechInputConfig } from "@speakeasy/speech-io/providers";
import { AudioWorkLease } from "./audio-lease.ts";
import { exactFields, HttpError, isRecord, readJson, readTakeSamples, requireLoopback,
  requestId, requestLifetime, sendHttpError, sendJson } from "./http-guards.ts";
import type { PromptReader } from "./prompt-reader.ts";
import { ProviderPolicy, type ProviderDispatchContext, type ProviderSource, type VerifiedOwnerReplayGrant } from "./provider-policy.ts";
import type { ProviderCredentials } from "./provider-availability.ts";
import type { SpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { ComparisonBusyError, SetupComparison } from "./setup-comparison.ts";
import { createCanonicalPcm, MAX_PCM_BYTES, pcmIdentity, type SetupId } from "./setup-protocol.ts";
import { parseSetupSelection, SETUPS, SETUP_MANIFEST, SETUP_MANIFEST_SHA256 } from "./setup-registry.ts";
import { CartesiaBatchTranscriber, CARTESIA_STT_MODEL } from "../../speech-io/src/engines/cartesia-stt.ts";
import { OpenAIFileTranscriber, OPENAI_FILE_MODEL } from "../../speech-io/src/engines/openai-file.ts";

export type SetupReadiness = { setupId: SetupId; ready: boolean; reason?: string;
  deadlines?: { loadMs: number; operationMs: number } };
export type InputApiAdapter = {
  kind: "cartesia" | "openai";
  model: string;
  label: string;
  transcribe(samples: Float32Array, context: ProviderDispatchContext, signal: AbortSignal): Promise<unknown>;
};
export type ProviderRoutesConfig = {
  policy: ProviderPolicy;
  catalogue(): SpeechCatalogue;
  registries(): SpeechConfigRegistries;
  inputAdapters: readonly InputApiAdapter[];
};

/** Actual researched file adapters, constructed only after policy authorization. */
export function createFileInputAdapters(options: { fetch?: typeof globalThis.fetch } = {}): readonly InputApiAdapter[] {
  return [
    { kind: "cartesia", model: CARTESIA_STT_MODEL, label: "Cartesia Ink Whisper",
      transcribe: (samples, context, signal) => new CartesiaBatchTranscriber({ getKey: context.apiKey!, fetch: options.fetch })
        .transcribe(samples, { signal }) },
    { kind: "openai", model: OPENAI_FILE_MODEL, label: "OpenAI transcription",
      transcribe: (samples, context, signal) => new OpenAIFileTranscriber({ getKey: context.apiKey!, fetch: options.fetch })
        .transcribe(samples, { signal }) },
  ];
}
type SetupRoutesOptions = {
  reader?: PromptReader; comparison: SetupComparison; lease: AudioWorkLease;
  readiness(): readonly SetupReadiness[]; providers?: ProviderRoutesConfig;
};
type SavedSource = { kind: "saved"; takeName: string };
const SESSION_COOKIE = "speakeasy-session";
const MAX_SESSIONS = 128;
const MAX_SAVED_WAV_BYTES = MAX_PCM_BYTES + 65_536;

/** Host route lifecycle. No metadata method invokes an executor or speech adapter. */
export class SetupRoutes {
  readonly #options: SetupRoutesOptions;
  readonly #sessions = new Set<string>();
  readonly #lifetimes = new Set<ReturnType<typeof requestLifetime>>();
  readonly #pending = new Set<Promise<void>>();
  readonly #apiSessions = new Map<string, Set<ReturnType<typeof requestLifetime>>>();
  #closed = false;

  constructor(options: SetupRoutesOptions) { this.#options = options; }

  handles(path: string): boolean {
    return ["/prompt-reader/setups", "/prompt-reader/comparisons", "/prompt-reader/api-transcriptions",
      "/speech-providers/catalogue", "/speech-providers/session", "/speech-providers/selection"].includes(path);
  }

  serve(request: IncomingMessage, response: ServerResponse): void {
    const task = this.#serve(request, response).catch((error) => {
      if (error instanceof ProviderError) sendJson(response,
        error.code === "PROVIDER_UNAVAILABLE" || error.code === "PROVIDER_FAILED" ? 503 :
          error.code === "CONFIG_UNSUPPORTED" ? 400 : 403, { code: error.code, error: error.message });
      else sendHttpError(response, error);
    });
    this.#pending.add(task);
    void task.finally(() => this.#pending.delete(task));
  }

  /** Cookie identity is minted only by a protected session route. */
  sessionId(request: IncomingMessage): string | undefined {
    const cookie = request.headers.cookie?.split(";").map((part) => part.trim())
      .find((part) => part.startsWith(`${SESSION_COOKIE}=`))?.slice(SESSION_COOKIE.length + 1);
    return cookie && this.#sessions.has(cookie) ? cookie : undefined;
  }

  async close(): Promise<void> {
    this.#closed = true;
    for (const lifetime of this.#lifetimes) lifetime.close();
    await this.#options.comparison.close();
    await Promise.allSettled(this.#pending);
    for (const id of this.#sessions) this.#options.providers?.policy.revoke(id);
    this.#sessions.clear();
  }

  async #serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    if (this.#closed) throw new HttpError(503, "Speech server is closing");
    const url = new URL(request.url ?? "/", "http://localhost");
    requireLoopback(request, request.method !== "GET");
    if (url.pathname === "/prompt-reader/setups" && request.method === "GET") {
      sendJson(response, 200, this.metadata());
    } else if (url.pathname === "/prompt-reader/comparisons" && request.method === "POST") {
      await this.#compare(request, response, url);
    } else if (url.pathname === "/speech-providers/session" && request.method === "POST") {
      await this.#session(request, response);
    } else if (url.pathname === "/speech-providers/catalogue" && request.method === "GET") {
      sendJson(response, 200, this.#providers().catalogue());
    } else if (url.pathname === "/speech-providers/selection" && request.method === "POST") {
      const sessionId = this.#requireSession(request);
      const value = await readJson(request);
      exactFields(value, ["direction", "catalogueRevision", "config"]);
      if (value.direction !== "input" && value.direction !== "output" || typeof value.catalogueRevision !== "string") {
        throw new HttpError(400, "selection requires direction, catalogueRevision and config");
      }
      const selected = this.#providers().policy.select({ sessionId, direction: value.direction,
        catalogueRevision: value.catalogueRevision, config: value.config });
      if (value.direction === "input") for (const active of this.#apiSessions.get(sessionId) ?? []) active.close();
      sendJson(response, 200, selected);
    } else if (url.pathname === "/prompt-reader/api-transcriptions" && request.method === "POST") {
      await this.#apiTranscribe(request, response, url);
    } else throw new HttpError(404, "Not found");
  }

  metadata() {
    const readiness = this.#options.readiness();
    return { schema: 1, manifestSha256: SETUP_MANIFEST_SHA256,
      defaultSetupId: SETUP_MANIFEST.defaultSetupId,
      setups: SETUPS.map((setup) => ({ ...setup, maxAudioMs: setup.maxSamples / 16, residency: "unobserved",
        deadlines: readiness.find((entry) => entry.setupId === setup.id)?.deadlines ?? null,
        readiness: readiness.find((entry) => entry.setupId === setup.id) ??
          { setupId: setup.id, ready: false, reason: "runtime not configured" } })),
      limits: { maxPcmBytes: MAX_PCM_BYTES, active: 1, queued: 1, idleReleaseMs: SETUP_MANIFEST.idleReleaseMs,
        requestTimeoutMs: null },
      events: ["accepted", "started", "row", "done"] };
  }

  async #compare(request: IncomingMessage, response: ServerResponse, url: URL): Promise<void> {
    const reader = this.#reader();
    const lifetime = requestLifetime(request, response);
    this.#lifetimes.add(lifetime);
    let release: (() => void) | undefined;
    try {
      let id: string;
      let setupIds: readonly SetupId[];
      let samples: Float32Array;
      let source: ProviderSource;
      if (request.headers["content-type"] === "application/json") {
        boundedQuery(url, []);
        const value = await readJson(request);
        exactFields(value, ["requestId", "setupIds", "source"]);
        const saved = savedSource(value.source);
        id = requestId(value.requestId, randomUUID);
        setupIds = selection(value.setupIds);
        const path = await savedPath(reader, saved);
        source = { kind: "saved-take", corpusId: reader.corpusId, takeId: saved.takeName };
        samples = await readSavedSamples(path);
      } else {
        boundedQuery(url, ["requestId", "setupIds"]);
        id = requestId(url.searchParams.get("requestId"), randomUUID);
        setupIds = selection(queryJson(url, "setupIds"));
        samples = await readTakeSamples(request);
        source = { kind: "live-capture", sessionId: randomUUID(), operationId: randomUUID() };
      }
      if (lifetime.signal.aborted) return;
      release = this.#options.lease.tryAcquire("comparison");
      if (!release) throw new HttpError(409, "Other speech work is in progress", "AUDIO_BUSY");
      let sequence = 0;
      const handle = this.#options.comparison.compare({ samples, setupIds, signal: lifetime.signal,
        onEvent: (event) => {
          if (lifetime.signal.aborted) return;
          if (!response.headersSent) response.writeHead(200, {
            "content-type": "application/x-ndjson", "cache-control": "no-store",
            "x-content-type-options": "nosniff",
          });
          response.write(JSON.stringify({ ...event, requestId: id, sequence: sequence++,
            manifestSha256: SETUP_MANIFEST_SHA256, ...(event.type === "accepted" ? { source } : {}) }) + "\n");
        } });
      await handle.done;
      if (!lifetime.signal.aborted) response.end();
    } catch (error) {
      if (error instanceof ComparisonBusyError) throw new HttpError(429, error.message, "COMPARISON_BUSY");
      throw error;
    } finally {
      release?.();
      lifetime.dispose();
      this.#lifetimes.delete(lifetime);
    }
  }

  #providers(): ProviderRoutesConfig {
    const providers = this.#options.providers;
    if (!providers) throw new HttpError(503, "Speech providers are not configured");
    return providers;
  }

  #reader(): PromptReader {
    if (!this.#options.reader) throw new HttpError(404, "prompt reader not configured");
    return this.#options.reader;
  }

  #requireSession(request: IncomingMessage): string {
    const id = this.sessionId(request);
    if (!id) throw new HttpError(403, "Create a speech session before selecting a provider", "SESSION_REQUIRED");
    return id;
  }

  async #session(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const value = await readJson(request);
    exactFields(value, []);
    let id = this.sessionId(request);
    if (!id) {
      if (this.#sessions.size >= MAX_SESSIONS) throw new HttpError(429, "Too many speech sessions");
      id = randomUUID();
      this.#sessions.add(id);
    }
    response.setHeader("set-cookie", `${SESSION_COOKIE}=${id}; HttpOnly; SameSite=Strict; Path=/`);
    sendJson(response, 201, { sessionId: id });
  }

  async #apiTranscribe(request: IncomingMessage, response: ServerResponse, url: URL): Promise<void> {
    const providers = this.#providers();
    const reader = this.#reader();
    const sessionId = this.#requireSession(request);
    const lifetime = requestLifetime(request, response);
    this.#lifetimes.add(lifetime);
    const sessionWork = this.#apiSessions.get(sessionId) ?? new Set<ReturnType<typeof requestLifetime>>();
    sessionWork.add(lifetime);
    this.#apiSessions.set(sessionId, sessionWork);
    let release: (() => void) | undefined;
    try {
      const saved = request.headers["content-type"] === "application/json";
      if (saved) boundedQuery(url, []);
      const value = saved ? await readJson(request) : liveApiMetadata(url);
      exactFields(value, ["requestId", "config", "catalogueRevision", "selectionRevision", ...(saved ? ["source"] : [])]);
      const id = requestId(value.requestId, randomUUID);
      if (typeof value.catalogueRevision !== "string" || !Number.isSafeInteger(value.selectionRevision)) {
        throw new HttpError(400, "API transcription requires selection revisions");
      }
      let path: string | undefined;
      let source: ProviderSource;
      if (saved) {
        const reference = savedSource(value.source);
        path = await savedPath(reader, reference);
        source = { kind: "saved-take", corpusId: reader.corpusId, takeId: reference.takeName };
      } else source = { kind: "live-capture", sessionId, operationId: randomUUID() };
      const dispatch = { sessionId, direction: "input" as const, config: value.config,
        catalogueRevision: value.catalogueRevision, selectionRevision: Number(value.selectionRevision), source };
      // Saved denial is deliberately before reading the WAV or invoking an adapter.
      const authorization = providers.policy.authorize(dispatch);
      if (authorization.direction !== "input") throw new HttpError(400, "input direction required");
      const config: SpeechInputConfig = authorization.config;
      if (config.provider.kind === "local") throw new HttpError(400, "Use local comparisons for local input");
      const provider = config.provider;
      const adapter = providers.inputAdapters.find((entry) => entry.kind === provider.kind && entry.model === provider.model);
      if (!adapter) throw new ProviderError("PROVIDER_UNAVAILABLE");
      release = this.#options.lease.tryAcquire("api-input");
      if (!release) throw new HttpError(409, "Other speech work is in progress", "AUDIO_BUSY");
      const samples = path ? await readSavedSamples(path) : await readTakeSamples(request);
      if (lifetime.signal.aborted) return;
      const pcm = createCanonicalPcm(samples);
      const result = await providers.policy.dispatch(dispatch, async (context) => {
        context.recheck();
        const result = await adapter.transcribe(pcm.samples, context, lifetime.signal);
        context.recheck();
        return result;
      });
      if (!lifetime.signal.aborted) sendJson(response, 200, { requestId: id, config, source, pcm: pcmIdentity(pcm), result });
    } catch (error) {
      if (error instanceof HttpError) throw error;
      throw safeProviderError(error);
    } finally {
      release?.();
      lifetime.dispose();
      this.#lifetimes.delete(lifetime);
      sessionWork.delete(lifetime);
      if (sessionWork.size === 0) this.#apiSessions.delete(sessionId);
    }
  }
}

function savedSource(value: unknown): SavedSource {
  if (!isRecord(value) || value.kind !== "saved" || typeof value.takeName !== "string") {
    throw new HttpError(400, "saved source requires kind and takeName");
  }
  exactFields(value, ["kind", "takeName"]);
  return { kind: "saved", takeName: value.takeName };
}

async function savedPath(reader: PromptReader, source: SavedSource): Promise<string> {
  const path = await reader.takePath(source.takeName);
  if (!path) throw new HttpError(404, "no such take");
  return path;
}

async function readSavedSamples(path: string): Promise<Float32Array> {
  if ((await stat(path)).size > MAX_SAVED_WAV_BYTES) throw new HttpError(413, "saved WAV exceeds limit");
  try { return decodeWavSamples(await readFile(path)); }
  catch { throw new HttpError(400, "Saved WAV must be finite mono 16 kHz audio"); }
}

function selection(value: unknown): readonly SetupId[] {
  try { return parseSetupSelection(value); }
  catch { throw new HttpError(400, "select unique local setup ids"); }
}

function boundedQuery(url: URL, fields: readonly string[]): void {
  if (url.search.length > 4096 || [...url.searchParams.keys()].some((key) =>
    !fields.includes(key) || url.searchParams.getAll(key).length !== 1)) {
    throw new HttpError(400, "invalid request metadata");
  }
}

function queryJson(url: URL, key: string): unknown {
  const value = url.searchParams.get(key);
  if (value === null) return undefined;
  try { return JSON.parse(value); }
  catch { throw new HttpError(400, "invalid JSON request metadata"); }
}

function liveApiMetadata(url: URL): Record<string, unknown> {
  boundedQuery(url, ["requestId", "config", "catalogueRevision", "selectionRevision"]);
  return { requestId: url.searchParams.get("requestId"), config: queryJson(url, "config"),
    catalogueRevision: url.searchParams.get("catalogueRevision"), selectionRevision: queryJson(url, "selectionRevision") };
}

/** Derive one catalogue/policy from the actual registries and server environment. */
export function createHostProviderConfig(options: {
  credentials: ProviderCredentials;
  readiness(): readonly SetupReadiness[];
  inputAdapters: readonly InputApiAdapter[];
  outputCatalogue?: () => SpeechOutputCatalogue;
  serverRunId?: string;
  replayGrants?: readonly VerifiedOwnerReplayGrant[];
}): ProviderRoutesConfig {
  const catalogue = (): SpeechCatalogue => {
    const readiness = options.readiness();
    const input: SpeechCatalogue["input"] = SETUPS.map((setup) => ({ id: setup.id, label: setup.label,
      config: { schema: 1, provider: { kind: "local", presetId: setup.id } },
      selectable: readiness.find((entry) => entry.setupId === setup.id)?.ready === true,
      transfer: "stays-on-mac", samples: [] }));
    for (const adapter of options.inputAdapters) {
      if (!options.credentials.configured(adapter.kind)) continue;
      input.push({ id: `${adapter.kind}/${adapter.model}`, label: adapter.label,
        config: { schema: 1, provider: { kind: adapter.kind, model: adapter.model } },
        selectable: true, transfer: "audio-to-api", samples: [] });
    }
    const output = options.outputCatalogue?.().output ?? [];
    const revision = createHash("sha256").update(JSON.stringify({ input, output,
      outputRevision: options.outputCatalogue?.().revision })).digest("hex");
    return { schema: 1, revision, input, output };
  };
  const registries = (): SpeechConfigRegistries => ({ input: { localPresets: SETUPS,
    apiModels: options.inputAdapters.map(({ kind, model }) => ({ kind, model })) },
    output: options.outputCatalogue?.().registry ?? [] });
  const policy = new ProviderPolicy({ serverRunId: options.serverRunId ?? randomUUID(),
    catalogue, registries, credentials: options.credentials, replayGrants: options.replayGrants });
  return { catalogue, registries, policy, inputAdapters: options.inputAdapters };
}
