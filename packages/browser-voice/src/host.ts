import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { createConversationRuntime } from "@speakeasy/convo-engine";
import { WebSocket, WebSocketServer } from "ws";
import { HOTWORDS_FILENAME } from "@speakeasy/speech-io";
import {
  PROMPT_SAMPLE_RATE,
  PromptReader,
  UnknownPromptError,
  type PromptReaderConfig,
} from "./prompt-reader.ts";
import { BrowserConversationSession, type BrowserRuntimeConfig, type RuntimeFactory } from "./session.ts";
import { AudioWorkLease } from "./audio-lease.ts";
import { HttpError, readTakeSamples, requireLoopback, sendHttpError, sendJson } from "./http-guards.ts";
import { SetupComparison } from "./setup-comparison.ts";
import { SetupWorkers } from "./setup-workers.ts";
import type { SetupExecutor } from "./setup-protocol.ts";
import { SetupRoutes, type ProviderRoutesConfig, type SetupReadiness } from "./setup-routes.ts";
import { SpeechOutputSession, type SpeechOutputSessionOptions } from "./speech-output-session.ts";
import { ChildTranscriber, TRANSCRIBE_MODEL_ID, TranscriptionError, type Transcriber } from "./transcriber.ts";

const PUBLIC_ROOT = fileURLToPath(new URL("../public/", import.meta.url));
const STATIC_FILES = new Map([
  ["/", "index.html"],
  ["/index.html", "index.html"],
  ["/styles.css", "styles.css"],
  ["/room-controls.css", "room-controls.css"],
  ["/app.js", "app.js"],
  ["/hold-release.js", "hold-release.js"],
  ["/pre-roll.js", "pre-roll.js"],
  ["/capture-worklet.js", "capture-worklet.js"],
  ["/playback-worklet.js", "playback-worklet.js"],
  ["/capture-constraints.js", "capture-constraints.js"],
  ["/prompt-reader", "prompt-reader.html"],
  ["/prompt-reader.css", "prompt-reader.css"],
  ["/prompt-reader.js", "prompt-reader.js"],
  ["/setup-results.js", "setup-results.js"],
  ["/speech-output-controls.js", "speech-output-controls.js"],
  ["/speech-observations.js", "speech-observations.js"],
]);
const PROMPT_READER_API = "/prompt-reader/";
const PROMPT_READER_TAKES = "/prompt-reader/takes/";
const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

export type BrowserVoiceServer = {
  server: Server;
  url: string;
  close(): Promise<void>;
};

export type BrowserVoiceServerOptions = {
  port?: number;
  runtimeConfig?: BrowserRuntimeConfig;
  createRuntime?: RuntimeFactory;
  /** Serve the prompt reader's API, saving takes to an outside folder. */
  promptReader?: PromptReaderConfig;
  /** Per take hotword list; defaults to hotwords.txt in the working directory. */
  hotwordsPath?: string;
  /** Display only take transcription; defaults to the Parakeet Ultra child. */
  transcriber?: Transcriber;
  setupExecutor?: SetupExecutor;
  setupReadiness?: () => readonly SetupReadiness[];
  providers?: ProviderRoutesConfig;
  speechOutput?: Pick<SpeechOutputSessionOptions, "catalogue" | "createOutput" | "quiesce"> & { close?(): Promise<void> };
};

export async function startBrowserVoiceServer(options: BrowserVoiceServerOptions = {}): Promise<BrowserVoiceServer> {
  const promptReader = options.promptReader
    ? await PromptReader.load(options.promptReader)
    : undefined;
  const hotwordsPath = resolve(options.hotwordsPath ?? HOTWORDS_FILENAME);
  const transcriber = promptReader
    ? (options.transcriber ?? new ChildTranscriber({ hotwordsPath }))
    : undefined;
  const lease = new AudioWorkLease();
  const comparison = new SetupComparison({ executor: options.setupExecutor ?? new SetupWorkers() });
  const setupRoutes = new SetupRoutes({ reader: promptReader, comparison, lease,
    providers: options.providers, readiness: options.setupReadiness ?? (() => []) });
  const sessions = new Set<BrowserConversationSession>();
  const server = createServer((request, response) => {
    const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
    if (pathname === "/speech-output/models") {
      try {
        requireLoopback(request);
        if (request.method !== "GET" || !options.speechOutput) sendJson(response, 404, { error: "Speech output not configured" });
        else sendJson(response, 200, currentOutputCatalogue(options.speechOutput, options.providers));
      } catch (error) { sendHttpError(response, error); }
      return;
    }
    if (setupRoutes.handles(pathname)) {
      setupRoutes.serve(request, response);
      return;
    }
    if (request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"ok":true}');
      return;
    }
    if (request.url?.startsWith(PROMPT_READER_API)) {
      void servePromptReader(request, response, promptReader, transcriber, hotwordsPath, lease);
      return;
    }
    void serveStatic(request.url ?? "/", response);
  });
  const sockets = new WebSocketServer({ noServer: true });
  const outputSockets = new WebSocketServer({ noServer: true });
  const outputSessions = new Set<SpeechOutputSession>();
  let activeSocket: WebSocket | undefined;
  const runtimeConfig = options.runtimeConfig ?? {};
  const createRuntime: RuntimeFactory = options.createRuntime ?? ((config, context) =>
    createConversationRuntime(config, undefined, undefined, context));

  server.on("upgrade", (request, socket, head) => {
    const output = request.url === "/speech-output" && !!options.speechOutput && !!options.providers;
    if (request.url !== "/voice" && !output) {
      socket.destroy();
      return;
    }
    try { requireLoopback(request); } catch {
      socket.write("HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    const target = output ? outputSockets : sockets;
    target.handleUpgrade(request, socket, head, (webSocket) => {
      target.emit("connection", webSocket, request);
    });
  });
  outputSockets.on("connection", (socket) => connectSpeechOutput(socket, options, lease, outputSessions));
  sockets.on("connection", (socket) => {
    if (activeSocket?.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "notice",
          level: "error",
          message: "Another local voice session is already open",
        }),
      );
      socket.close(1013, "voice session busy");
      return;
    }
    activeSocket = socket;
    const session = new BrowserConversationSession({ socket, sessionId: randomUUID(), runtimeConfig, createRuntime,
      acquire: () => lease.tryAcquire("room"), policy: options.providers?.policy,
      outputCatalogue: options.speechOutput
        ? () => currentOutputCatalogue(options.speechOutput!, options.providers) : undefined,
      outputRegistry: () => options.providers?.registries().output ?? [] });
    sessions.add(session);
    socket.once("close", () => {
      if (activeSocket === socket) activeSocket = undefined;
      void session.close().then(() => sessions.delete(session), () => {});
    });
  });

  await listen(server, options.port ?? 4317);
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("browser voice server did not bind a TCP port");
  }
  const url = `http://127.0.0.1:${address.port}`;
  let closing: Promise<void> | undefined;
  return {
    server,
    url,
    close: () => closing ??= (async () => {
      lease.close();
      const transcriberClosed = transcriber instanceof ChildTranscriber
        ? transcriber.closeAndWait() : Promise.resolve(transcriber?.close());
      for (const socket of sockets.clients) socket.terminate();
      for (const socket of outputSockets.clients) socket.terminate();
      const stopped = Promise.allSettled([transcriberClosed, setupRoutes.close(), ...[...sessions].map((session) => session.close()),
        ...[...outputSessions].map((session) => session.close())]);
      sockets.close();
      outputSockets.close();
      server.closeIdleConnections();
      await new Promise<void>((resolveClose, rejectClose) => {
        server.close((error) => (error ? rejectClose(error) : resolveClose()));
      });
      const results = await stopped;
      await options.speechOutput?.close?.();
      const failed = results.find((result) => result.status === "rejected");
      if (failed?.status === "rejected") throw failed.reason;
    })(),
  };
}

function connectSpeechOutput(
  socket: WebSocket,
  options: BrowserVoiceServerOptions,
  lease: AudioWorkLease,
  outputSessions: Set<SpeechOutputSession>,
): void {
  const session = new SpeechOutputSession({ sessionId: randomUUID(),
    ...options.speechOutput!, catalogue: () => currentOutputCatalogue(options.speechOutput!, options.providers),
    policy: options.providers!.policy,
    acquire: () => lease.tryAcquire("tryout"),
    sendEvent: (event) => { if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(event)); },
    sendAudio: (packet) => { if (socket.readyState === WebSocket.OPEN) socket.send(packet); } });
  outputSessions.add(session);
  socket.on("message", (data, binary) => {
    if (!binary) session.handle(data.toString());
  });
  socket.on("error", () => { void session.close().catch(() => {}); });
  socket.once("close", () => { void session.close().then(() => outputSessions.delete(session), () => {}); });
}

/** Output metadata and dispatch bind to the same complete catalogue identity. */
function currentOutputCatalogue(
  output: NonNullable<BrowserVoiceServerOptions["speechOutput"]>,
  providers: ProviderRoutesConfig | undefined,
) {
  const snapshot = output.catalogue();
  return { ...snapshot, revision: providers?.catalogue().revision ?? snapshot.revision };
}

async function servePromptReader(
  request: IncomingMessage,
  response: ServerResponse,
  reader: PromptReader | undefined,
  transcriber: Transcriber | undefined,
  hotwordsPath: string,
  lease: AudioWorkLease,
): Promise<void> {
  const url = new URL(request.url ?? "/", "http://localhost");
  try {
    requireLoopback(request, request.method !== "GET");
    if (!reader) {
      sendJson(response, 404, {
        error: "prompt reader not configured: start with --prompts <file> --out <dir>",
      });
    } else if (url.pathname === "/prompt-reader/prompts" && request.method === "GET") {
      sendJson(response, 200, { ...(await reader.listing()),
        transcription: { model: TRANSCRIBE_MODEL_ID, hotwordsPath } });
    } else if (url.pathname === "/prompt-reader/recordings" && request.method === "POST") {
      await saveTake(request, response, reader, url.searchParams);
    } else if (url.pathname === "/prompt-reader/transcriptions" && request.method === "POST" && transcriber) {
      await transcribeTake(request, response, transcriber, lease);
    } else if (url.pathname.startsWith(PROMPT_READER_TAKES) && request.method === "GET") {
      await serveTake(request, response, reader, url.pathname.slice(PROMPT_READER_TAKES.length));
    } else {
      sendJson(response, 404, { error: "Not found" });
    }
  } catch (error) {
    sendHttpError(response, error);
  }
}

/** Save route: writes the posted take to disk as a corpus pair. */
async function saveTake(
  request: IncomingMessage,
  response: ServerResponse,
  reader: PromptReader,
  query: URLSearchParams,
): Promise<void> {
  const samples = await readTakeSamples(request);
  const preRoll = query.get("preRollMs");
  const preRollMs = preRoll === null ? undefined : Number(preRoll);
  const takeMs = (samples.length / PROMPT_SAMPLE_RATE) * 1_000;
  if (preRoll !== null && (!/^\d+$/.test(preRoll) || preRollMs! > takeMs)) {
    sendJson(response, 400, { error: "preRollMs must be whole milliseconds within the take" });
    return;
  }
  try {
    const saved = await reader.save(
      query.get("promptId") ?? "",
      samples,
      query.get("device") ?? "",
      preRollMs,
    );
    sendJson(response, 201, saved);
  } catch (error) {
    if (!(error instanceof UnknownPromptError)) throw error;
    sendJson(response, 404, { error: error.message });
  }
}

/**
 * Transcription route: display only text for the posted take. Nothing is
 * written; a failed or slow transcription answers 503 and the host serves on.
 */
async function transcribeTake(
  request: IncomingMessage,
  response: ServerResponse,
  transcriber: Transcriber,
  lease: AudioWorkLease,
): Promise<void> {
  const samples = await readTakeSamples(request);
  const release = lease.tryAcquire("legacy-input");
  if (!release) throw new HttpError(409, "Other speech work is in progress", "AUDIO_BUSY");
  try {
    sendJson(response, 200, await transcriber.transcribe(samples));
  } catch (error) {
    sendJson(response, 503, {
      error: `transcription unavailable: ${error instanceof Error ? error.message : String(error)}`,
      model: error instanceof TranscriptionError ? error.model : TRANSCRIBE_MODEL_ID,
      ...(error instanceof TranscriptionError && error.hotwords ? { hotwords: error.hotwords } : {}),
    });
  } finally { release(); }
}

/**
 * Read only playback of a saved take. The name must equal the audio file a
 * reader sidecar names, so no other path in or outside the folder is served.
 * The Host check keeps a rebound DNS name from reading the recordings.
 */
async function serveTake(
  request: IncomingMessage,
  response: ServerResponse,
  reader: PromptReader,
  encodedName: string,
): Promise<void> {
  requireLoopback(request);
  let name: string;
  try {
    name = decodeURIComponent(encodedName);
  } catch {
    name = "";
  }
  const path = name ? await reader.takePath(name) : undefined;
  if (!path) {
    sendJson(response, 404, { error: "no such take" });
    return;
  }
  const info = await stat(path);
  response.writeHead(200, {
    "content-type": "audio/wav",
    "content-length": info.size,
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  createReadStream(path).pipe(response);
}

async function serveStatic(
  requestUrl: string,
  response: ServerResponse,
): Promise<void> {
  const pathname = new URL(requestUrl, "http://localhost").pathname;
  const relative = STATIC_FILES.get(pathname);
  if (!relative) {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
    return;
  }
  const path = resolve(PUBLIC_ROOT, relative);
  try {
    const info = await stat(path);
    response.writeHead(200, {
      "content-type": CONTENT_TYPES[extname(path)] ?? "application/octet-stream",
      "content-length": info.size,
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
      "content-security-policy": [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self' ws:",
        "media-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
      ].join("; "),
    });
    createReadStream(path).pipe(response);
  } catch {
    response.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    response.end("Static asset unavailable");
  }
}

function listen(server: Server, port: number): Promise<void> {
  return new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });
}
