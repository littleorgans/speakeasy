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
import {
  createConversationRuntime,
  type RuntimeConfig,
} from "@speakeasy/convo-engine";
import { WebSocket, WebSocketServer } from "ws";
import {
  PROMPT_MAX_TAKE_BYTES,
  PromptReader,
  UnknownPromptError,
  type PromptReaderConfig,
} from "./prompt-reader.ts";
import { BrowserConversationSession, type RuntimeFactory } from "./session.ts";

const PUBLIC_ROOT = fileURLToPath(new URL("../public/", import.meta.url));
const STATIC_FILES = new Map([
  ["/", "index.html"],
  ["/index.html", "index.html"],
  ["/styles.css", "styles.css"],
  ["/room-controls.css", "room-controls.css"],
  ["/app.js", "app.js"],
  ["/hold-release.js", "hold-release.js"],
  ["/capture-worklet.js", "capture-worklet.js"],
  ["/playback-worklet.js", "playback-worklet.js"],
  ["/capture-constraints.js", "capture-constraints.js"],
  ["/prompt-reader", "prompt-reader.html"],
  ["/prompt-reader.css", "prompt-reader.css"],
  ["/prompt-reader.js", "prompt-reader.js"],
]);
const PROMPT_READER_API = "/prompt-reader/";
const PROMPT_READER_TAKES = "/prompt-reader/takes/";
const LOOPBACK_HOSTNAMES = new Set(["127.0.0.1", "localhost"]);
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

export async function startBrowserVoiceServer(options: {
  port?: number;
  runtimeConfig?: RuntimeConfig;
  createRuntime?: RuntimeFactory;
  /** Serve the prompt reader's API, saving takes to an outside folder. */
  promptReader?: PromptReaderConfig;
} = {}): Promise<BrowserVoiceServer> {
  const promptReader = options.promptReader
    ? await PromptReader.load(options.promptReader)
    : undefined;
  const server = createServer((request, response) => {
    if (request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"ok":true}');
      return;
    }
    if (request.url?.startsWith(PROMPT_READER_API)) {
      void servePromptReader(request, response, promptReader);
      return;
    }
    void serveStatic(request.url ?? "/", response);
  });
  const sockets = new WebSocketServer({ noServer: true });
  let activeSocket: WebSocket | undefined;
  const runtimeConfig = options.runtimeConfig ?? {};
  const createRuntime = options.createRuntime ?? createConversationRuntime;

  server.on("upgrade", (request, socket, head) => {
    if (request.url !== "/voice") {
      socket.destroy();
      return;
    }
    const origin = request.headers.origin;
    if (origin !== undefined && origin !== hostOrigin(request)) {
      socket.write("HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    sockets.handleUpgrade(request, socket, head, (webSocket) => {
      sockets.emit("connection", webSocket, request);
    });
  });
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
    new BrowserConversationSession({ socket, runtimeConfig, createRuntime });
    socket.once("close", () => {
      if (activeSocket === socket) activeSocket = undefined;
    });
  });

  await listen(server, options.port ?? 4317);
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("browser voice server did not bind a TCP port");
  }
  const url = `http://127.0.0.1:${address.port}`;
  return {
    server,
    url,
    close: async () => {
      for (const socket of sockets.clients) socket.close();
      sockets.close();
      await new Promise<void>((resolveClose, rejectClose) => {
        server.close((error) => (error ? rejectClose(error) : resolveClose()));
      });
    },
  };
}

async function servePromptReader(
  request: IncomingMessage,
  response: ServerResponse,
  reader: PromptReader | undefined,
): Promise<void> {
  const url = new URL(request.url ?? "/", "http://localhost");
  try {
    if (!reader) {
      sendJson(response, 404, {
        error: "prompt reader not configured: start with --prompts <file> --out <dir>",
      });
    } else if (url.pathname === "/prompt-reader/prompts" && request.method === "GET") {
      sendJson(response, 200, await reader.listing());
    } else if (url.pathname === "/prompt-reader/recordings" && request.method === "POST") {
      await saveTake(request, response, reader, url.searchParams);
    } else if (url.pathname.startsWith(PROMPT_READER_TAKES) && request.method === "GET") {
      await serveTake(request, response, reader, url.pathname.slice(PROMPT_READER_TAKES.length));
    } else {
      sendJson(response, 404, { error: "Not found" });
    }
  } catch (error) {
    sendJson(response, 500, {
      error: `prompt reader failed: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
}

/**
 * Save route: the body is the take as 16 kHz mono Float32 samples. Writes
 * to disk, so only a same origin page on a loopback host may call it: a
 * browser always sends Origin on POST, and the Host check defeats DNS
 * rebinding.
 */
async function saveTake(
  request: IncomingMessage,
  response: ServerResponse,
  reader: PromptReader,
  query: URLSearchParams,
): Promise<void> {
  if (!isLoopbackHost(request) || request.headers.origin !== hostOrigin(request)) {
    sendJson(response, 403, { error: "same origin loopback pages only" });
    return;
  }
  if (request.headers["content-type"] !== "application/octet-stream") {
    sendJson(response, 415, { error: "body must be application/octet-stream Float32 samples" });
    return;
  }
  const body = await readBody(request, PROMPT_MAX_TAKE_BYTES);
  if (body === null) {
    sendJson(response, 413, { error: `take exceeds ${PROMPT_MAX_TAKE_BYTES} bytes` });
    return;
  }
  if (body.length === 0 || body.length % Float32Array.BYTES_PER_ELEMENT !== 0) {
    sendJson(response, 400, { error: "body must be a non-empty whole number of Float32 samples" });
    return;
  }
  try {
    // Copy into a fresh buffer: pooled Buffers need not be 4 byte aligned.
    const samples = new Float32Array(new Uint8Array(body).buffer);
    const saved = await reader.save(
      query.get("promptId") ?? "",
      samples,
      query.get("device") ?? "",
    );
    sendJson(response, 201, saved);
  } catch (error) {
    if (!(error instanceof UnknownPromptError)) throw error;
    sendJson(response, 404, { error: error.message });
  }
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
  if (!isLoopbackHost(request)) {
    sendJson(response, 403, { error: "loopback pages only" });
    return;
  }
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

/** The request body, or null once it passes `limit` bytes. */
async function readBody(request: IncomingMessage, limit: number): Promise<Buffer | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request as AsyncIterable<Buffer>) {
    size += chunk.length;
    if (size > limit) return null;
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function sendJson(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...(status === 413 ? { connection: "close" } : {}),
  });
  response.end(JSON.stringify(body));
}

function isLoopbackHost(request: IncomingMessage): boolean {
  const host = request.headers.host;
  return !!host && LOOPBACK_HOSTNAMES.has(new URL(`http://${host}`).hostname);
}

function hostOrigin(request: IncomingMessage): string | undefined {
  return request.headers.host ? `http://${request.headers.host}` : undefined;
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
