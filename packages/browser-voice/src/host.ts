import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createConversationRuntime,
  type RuntimeConfig,
} from "@speakeasy/convo-engine";
import { WebSocket, WebSocketServer } from "ws";
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
]);
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
} = {}): Promise<BrowserVoiceServer> {
  const server = createServer((request, response) => {
    if (request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"ok":true}');
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
    const expectedOrigin = request.headers.host
      ? `http://${request.headers.host}`
      : undefined;
    if (origin !== undefined && origin !== expectedOrigin) {
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

async function serveStatic(
  requestUrl: string,
  response: import("node:http").ServerResponse,
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
