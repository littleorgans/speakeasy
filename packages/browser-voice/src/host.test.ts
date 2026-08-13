import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import type { ConversationRuntime, RuntimeConfig } from "@speakeasy/convo-engine";
import { WebSocket } from "ws";
import { startBrowserVoiceServer } from "./host.ts";
import { safeMessage } from "./session.ts";

test("host serves the room shell and reports health", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  try {
    const health = await fetch(`${host.url}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { ok: true });
    const page = await fetch(host.url);
    assert.equal(page.status, 200);
    assert.match(
      page.headers.get("content-security-policy") ?? "",
      /frame-ancestors 'none'/,
    );
    assert.match(await page.text(), /Speak Easy · Live room/);
  } finally {
    await host.close();
  }
});

test("browser errors redact provider credentials", () => {
  assert.equal(
    safeMessage(new Error("upstream rejected Bearer sk-proj-exampleSecret123")),
    "upstream rejected Bearer [redacted]",
  );
  assert.equal(
    safeMessage("provider csk-exampleSecret123 failed"),
    "provider [redacted] failed",
  );
});

test("host rejects browser sockets from another origin", async () => {
  const host = await startBrowserVoiceServer({ port: 0 });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice", {
    origin: "https://example.com",
  });
  try {
    const error = await new Promise<Error>((resolve) => socket.once("error", resolve));
    assert.match(error.message, /403/);
  } finally {
    socket.close();
    await host.close();
  }
});

test("browser socket drives one complete transcript and acknowledged playback", async () => {
  const captured: {
    runtime?: RuntimeConfig;
    endpoint?: { mode?: string; minTrailingSilenceMs?: number };
  } = {};
  const host = await startBrowserVoiceServer({
    port: 0,
    createRuntime: async (config) => {
      captured.runtime = config;
      return fakeRuntime((endpoint) => {
        captured.endpoint = endpoint;
      });
    },
  });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice");
  const events: Array<Record<string, unknown>> = [];
  const audio: Buffer[] = [];
  socket.on("message", (data, isBinary) => {
    if (isBinary) {
      audio.push(Buffer.from(data as Buffer));
      return;
    }
    const event = JSON.parse(data.toString()) as Record<string, unknown>;
    events.push(event);
    if (event.type === "playback" && event.action === "end") {
      socket.send(
        JSON.stringify({ type: "playback-drained", playbackId: event.playbackId }),
      );
    }
  });

  try {
    await onceOpen(socket);
    socket.send(JSON.stringify({
      type: "start",
      mode: "hold",
      pauseMs: 700,
      voice: "cedar",
      barge: true,
    }));
    await waitFor(() => events.some((event) => event.type === "session" && event.phase === "ready"));
    socket.send(new Float32Array(320).fill(0.2));
    await waitFor(() => events.some((event) => event.type === "transcript"));
    assert.ok(!events.some((event) => event.type === "metrics"));
    socket.send(JSON.stringify({ type: "commit-input" }));
    await waitFor(() => events.some((event) => event.type === "metrics"));

    const transcripts = events.filter((event) => event.type === "transcript");
    assert.ok(transcripts.some((event) => event.role === "user" && event.text === "hello browser"));
    assert.ok(transcripts.some((event) => event.role === "assistant" && event.text === "Hello back." && event.final === true));
    assert.equal(audio.length, 1);
    assert.ok(events.some((event) => event.type === "playback" && event.action === "start"));
    assert.ok(events.some((event) => event.type === "playback" && event.action === "end"));
    assert.equal(captured.runtime?.voice, "cedar");
    assert.deepEqual(captured.endpoint, { mode: "manual" });
  } finally {
    socket.close();
    await onceClose(socket);
    await host.close();
  }
});

test("natural mode maps the configured pause to eager endpointing", async () => {
  const captured: { endpoint?: unknown } = {};
  const host = await startBrowserVoiceServer({
    port: 0,
    createRuntime: async () => fakeRuntime((endpoint) => {
      captured.endpoint = endpoint;
    }),
  });
  const socket = new WebSocket(host.url.replace("http", "ws") + "/voice");
  try {
    await onceOpen(socket);
    socket.send(JSON.stringify({
      type: "start",
      mode: "natural",
      pauseMs: 950,
      voice: "marin",
      barge: true,
    }));
    await waitFor(() => captured.endpoint !== undefined);
    assert.deepEqual(captured.endpoint, {
      mode: "eager",
      minTrailingSilenceMs: 950,
    });
  } finally {
    socket.close();
    await onceClose(socket);
    await host.close();
  }
});

function fakeRuntime(
  onOpen: (endpoint: { mode?: string; minTrailingSilenceMs?: number }) => void = () => {},
): ConversationRuntime {
  const session = new FakeSttSession();
  return {
    label: "test speech · test voice",
    stt: {
      open: async (config) => {
        const endpoint = config?.endpoint ?? {};
        session.eager = endpoint.mode !== "manual";
        onOpen(endpoint);
        return session;
      },
    },
    responder: {
      open: async () => ({
        async *respond() {
          yield { type: "token" as const, text: "Hello back.", at: performance.now() };
          yield {
            type: "audio" as const,
            segment: {
              index: 0,
              sentence: "Hello back.",
              samples: new Float32Array(320).fill(0.1),
              sampleRate: 16_000,
              readyAtMs: 1,
              synthMs: 1,
              audioDurationMs: 20,
            },
          };
        },
        interrupt() {},
        async close() {},
      }),
    },
  };
}

class FakeSttSession extends EventEmitter {
  #spoken = false;
  eager = true;
  pushAudio(): void {
    if (this.#spoken) return;
    this.#spoken = true;
    this.emit("partial", { text: "hello" });
    if (this.eager) this.commit();
  }
  flush(): void {
    if (this.#spoken) this.commit();
  }
  commit(): void {
    this.emit("endpoint", {});
    this.emit("final", { text: "hello browser" });
  }
  reset(): void {}
  async end(): Promise<void> {}
}

function onceOpen(socket: WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.once("open", resolve);
    socket.once("error", reject);
  });
}

function onceClose(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.CLOSED) return Promise.resolve();
  return new Promise((resolve) => socket.once("close", resolve));
}

async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out waiting for browser event");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
