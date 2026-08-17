import { Buffer } from "node:buffer";
import {
  ConversationLoop,
  type ConversationRuntime,
  type RuntimeConfig,
} from "@speakeasy/convo-engine";
import { WebSocket, type RawData } from "ws";
import { BrowserAudioSink, BrowserAudioSource } from "./browser-audio.ts";
import {
  decodeMicFrame,
  parseClientCommand,
  type BrowserHostEvent,
  type ClientCommand,
} from "./protocol.ts";

export type RuntimeFactory = (
  config: RuntimeConfig,
) => Promise<ConversationRuntime>;

/** One browser socket owns one conversation loop and its audio adapters. */
export class BrowserConversationSession {
  readonly #socket: WebSocket;
  readonly #runtimeConfig: RuntimeConfig;
  readonly #createRuntime: RuntimeFactory;
  readonly #mic = new BrowserAudioSource();
  #loop: ConversationLoop | undefined;
  #activeSink: BrowserAudioSink | undefined;
  #nextPlaybackId = 1;
  #starting = false;
  #closed = false;

  constructor(options: {
    socket: WebSocket;
    runtimeConfig: RuntimeConfig;
    createRuntime: RuntimeFactory;
  }) {
    this.#socket = options.socket;
    this.#runtimeConfig = options.runtimeConfig;
    this.#createRuntime = options.createRuntime;
    this.#wireSocket();
  }

  async stop(): Promise<void> {
    this.#starting = false;
    await this.#loop?.stop();
    this.#loop = undefined;
    this.#activeSink = undefined;
    if (!this.#closed) {
      this.#send({ type: "session", phase: "stopped" });
    }
  }

  #wireSocket(): void {
    this.#socket.on("message", (data, isBinary) => {
      if (isBinary) {
        this.#acceptAudio(data);
        return;
      }
      try {
        const command = parseClientCommand(rawText(data));
        void this.#handle(command);
      } catch (error) {
        this.#sendNotice("error", safeMessage(error));
      }
    });
    this.#socket.once("close", () => {
      this.#closed = true;
      void this.stop();
    });
    this.#socket.on("error", () => {
      this.#closed = true;
      void this.stop();
    });
  }

  async #handle(command: ClientCommand): Promise<void> {
    switch (command.type) {
      case "start":
        await this.#start(command);
        break;
      case "stop":
        await this.stop();
        break;
      case "interrupt":
        this.#loop?.interrupt();
        break;
      case "commit-input":
        this.#loop?.commitInput();
        break;
      case "playback-drained":
        if (this.#activeSink?.playbackId === command.playbackId) {
          this.#activeSink.drained(command.audioEndMs);
        }
        break;
    }
  }

  async #start(command: Extract<ClientCommand, { type: "start" }>): Promise<void> {
    if (this.#starting || this.#loop) {
      return;
    }
    this.#starting = true;
    this.#send({ type: "session", phase: "loading" });
    try {
      const runtimeConfig =
        runtimeConfigFor(command, this.#runtimeConfig);
      const runtime = await this.#createRuntime(runtimeConfig);
      if (!this.#starting || this.#closed) {
        return;
      }
      const loop = new ConversationLoop(
        {
          stt: runtime.stt,
          responder: runtime.responder,
          mic: this.#mic,
          createSink: (sampleRate) => this.#createSink(sampleRate),
        },
        {
          systemPrompt: command.systemPrompt,
          barge: command.mode === "natural" && command.barge,
          sttConfig: {
            sampleRate: 16_000,
            endpoint:
              command.mode === "hold"
                ? { mode: "manual" }
                : { mode: "eager", minTrailingSilenceMs: command.pauseMs },
          },
          onEvent: (event) => this.#send(event),
        },
      );
      this.#loop = loop;
      await loop.start();
      this.#send({ type: "session", phase: "ready", label: runtime.label });
    } catch (error) {
      this.#loop = undefined;
      this.#sendNotice("error", safeMessage(error));
      this.#send({ type: "session", phase: "stopped" });
    } finally {
      this.#starting = false;
    }
  }

  #createSink(sampleRate: number): BrowserAudioSink {
    const sink = new BrowserAudioSink({
      playbackId: this.#nextPlaybackId++,
      sampleRate,
      sendEvent: (event) => this.#send(event),
      sendAudio: (packet) => {
        if (this.#socket.readyState === WebSocket.OPEN) {
          this.#socket.send(packet);
        }
      },
    });
    this.#activeSink = sink;
    return sink;
  }

  #acceptAudio(data: RawData): void {
    try {
      this.#mic.accept(decodeMicFrame(rawBytes(data)));
    } catch (error) {
      const message = safeMessage(error);
      this.#mic.fail(new Error(message));
      this.#sendNotice("error", message);
    }
  }

  #sendNotice(level: "info" | "error", message: string): void {
    this.#send({ type: "notice", level, message });
  }

  #send(event: BrowserHostEvent): void {
    if (this.#socket.readyState === WebSocket.OPEN) {
      this.#socket.send(JSON.stringify(event));
    }
  }
}

export function runtimeConfigFor(
  command: Extract<ClientCommand, { type: "start" }>,
  defaults: RuntimeConfig,
): RuntimeConfig {
  const {
    responder: _responder,
    llmProvider: _llmProvider,
    llmModel: _llmModel,
    llmReasoningEffort: _llmReasoningEffort,
    voice,
    ...shared
  } = defaults;
  if (command.engine === "realtime") {
    return { ...shared, responder: "realtime", voice: command.voice };
  }
  return {
    ...shared,
    responder: "cascade",
    llmProvider: "mercury",
    llmReasoningEffort:
      command.engine === "mercury-instant" ? "instant" : "low",
    ...(voice === undefined ? {} : { voice }),
  };
}

function rawText(data: RawData): string {
  return Buffer.isBuffer(data) ? data.toString("utf8") : Buffer.from(rawBytes(data)).toString("utf8");
}

function rawBytes(data: RawData): Uint8Array {
  if (Array.isArray(data)) {
    return Buffer.concat(data);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function safeMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message
    .replace(/\b(?:sk|csk|sk-proj)-[A-Za-z0-9_-]{8,}\b/g, "[redacted]")
    .replace(/\bBearer\s+\S+/gi, "Bearer [redacted]");
}
