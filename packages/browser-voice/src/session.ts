import { Buffer } from "node:buffer";
import {
  ConversationLoop,
  type ConversationRuntime,
  type RuntimeConfig,
} from "@speakeasy/convo-engine";
import { WebSocket, type RawData } from "ws";
import { BrowserAudioSink, BrowserAudioSource } from "./browser-audio.ts";
import type { SpeechOutputRegistry } from "@speakeasy/speech-io/providers";
import type { ReleaseAudioWork } from "./audio-lease.ts";
import {
  decodeMicFrame,
  parseClientCommand,
  type BrowserHostEvent,
  type ClientCommand,
} from "./protocol.ts";

export type BrowserRuntimeConfig = RuntimeConfig;
export type BrowserConversationRuntime = ConversationRuntime;
export type RuntimeFactory = (
  config: BrowserRuntimeConfig,
) => Promise<BrowserConversationRuntime>;

/** One browser socket owns one conversation loop and its audio adapters. */
export class BrowserConversationSession {
  readonly #socket: WebSocket;
  readonly #runtimeConfig: BrowserRuntimeConfig;
  readonly #createRuntime: RuntimeFactory;
  readonly #acquire: (() => ReleaseAudioWork | undefined) | undefined;
  readonly #outputRegistry: () => SpeechOutputRegistry;
  readonly #mic = new BrowserAudioSource();
  #loop: ConversationLoop | undefined;
  #runtime: BrowserConversationRuntime | undefined;
  #activeSink: BrowserAudioSink | undefined;
  #nextPlaybackId = 1;
  #starting = false;
  #closed = false;
  #startTask: Promise<void> | undefined;
  #stopTask: Promise<void> | undefined;
  #release: ReleaseAudioWork | undefined;
  #retirementFailed = false;

  constructor(options: {
    socket: WebSocket;
    runtimeConfig: BrowserRuntimeConfig;
    createRuntime: RuntimeFactory;
    acquire?: () => ReleaseAudioWork | undefined;
    outputRegistry?: () => SpeechOutputRegistry;
  }) {
    this.#socket = options.socket;
    this.#runtimeConfig = options.runtimeConfig;
    this.#createRuntime = options.createRuntime;
    this.#acquire = options.acquire;
    this.#outputRegistry = options.outputRegistry ?? (() => []);
    this.#wireSocket();
  }

  stop(): Promise<void> {
    if (this.#stopTask) return this.#stopTask;
    const task = this.#stop();
    this.#stopTask = task;
    void task.then(() => { this.#stopTask = undefined; }, () => { this.#stopTask = undefined; });
    return task;
  }

  async close(): Promise<void> {
    this.#closed = true;
    await this.stop();
  }

  async #stop(): Promise<void> {
    this.#starting = false;
    this.#activeSink?.close();
    await this.#startTask;
    await this.#loop?.stop();
    await this.#quiesce();
    this.#loop = undefined;
    this.#runtime = undefined;
    this.#activeSink = undefined;
    this.#release?.();
    this.#release = undefined;
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
        const command = parseClientCommand(rawText(data), this.#outputRegistry());
        void this.#handle(command).catch((error) => this.#sendNotice("error", safeMessage(error)));
      } catch (error) {
        this.#sendNotice("error", safeMessage(error));
      }
    });
    this.#socket.once("close", () => {
      void this.close().catch(() => {});
    });
    this.#socket.on("error", () => {
      void this.close().catch(() => {});
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
    if (this.#closed || this.#startTask || this.#loop || this.#stopTask || this.#retirementFailed) return;
    const release = this.#acquire?.();
    if (this.#acquire && !release) {
      this.#sendNotice("error", "Other speech work is in progress. Try again after it finishes.");
      this.#send({ type: "session", phase: "stopped" });
      return;
    }
    this.#release = release;
    this.#starting = true;
    this.#send({ type: "session", phase: "loading" });
    this.#startTask = this.#open(command).finally(() => {
      this.#starting = false;
      this.#startTask = undefined;
      if (!this.#loop && !this.#retirementFailed) {
        this.#release?.();
        this.#release = undefined;
      }
    });
    await this.#startTask;
  }

  async #open(command: Extract<ClientCommand, { type: "start" }>): Promise<void> {
    try {
      const runtimeConfig =
        runtimeConfigFor(command, this.#runtimeConfig);
      const runtime = await this.#createRuntime(runtimeConfig);
      this.#runtime = runtime;
      if (!this.#starting || this.#closed) {
        await this.#quiesce();
        this.#runtime = undefined;
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
      if (!this.#starting || this.#closed) {
        await loop.stop();
        await this.#quiesce();
        this.#loop = undefined;
        this.#runtime = undefined;
        return;
      }
      this.#send({ type: "session", phase: "ready", label: runtime.label });
    } catch (error) {
      await this.#loop?.stop();
      await this.#quiesce();
      this.#loop = undefined;
      this.#runtime = undefined;
      this.#sendNotice("error", safeMessage(error));
      this.#send({ type: "session", phase: "stopped" });
    }
  }

  async #quiesce(): Promise<void> {
    try { await this.#runtime?.quiesce?.(); }
    catch (error) { this.#retirementFailed = true; throw error; }
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
    if (this.#closed || this.#stopTask) sink.close();
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
  defaults: BrowserRuntimeConfig,
): BrowserRuntimeConfig {
  const {
    responder: _responder,
    llmProvider: _llmProvider,
    llmReasoningEffort: _llmReasoningEffort,
    voice,
    speechOutput,
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
    ...(command.speechOutput === undefined && speechOutput === undefined ? {} : {
      speechOutput: structuredClone(command.speechOutput ?? speechOutput),
    }),
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
