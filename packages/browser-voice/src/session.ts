import { Buffer } from "node:buffer";
import { randomUUID } from "node:crypto";
import {
  ConversationLoop,
  type ConversationRuntime,
  type RuntimeConfig,
  type RuntimeOptions,
} from "@speakeasy/convo-engine";
import { WebSocket, type RawData } from "ws";
import { BrowserAudioSink, BrowserAudioSource } from "./browser-audio.ts";
import { ProviderError, sameSpeechOutputConfig, type SpeechOutputRegistry } from "@speakeasy/speech-io/providers";
import type { ReleaseAudioWork } from "./audio-lease.ts";
import type { ProviderDispatchContext, ProviderPolicy } from "./provider-policy.ts";
import type { SpeechOutputCatalogue } from "./speech-output-catalog.ts";
import { safeSpeechOutputError, SpeechOutputError, type SpeechOutputEvent } from "./speech-output-protocol.ts";
import {
  decodeMicFrame,
  parseClientCommand,
  type BrowserHostEvent,
  type ClientCommand,
} from "./protocol.ts";

export type BrowserRuntimeConfig = RuntimeConfig;
export type BrowserConversationRuntime = ConversationRuntime;
export type BrowserRuntimeContext = Pick<RuntimeOptions, "speechOutputContext">;
export type RuntimeFactory = (
  config: BrowserRuntimeConfig,
  context?: BrowserRuntimeContext,
) => Promise<BrowserConversationRuntime>;

type RoomOperation = { id: string; controller: AbortController };
type RoomSelection = Extract<SpeechOutputEvent, { type: "output-selected" }>;

/** One browser socket owns one conversation loop and its audio adapters. */
export class BrowserConversationSession {
  readonly #socket: WebSocket;
  readonly #sessionId: string;
  readonly #runtimeConfig: BrowserRuntimeConfig;
  readonly #createRuntime: RuntimeFactory;
  readonly #acquire: (() => ReleaseAudioWork | undefined) | undefined;
  readonly #outputRegistry: () => SpeechOutputRegistry;
  readonly #policy: ProviderPolicy | undefined;
  readonly #outputCatalogue: (() => SpeechOutputCatalogue) | undefined;
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
  #selection: RoomSelection | undefined;
  #operation: RoomOperation | undefined;

  constructor(options: {
    socket: WebSocket;
    sessionId: string;
    runtimeConfig: BrowserRuntimeConfig;
    createRuntime: RuntimeFactory;
    acquire?: () => ReleaseAudioWork | undefined;
    outputRegistry?: () => SpeechOutputRegistry;
    policy?: ProviderPolicy;
    outputCatalogue?: () => SpeechOutputCatalogue;
  }) {
    this.#socket = options.socket;
    this.#sessionId = options.sessionId;
    this.#runtimeConfig = options.runtimeConfig;
    this.#createRuntime = options.createRuntime;
    this.#acquire = options.acquire;
    this.#outputRegistry = options.outputRegistry ?? (() => []);
    this.#policy = options.policy;
    this.#outputCatalogue = options.outputCatalogue;
    this.#wireSocket();
  }

  stop(): Promise<void> {
    this.#invalidate();
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
    if (this.#retirementFailed) throw new ProviderError("PROVIDER_FAILED");
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
      this.#receive(rawText(data));
    });
    this.#socket.once("close", () => {
      void this.close().catch(() => {});
    });
    this.#socket.on("error", () => {
      void this.close().catch(() => {});
    });
  }

  #receive(raw: string): void {
    let type: unknown;
    let requestId: string | null = null;
    try {
      type = JSON.parse(raw)?.type;
      if (type === "select-output" && this.#stopped()) this.#invalidate();
      const command = parseClientCommand(raw, this.#outputRegistry());
      if (command.type === "select-output") {
        requestId = command.requestId;
        this.#select(command);
      } else {
        void this.#handle(command).catch((error) => this.#sendNotice("error", safeMessage(error)));
      }
    } catch (error) {
      if (type === "select-output") {
        this.#send({ type: "output-selection-error", requestId, ...safeSpeechOutputError(error) });
      } else {
        if (this.#stopped()) {
          this.#invalidate();
          this.#send({ type: "session", phase: "stopped" });
        }
        this.#sendNotice("error", error instanceof ProviderError ? error.message : safeMessage(error));
      }
    }
  }

  #stopped(): boolean {
    return !this.#closed && !this.#startTask && !this.#loop && !this.#stopTask && !this.#retirementFailed;
  }

  #select(command: Extract<ClientCommand, { type: "select-output" }>): void {
    if (!this.#stopped()) throw new SpeechOutputError("BUSY");
    if (!this.#policy || !this.#outputCatalogue) throw new ProviderError("PROVIDER_UNAVAILABLE");
    if (command.catalogueRevision !== this.#outputCatalogue().revision) throw new ProviderError("SELECTION_STALE");
    const selected = this.#policy.select({ sessionId: this.#sessionId, direction: "output",
      catalogueRevision: command.catalogueRevision, config: command.selection });
    this.#selection = { type: "output-selected", requestId: command.requestId,
      catalogueRevision: selected.catalogueRevision, selectionRevision: selected.selectionRevision,
      selection: structuredClone(command.selection) };
    this.#send(this.#selection);
  }

  #invalidate(): void {
    this.#starting = false;
    const operation = this.#operation;
    this.#operation = undefined;
    this.#selection = undefined;
    operation?.controller.abort();
    this.#policy?.revoke(this.#sessionId, "output");
    void this.#activeSink?.interrupt();
    this.#activeSink?.close();
  }

  #assertCurrent(operation: RoomOperation): void {
    if (this.#operation !== operation || this.#closed || operation.controller.signal.aborted) {
      throw new ProviderError("SELECTION_STALE");
    }
  }

  async #handle(command: Exclude<ClientCommand, { type: "select-output" }>): Promise<void> {
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
    if (this.#closed || this.#startTask || this.#loop) return;
    if (this.#stopTask || this.#retirementFailed) {
      this.#invalidate();
      this.#sendNotice("error", "Previous speech work has not retired.");
      this.#send({ type: "session", phase: "stopped" });
      return;
    }
    const release = this.#acquire?.();
    if (this.#acquire && !release) {
      this.#invalidate();
      this.#sendNotice("error", "Other speech work is in progress. Try again after it finishes.");
      this.#send({ type: "session", phase: "stopped" });
      return;
    }
    this.#release = release;
    const operation: RoomOperation = { id: randomUUID(), controller: new AbortController() };
    this.#operation = operation;
    this.#starting = true;
    this.#send({ type: "session", phase: "loading" });
    this.#startTask = this.#open(command, operation).finally(() => {
      this.#starting = false;
      this.#startTask = undefined;
      if (!this.#loop && !this.#retirementFailed) {
        this.#release?.();
        this.#release = undefined;
      }
    });
    await this.#startTask;
  }

  async #open(command: Extract<ClientCommand, { type: "start" }>, operation: RoomOperation): Promise<void> {
    try {
      const runtimeConfig =
        runtimeConfigFor(command, this.#runtimeConfig);
      const runtime = await this.#createSelectedRuntime(command, runtimeConfig, operation);
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
          createSink: (sampleRate) => this.#createSink(sampleRate, operation),
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
          onEvent: (event) => {
            if (this.#loop === loop && this.#operation === operation && !this.#stopTask && !this.#closed) this.#send(event);
          },
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
      const current = this.#operation === operation;
      if (current) this.#invalidate();
      await this.#loop?.stop();
      await this.#quiesce();
      this.#loop = undefined;
      this.#runtime = undefined;
      if (current && !this.#closed) {
        this.#sendNotice("error", safeSpeechOutputError(error).message);
        this.#send({ type: "session", phase: "stopped" });
      }
    }
  }

  #createSelectedRuntime(
    command: Extract<ClientCommand, { type: "start" }>,
    config: BrowserRuntimeConfig,
    operation: RoomOperation,
  ): Promise<BrowserConversationRuntime> {
    const create = (dispatch?: ProviderDispatchContext) => {
      const guarded = <T>(read: () => T): T => {
        this.#assertCurrent(operation);
        try { return read(); }
        catch (error) {
          this.#invalidate();
          this.#sendNotice("error", safeSpeechOutputError(error).message);
          // Opening may still be assigning #startTask. Retire after it is tracked.
          void Promise.resolve().then(() => this.stop()).catch((failure) =>
            this.#sendNotice("error", safeSpeechOutputError(failure).message));
          throw error;
        }
      };
      const beforeRequest = () => { guarded(() => dispatch?.recheck()); };
      beforeRequest();
      return this.#createRuntime(config, { speechOutputContext: {
        signal: operation.controller.signal,
        beforeRequest,
        getKey: dispatch?.apiKey ? () => guarded(() => dispatch.apiKey!()) : undefined,
      } });
    };
    const hosted = config.speechOutput ? config.speechOutput.provider.kind !== "local" : config.ttsEngine === "cartesia";
    if (command.engine === "realtime" ||
        (!hosted && command.catalogueRevision === undefined && command.selectionRevision === undefined)) {
      this.#selection = undefined;
      this.#policy?.revoke(this.#sessionId, "output");
      return create();
    }
    const selected = this.#selection;
    if (!command.speechOutput || !command.catalogueRevision || !command.selectionRevision ||
        !selected || !this.#policy || !this.#outputCatalogue) throw new ProviderError("SELECTION_REQUIRED");
    if (command.catalogueRevision !== selected.catalogueRevision ||
        command.selectionRevision !== selected.selectionRevision ||
        this.#outputCatalogue().revision !== selected.catalogueRevision ||
        !sameSpeechOutputConfig(command.speechOutput, selected.selection)) throw new ProviderError("SELECTION_STALE");
    return this.#policy.dispatch({ sessionId: this.#sessionId, direction: "output",
      config: structuredClone(command.speechOutput), catalogueRevision: command.catalogueRevision,
      selectionRevision: command.selectionRevision,
      source: { kind: "live-response-text", sessionId: this.#sessionId, operationId: operation.id } }, create);
  }

  async #quiesce(): Promise<void> {
    if (this.#retirementFailed) throw new ProviderError("PROVIDER_FAILED");
    try { await this.#runtime?.quiesce?.(); }
    catch (error) { this.#retirementFailed = true; throw error; }
  }

  #createSink(sampleRate: number, operation: RoomOperation): BrowserAudioSink {
    const sink = new BrowserAudioSink({
      playbackId: this.#nextPlaybackId++,
      sampleRate,
      sendEvent: (event) => {
        if (this.#activeSink === sink && (this.#operation === operation ||
            (event.type === "playback" && event.action === "clear"))) this.#send(event);
      },
      sendAudio: (packet) => {
        if (this.#operation === operation && this.#socket.readyState === WebSocket.OPEN) {
          this.#socket.send(packet);
        }
      },
    });
    this.#activeSink = sink;
    if (this.#operation !== operation || this.#closed || this.#stopTask) sink.close();
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
