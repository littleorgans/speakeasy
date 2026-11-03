import { WebSocket } from "ws";
import type { ReleaseAudioWork } from "./audio-lease.ts";
import type { LiveTake } from "./live-take.ts";
import { decodeMicFrame, rawBytes } from "./protocol.ts";
import { liveStream, resolveLiveSettings } from "./live-settings.ts";
import type { SetupComparison } from "./setup-comparison.ts";
import { getSetup } from "./setup-registry.ts";

export type LiveWordsCommand =
  | { type: "begin"; takeId: string; setupId: string; settings?: unknown }
  | { type: "end"; takeId: string };

/** Text commands on the live words socket; mic frames arrive as binary Float32 PCM. */
export function parseLiveWordsCommand(raw: string): LiveWordsCommand {
  let value: unknown;
  try { value = JSON.parse(raw); } catch { throw new Error("invalid JSON command"); }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("command must be an object");
  const { type, takeId, setupId, settings } = value as Record<string, unknown>;
  const fields = Object.keys(value);
  if (typeof takeId !== "string" || !/^[a-zA-Z0-9_.:-]{1,64}$/.test(takeId)) throw new Error("takeId required");
  if (type === "begin" && typeof setupId === "string" && fields.length === (settings === undefined ? 3 : 4)) {
    return { type, takeId, setupId, ...(settings === undefined ? {} : { settings }) };
  }
  if (type === "end" && fields.length === 2) return { type, takeId };
  throw new Error("unknown live words command");
}

/**
 * One page's live words: mic frames in while the owner holds the key, the
 * stream's settled and still changing words out, tagged with the take. The
 * page names the live model, which may differ from the final one, and its
 * settings; the first event names the stream and settings applied. A setup
 * whose engine gives final text only is refused.
 */
export class LiveWordsSession {
  readonly #socket: WebSocket;
  readonly #comparison: SetupComparison;
  readonly #acquire: () => ReleaseAudioWork | undefined;
  #take: { id: string; live: LiveTake } | undefined;

  constructor(options: { socket: WebSocket; comparison: SetupComparison; acquire(): ReleaseAudioWork | undefined }) {
    this.#socket = options.socket;
    this.#comparison = options.comparison;
    this.#acquire = options.acquire;
    this.#socket.on("message", (data, binary) => {
      try {
        if (binary) this.#take?.live.push(decodeMicFrame(rawBytes(data)));
        else this.#handle(parseLiveWordsCommand(data.toString()));
      } catch (error) {
        this.#send({ type: "error", takeId: this.#take?.id ?? null, message: error instanceof Error ? error.message : String(error) });
      }
    });
  }

  /** Lets an in flight feed settle, so the model stays loaded for the final pass. */
  close(): Promise<void> {
    const take = this.#take;
    this.#take = undefined;
    return take ? take.live.stop() : Promise.resolve();
  }

  #handle(command: LiveWordsCommand): void {
    if (command.type === "end") {
      if (this.#take?.id === command.takeId) void this.#take.live.stop();
      return;
    }
    void this.#take?.live.stop();
    this.#take = undefined;
    const stream = liveStream(command.setupId);
    if (!stream) throw new Error(`${getSetup(command.setupId).label} gives final text only`);
    const settings = resolveLiveSettings(stream, command.settings);
    const takeId = command.takeId;
    const live = this.#comparison.live(stream.setupId, { settings, acquire: this.#acquire,
      onEvent: (event) => { if (this.#take?.live === live) this.#send({ ...event, takeId }); } });
    this.#take = { id: takeId, live };
    this.#send({ type: "stream", takeId, setupId: stream.setupId, settings });
  }

  #send(event: Record<string, unknown>): void {
    if (this.#socket.readyState === WebSocket.OPEN) this.#socket.send(JSON.stringify(event));
  }
}

