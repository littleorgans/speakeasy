import { setTimeout as delay } from "node:timers/promises";
import { formatMs } from "../bench/format.ts";
import { readWavFrames, type WavAudio } from "../bench/wav.ts";
import {
  CAPTURE_FRAME_MS,
  CAPTURE_SAMPLE_RATE,
  startMicCapture,
  type MicCapture,
} from "../capture/ffmpeg.ts";
import type { FinalEvent, PartialEvent, STTSession } from "../contract.ts";
import { SherpaEngine } from "../engines/sherpa.ts";

/**
 * Live push-to-talk demo: Enter starts an utterance (frames flow to the
 * session, partials render live), Enter again releases (flush() commits, the
 * final prints with release->final latency). The SAME session serves every
 * utterance, which is the point: flush() commits without closing the session.
 *
 * Audio source is the microphone (ffmpeg capture helper) or, with --wav, a
 * recorded file replayed at real-time cadence from the top of each utterance.
 * --script drives the Enter presses deterministically for unattended runs.
 */

const FINAL_TIMEOUT_MS = 2_000;
const SCRIPT_EVENT_PATTERN = /^(start|release)@(\d+(?:\.\d+)?)ms?$/;
const USAGE =
  'usage: pnpm demo [--wav <path>] [--script "start@0ms,release@2200ms,..."] [--device <avfoundation-input>]';

type ScriptAction = "start" | "release";
type ScriptEvent = { action: ScriptAction; atMs: number };
type DemoOptions = {
  wavPath?: string;
  script?: ScriptEvent[];
  device?: string;
};

type FinalObservation = { text: string; at: number };

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const wav = options.wavPath
    ? await readWavFrames(options.wavPath, CAPTURE_FRAME_MS)
    : undefined;

  const engine = new SherpaEngine();
  await engine.prepare();
  const session = await engine.open({
    sampleRate: CAPTURE_SAMPLE_RATE,
    endpoint: { mode: "manual" },
  });

  const demo = new PttDemo(session, wav, options);
  await demo.run();
}

class PttDemo {
  readonly #session: STTSession;
  readonly #wav: WavAudio | undefined;
  readonly #options: DemoOptions;
  #capture: MicCapture | undefined;
  #talking = false;
  #utterance = 0;
  #lastPartial = "";
  #wavFeed: Promise<void> = Promise.resolve();
  #fatal: Error | undefined;
  #fatalNotify: (() => void) | undefined;

  constructor(
    session: STTSession,
    wav: WavAudio | undefined,
    options: DemoOptions,
  ) {
    this.#session = session;
    this.#wav = wav;
    this.#options = options;
    session.on("partial", (event: PartialEvent) => {
      this.#renderPartial(event.text);
    });
    session.on("error", (event: { err: unknown }) => {
      this.#setFatal(new Error(`session error: ${String(event.err)}`));
    });
  }

  async run(): Promise<void> {
    const source = this.#wav
      ? `wav replay ${this.#options.wavPath}`
      : "microphone (ffmpeg avfoundation)";
    console.log(`speak-easy ptt demo | engine=sherpa | source=${source}`);

    if (!this.#wav) {
      this.#capture = startMicCapture({
        device: this.#options.device,
        onFrame: (frame) => {
          if (this.#talking && !this.#fatal) {
            this.#session.pushAudio(frame);
          }
        },
        onError: (error) => {
          this.#setFatal(error);
        },
      });
    }

    try {
      if (this.#options.script) {
        await this.#runScript(this.#options.script);
      } else {
        await this.#runTty();
      }
    } finally {
      await this.#shutdown();
    }
    this.#throwIfFatal();
  }

  async #runScript(events: ScriptEvent[]): Promise<void> {
    console.log(
      `script mode: ${events.map((event) => `${event.action}@${event.atMs}ms`).join(",")}`,
    );
    const startAt = performance.now();
    for (const event of events) {
      const wait = event.atMs - (performance.now() - startAt);
      if (wait > 0) {
        await delay(wait);
      }
      this.#throwIfFatal();
      if (event.action === "start") {
        this.#handleStart();
      } else {
        await this.#handleRelease();
      }
    }
    if (this.#talking) {
      await this.#handleRelease();
    }
  }

  async #runTty(): Promise<void> {
    if (!process.stdin.isTTY) {
      throw new Error(
        "stdin is not a TTY; drive the demo deterministically with --script",
      );
    }
    console.log("Enter = start/release talking, Ctrl+C = quit.");
    process.stdin.setRawMode(true);
    process.stdin.resume();

    await new Promise<void>((resolve) => {
      this.#fatalNotify = resolve;
      // Serialize toggles so a release (which awaits the final) completes
      // before the next Enter is acted on.
      let queue = Promise.resolve();
      const enqueue = (task: () => Promise<void> | void): void => {
        queue = queue.then(task).catch((error: unknown) => {
          this.#setFatal(toError(error));
          resolve();
        });
      };
      process.stdin.on("data", (chunk: Buffer) => {
        for (const byte of chunk) {
          if (byte === 0x03) {
            enqueue(() => {
              resolve();
            });
            return;
          }
          if (byte === 0x0d || byte === 0x0a) {
            enqueue(async () => {
              if (this.#fatal) {
                resolve();
              } else if (this.#talking) {
                await this.#handleRelease();
              } else {
                this.#handleStart();
              }
            });
          }
        }
      });
    });
  }

  #handleStart(): void {
    if (this.#talking) {
      throw new Error("script error: start while already talking");
    }
    this.#talking = true;
    this.#utterance += 1;
    this.#lastPartial = "";
    console.log(`utterance ${this.#utterance}: talking (Enter to release)`);
    if (this.#wav) {
      this.#wavFeed = this.#feedWav(this.#wav);
    }
  }

  async #handleRelease(): Promise<void> {
    if (!this.#talking) {
      throw new Error("script error: release while not talking");
    }
    this.#talking = false;
    await this.#wavFeed;
    this.#clearPartialLine();
    this.#throwIfFatal();

    // Register the final waiter BEFORE flush(): the contract makes no
    // synchrony guarantee (sherpa emits during flush, others may not).
    const finalPromise = this.#nextFinal(FINAL_TIMEOUT_MS);
    const releaseAt = performance.now();
    this.#session.flush();
    const final = await finalPromise;

    if (final) {
      console.log(
        `utterance ${this.#utterance}: final=${JSON.stringify(final.text)} release->final=${formatMs(final.at - releaseAt)}`,
      );
    } else {
      console.log(
        `utterance ${this.#utterance}: no final within ${FINAL_TIMEOUT_MS}ms (no speech committed)`,
      );
    }
  }

  /** Never rejects; failures surface through #fatal at the next checkpoint. */
  async #feedWav(wav: WavAudio): Promise<void> {
    try {
      // Drift-corrected cadence: schedule each frame against an absolute
      // deadline so audio time tracks wall time. A fixed sleep per frame
      // accumulates timer overshoot and starves the release point of audio.
      const feedStart = performance.now();
      for (const [index, frame] of wav.frames.entries()) {
        if (!this.#talking || this.#fatal) {
          return;
        }
        this.#session.pushAudio(frame);
        const wait =
          feedStart + (index + 1) * CAPTURE_FRAME_MS - performance.now();
        if (wait > 0) {
          await delay(wait);
        }
      }
      this.#clearPartialLine();
      console.log("(wav exhausted; release to commit)");
    } catch (error) {
      this.#setFatal(toError(error));
    }
  }

  #nextFinal(timeoutMs: number): Promise<FinalObservation | undefined> {
    return new Promise((resolve) => {
      const handler = (event: FinalEvent): void => {
        clearTimeout(timer);
        resolve({ text: event.text, at: performance.now() });
      };
      const timer = setTimeout(() => {
        this.#session.off("final", handler);
        resolve(undefined);
      }, timeoutMs);
      this.#session.once("final", handler);
    });
  }

  #renderPartial(text: string): void {
    if (!this.#talking || text === this.#lastPartial) {
      return;
    }
    this.#lastPartial = text;
    if (process.stdout.isTTY) {
      process.stdout.write(`\r\x1b[K  ${text}`);
    } else {
      console.log(`  partial: ${text}`);
    }
  }

  #clearPartialLine(): void {
    if (process.stdout.isTTY && this.#lastPartial) {
      process.stdout.write("\r\x1b[K");
    }
  }

  #setFatal(error: Error): void {
    if (this.#fatal) {
      return;
    }
    this.#fatal = error;
    this.#fatalNotify?.();
  }

  async #shutdown(): Promise<void> {
    this.#talking = false;
    await this.#wavFeed;
    await this.#capture?.stop();
    await this.#session.end().catch(() => {});
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
    }
    process.stdin.pause();
  }

  #throwIfFatal(): void {
    if (this.#fatal) {
      throw this.#fatal;
    }
  }
}

function parseArgs(argv: string[]): DemoOptions {
  const options: DemoOptions = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    switch (arg) {
      case "--wav":
        options.wavPath = expectValue(argv, (index += 1), arg);
        break;
      case "--script":
        options.script = parseScript(expectValue(argv, (index += 1), arg));
        break;
      case "--device":
        options.device = expectValue(argv, (index += 1), arg);
        break;
      default:
        throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }
  return options;
}

function expectValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (value === undefined) {
    throw new Error(`${flag} requires a value\n${USAGE}`);
  }
  return value;
}

function parseScript(spec: string): ScriptEvent[] {
  const events = spec.split(",").map((part): ScriptEvent => {
    const match = SCRIPT_EVENT_PATTERN.exec(part.trim());
    if (!match) {
      throw new Error(
        `Invalid script event ${JSON.stringify(part)}; expected <start|release>@<ms>, e.g. start@0ms`,
      );
    }
    return { action: match[1] as ScriptAction, atMs: Number(match[2]) };
  });
  return events.sort((left, right) => left.atMs - right.atMs);
}

function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

main()
  .catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  })
  .finally(() => {
    // Raw-mode stdin keeps the event loop alive even after pause(); force the
    // exit only in interactive mode, where stdout is a TTY and writes are
    // synchronous (a piped script-mode run must drain stdout naturally).
    if (process.stdin.isTTY) {
      process.exit(process.exitCode ?? 0);
    }
  });
