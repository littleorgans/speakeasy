import { createInterface } from "node:readline/promises";
import { setTimeout as delay } from "node:timers/promises";
import { formatMs } from "../bench/format.ts";
import { readWavFrames, type WavAudio } from "../bench/wav.ts";
import {
  CAPTURE_FRAME_MS,
  CAPTURE_SAMPLE_RATE,
  DEFAULT_MIC_DEVICE,
  listAudioDevices,
  resolveDefaultMicDevice,
  startMicCapture,
  type MicCapture,
} from "../capture/ffmpeg.ts";
import type { FinalEvent, PartialEvent, STTSession } from "../contract.ts";
import {
  DEFAULT_CORPUS_DIR,
  saveCorpusPair,
  type CorpusSidecarMeta,
} from "../corpus/store.ts";
import { SherpaEngine } from "../engines/sherpa.ts";

/**
 * Live push-to-talk demo: Enter starts an utterance (frames flow to the
 * session, partials render live with an input level meter), Enter again
 * releases (flush() commits, the final prints with release->final latency).
 * The SAME session serves every utterance, which is the point: flush()
 * commits without closing the session.
 *
 * Audio source is the microphone (ffmpeg capture helper) or, with --wav, a
 * recorded file replayed at real-time cadence from the top of each utterance.
 * --script drives the Enter presses deterministically for unattended runs.
 *
 * The demo is a labeled-corpus collector by default: each kept utterance is
 * written as a wav + json sidecar pair (see src/corpus/store.ts) that
 * `pnpm bench --corpus <dir>` scores for WER. --save <dir> overrides the
 * directory, --no-save disarms, --save-all skips the prompts.
 */

const FINAL_TIMEOUT_MS = 2_000;
const LEVEL_WINDOW_FRAMES = 25; // 500ms rolling peak
const LEVEL_RENDER_INTERVAL_MS = 100;
const LEVEL_BAR_CELLS = 10;
const SCRIPT_EVENT_PATTERN = /^(start|release)@(\d+(?:\.\d+)?)ms?$/;
const USAGE =
  'usage: pnpm demo [--wav <path>] [--script "start@0ms,release@2200ms,..."] [--device <index>] [--list-devices] [--save <dir>] [--no-save] [--save-all]\n' +
  `  corpus collection is ON by default (dir: ${DEFAULT_CORPUS_DIR}/): after each final, s = save wav+json pair and label it, any other key = discard\n` +
  "  --save <dir>  override the corpus directory\n" +
  "  --no-save     disarm corpus collection\n" +
  "  --save-all    save every utterance without prompting (expected=null; label the sidecars by hand)";

type ScriptAction = "start" | "release";
type ScriptEvent = { action: ScriptAction; atMs: number };
type DemoOptions = {
  wavPath?: string;
  script?: ScriptEvent[];
  device?: string;
  listDevices?: boolean;
  saveDir?: string;
  noSave?: boolean;
  saveAll?: boolean;
};

type FinalObservation = { text: string; at: number };

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (options.listDevices) {
    const devices = await listAudioDevices();
    console.log("audio input devices (ffmpeg avfoundation):");
    for (const device of devices) {
      console.log(`  [${device.index}] ${device.name}`);
    }
    console.log(
      `select with --device <index>; default is the system default input (${DEFAULT_MIC_DEVICE})`,
    );
    return;
  }

  resolveSavePlan(options);

  const wav = options.wavPath
    ? await readWavFrames(options.wavPath, CAPTURE_FRAME_MS)
    : undefined;

  const engine = new SherpaEngine();
  await engine.prepare();
  const session = await engine.open({
    sampleRate: CAPTURE_SAMPLE_RATE,
    endpoint: { mode: "manual" },
  });

  const demo = new PttDemo(session, wav, options, engine.label);
  await demo.run();
}

class PttDemo {
  readonly #session: STTSession;
  readonly #wav: WavAudio | undefined;
  readonly #options: DemoOptions;
  readonly #engineLabel: string;
  #capture: MicCapture | undefined;
  #micSpec: string = DEFAULT_MIC_DEVICE;
  #sourceLabel = "";
  #talking = false;
  #quitting = false;
  #utterance = 0;
  #lastPartial = "";
  #framePeaks: number[] = [];
  #utterancePeak = 0;
  #utteranceFrames: Float32Array[] = [];
  #levelTimer: NodeJS.Timeout | undefined;
  #wavFeed: Promise<void> = Promise.resolve();
  #fatal: Error | undefined;
  #fatalNotify: (() => void) | undefined;
  #keyWaiter: ((byte: number) => void) | undefined;
  #lineInputActive = false;

  constructor(
    session: STTSession,
    wav: WavAudio | undefined,
    options: DemoOptions,
    engineLabel: string,
  ) {
    this.#session = session;
    this.#wav = wav;
    this.#options = options;
    this.#engineLabel = engineLabel;
    session.on("partial", (event: PartialEvent) => {
      if (this.#talking) {
        this.#lastPartial = event.text;
        this.#renderLive();
      }
    });
    session.on("error", (event: { err: unknown }) => {
      this.#setFatal(new Error(`session error: ${String(event.err)}`));
    });
  }

  async run(): Promise<void> {
    if (this.#wav) {
      this.#sourceLabel = `wav replay ${this.#options.wavPath}`;
    } else {
      const mic = await this.#resolveMic();
      this.#micSpec = mic.spec;
      this.#sourceLabel = `microphone ${mic.label}`;
    }
    console.log(
      `speak-easy ptt demo | engine=sherpa | source=${this.#sourceLabel}`,
    );

    if (!this.#wav) {
      this.#capture = startMicCapture({
        device: this.#micSpec,
        onFrame: (frame) => {
          if (this.#talking && !this.#fatal) {
            this.#ingestFrame(frame);
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
    // Raw mode swallows SIGINT, so 0x03 handling below is the only Ctrl+C
    // path; restore the TTY on every exit, even a hard process.exit().
    process.on("exit", restoreTty);

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
          if (byte === 0x03 || byte === 0x04) {
            // Ctrl+C / Ctrl+D: quit IMMEDIATELY, never behind the queue (a
            // stuck release must not make the demo unkillable). A second
            // press hard-exits. A pending save prompt is answered with the
            // interrupt byte so the queued release discards and completes.
            if (this.#quitting) {
              process.exit(130);
            }
            this.#quitting = true;
            this.#clearLiveLine();
            console.log("quitting...");
            this.#answerKeyWaiter(byte);
            resolve();
            return;
          }
          if (this.#keyWaiter) {
            // A save prompt owns the next keystroke; nothing else sees it.
            this.#answerKeyWaiter(byte);
            continue;
          }
          if (this.#lineInputActive) {
            // readline owns the stream during the expected-label prompt.
            continue;
          }
          if (byte === 0x0d || byte === 0x0a) {
            enqueue(async () => {
              if (this.#fatal || this.#quitting) {
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
    this.#framePeaks = [];
    this.#utterancePeak = 0;
    this.#utteranceFrames = [];
    console.log(`utterance ${this.#utterance}: talking (Enter to release)`);
    if (process.stdout.isTTY) {
      this.#levelTimer = setInterval(() => {
        this.#renderLive();
      }, LEVEL_RENDER_INTERVAL_MS);
    }
    if (this.#wav) {
      this.#wavFeed = this.#feedWav(this.#wav);
    }
  }

  async #handleRelease(): Promise<void> {
    if (!this.#talking) {
      throw new Error("script error: release while not talking");
    }
    this.#talking = false;
    this.#stopLevelTimer();
    await this.#wavFeed;
    this.#clearLiveLine();
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
      if (this.#options.saveDir && !this.#quitting) {
        await this.#saveUtterance(final, releaseAt);
      }
    } else {
      console.log(
        `utterance ${this.#utterance}: no speech committed (peak level ${this.#utterancePeak.toFixed(3)} on ${this.#sourceLabel}). If the level stays at 0, check mic permission or pick another input with --list-devices / --device <index>.`,
      );
      if (this.#options.saveDir) {
        console.log(`utterance ${this.#utterance}: nothing to save`);
      }
    }
    this.#utteranceFrames = [];
  }

  /**
   * Corpus collection: with --save-all every utterance is written unlabeled
   * (expected=null, for hand labeling); otherwise a single raw keystroke
   * decides (s = save, anything else = discard) and the expected transcript
   * is accepted or corrected on one line.
   */
  async #saveUtterance(
    final: FinalObservation,
    releaseAt: number,
  ): Promise<void> {
    const frames = this.#utteranceFrames;
    this.#utteranceFrames = [];
    if (frames.length === 0) {
      console.log(`utterance ${this.#utterance}: nothing to save (no frames)`);
      return;
    }

    let expected: string | null = null;
    if (!this.#options.saveAll) {
      const key = await this.#promptKey(
        "  save? [s = save, any other key = discard] ",
      );
      if (key !== "s") {
        console.log(`utterance ${this.#utterance}: discarded`);
        return;
      }
      expected = await this.#promptExpected(final.text);
    }

    const meta: CorpusSidecarMeta = {
      recordedAt: new Date().toISOString(),
      hypothesis: final.text,
      expected,
      engineLabel: this.#engineLabel,
      endpoint: "manual",
      flushToFinalMs: roundMs(final.at - releaseAt),
      device: this.#deviceSpec(),
      peakLevel: Number(this.#utterancePeak.toFixed(4)),
    };
    const saved = await saveCorpusPair(
      this.#options.saveDir!,
      frames,
      CAPTURE_SAMPLE_RATE,
      meta,
    );
    console.log(
      `utterance ${this.#utterance}: saved ${saved.wavPath} + sidecar${expected === null ? " (expected=null; label by hand before scoring)" : ""}`,
    );
  }

  /** Resolve with the next raw keystroke; the stdin handler routes it here. */
  #promptKey(prompt: string): Promise<string> {
    process.stdout.write(prompt);
    return new Promise((resolve) => {
      this.#keyWaiter = (byte) => {
        process.stdout.write("\n");
        resolve(String.fromCharCode(byte).toLowerCase());
      };
    });
  }

  #answerKeyWaiter(byte: number): void {
    const waiter = this.#keyWaiter;
    this.#keyWaiter = undefined;
    waiter?.(byte);
  }

  /** Line input needs cooked mode; restore raw mode afterwards. */
  async #promptExpected(hypothesis: string): Promise<string> {
    process.stdin.setRawMode(false);
    this.#lineInputActive = true;
    const readlinePrompt = createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    try {
      const answer = (
        await readlinePrompt.question(
          "  expected [Enter = transcript is correct]: ",
        )
      ).trim();
      return answer || hypothesis;
    } finally {
      readlinePrompt.close();
      this.#lineInputActive = false;
      if (process.stdin.isTTY) {
        process.stdin.setRawMode(true);
        process.stdin.resume();
      }
    }
  }

  /** --device overrides; otherwise the name-preference list, else :default. */
  async #resolveMic(): Promise<{ spec: string; label: string }> {
    if (this.#options.device) {
      return {
        spec: this.#options.device,
        label: await resolveDeviceLabel(this.#options.device),
      };
    }
    const resolved = await resolveDefaultMicDevice();
    return {
      spec: resolved.spec,
      label: resolved.device
        ? `[${resolved.device.index}] ${resolved.device.name} (preferred name match)`
        : await resolveDeviceLabel(resolved.spec),
    };
  }

  #deviceSpec(): string {
    return this.#wav ? `wav:${this.#options.wavPath}` : this.#micSpec;
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
        this.#ingestFrame(frame);
        const wait =
          feedStart + (index + 1) * CAPTURE_FRAME_MS - performance.now();
        if (wait > 0) {
          await delay(wait);
        }
      }
      this.#clearLiveLine();
      console.log("(wav exhausted; release to commit)");
    } catch (error) {
      this.#setFatal(toError(error));
    }
  }

  /** Single ingest path for mic and wav frames: level tracking + push. */
  #ingestFrame(frame: Float32Array): void {
    let peak = 0;
    for (const sample of frame) {
      const abs = Math.abs(sample);
      if (abs > peak) {
        peak = abs;
      }
    }
    this.#framePeaks.push(peak);
    if (this.#framePeaks.length > LEVEL_WINDOW_FRAMES) {
      this.#framePeaks.shift();
    }
    if (peak > this.#utterancePeak) {
      this.#utterancePeak = peak;
    }
    if (this.#options.saveDir) {
      this.#utteranceFrames.push(frame);
    }
    this.#session.pushAudio(frame);
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

  /** One overwriting status line: [level bar] peak + latest partial. */
  #renderLive(): void {
    if (!this.#talking) {
      return;
    }
    const level = this.#framePeaks.length > 0 ? Math.max(...this.#framePeaks) : 0;
    if (process.stdout.isTTY) {
      process.stdout.write(
        `\r\x1b[K  [${levelBar(level)}] ${level.toFixed(2)} ${this.#lastPartial}`,
      );
    } else if (this.#lastPartial) {
      console.log(`  partial: ${this.#lastPartial} (level=${level.toFixed(2)})`);
    }
  }

  #clearLiveLine(): void {
    if (process.stdout.isTTY) {
      process.stdout.write("\r\x1b[K");
    }
  }

  #stopLevelTimer(): void {
    if (this.#levelTimer) {
      clearInterval(this.#levelTimer);
      this.#levelTimer = undefined;
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
    this.#stopLevelTimer();
    await this.#wavFeed;
    await this.#capture?.stop();
    await this.#session.end().catch(() => {});
    restoreTty();
    process.stdin.pause();
  }

  #throwIfFatal(): void {
    if (this.#fatal) {
      throw this.#fatal;
    }
  }
}

async function resolveDeviceLabel(spec: string): Promise<string> {
  if (spec === DEFAULT_MIC_DEVICE) {
    return `${DEFAULT_MIC_DEVICE} (system default input)`;
  }
  const indexMatch = /^:(\d+)$/.exec(spec);
  if (!indexMatch) {
    return spec;
  }
  const devices = await listAudioDevices().catch(() => []);
  const device = devices.find(
    (candidate) => candidate.index === Number(indexMatch[1]),
  );
  return device ? `[${device.index}] ${device.name}` : spec;
}

function levelBar(level: number): string {
  const filled = Math.min(
    LEVEL_BAR_CELLS,
    Math.round(level * LEVEL_BAR_CELLS),
  );
  return "#".repeat(filled).padEnd(LEVEL_BAR_CELLS, "-");
}

function restoreTty(): void {
  if (process.stdin.isTTY) {
    try {
      process.stdin.setRawMode(false);
    } catch {
      // The TTY may already be gone at exit time.
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
      case "--device": {
        const value = expectValue(argv, (index += 1), arg);
        options.device = /^\d+$/.test(value) ? `:${value}` : value;
        break;
      }
      case "--list-devices":
        options.listDevices = true;
        break;
      case "--save":
        options.saveDir = expectValue(argv, (index += 1), arg);
        break;
      case "--no-save":
        options.noSave = true;
        break;
      case "--save-all":
        options.saveAll = true;
        break;
      default:
        throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }
  return options;
}

/**
 * Corpus collection is armed by default. The keep/label prompts need an
 * interactive TTY, so unattended runs (--script or piped stdin) quietly
 * disarm the default; an EXPLICIT --save there is an error unless --save-all
 * removes the prompting.
 */
function resolveSavePlan(options: DemoOptions): void {
  if (options.noSave) {
    if (options.saveDir !== undefined || options.saveAll) {
      throw new Error(`--no-save conflicts with --save/--save-all\n${USAGE}`);
    }
    return;
  }
  if (options.saveAll) {
    options.saveDir ??= DEFAULT_CORPUS_DIR;
    return;
  }
  const unattended = Boolean(options.script) || !process.stdin.isTTY;
  if (unattended) {
    if (options.saveDir !== undefined) {
      throw new Error(
        "--save prompts for keep/label after each utterance; unattended runs (--script or piped stdin) need --save-all",
      );
    }
    console.log(
      "corpus saving disarmed (unattended run); pass --save-all to keep every utterance",
    );
    return;
  }
  options.saveDir ??= DEFAULT_CORPUS_DIR;
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

/** Keep sidecar numbers plain: one decimal of milliseconds. */
function roundMs(value: number): number {
  return Number(value.toFixed(1));
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
