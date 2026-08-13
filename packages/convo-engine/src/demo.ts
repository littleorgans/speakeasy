import {
  CAPTURE_FRAME_MS,
  CAPTURE_SAMPLE_RATE,
  createSegmentPlayer,
  readWavFrames,
  resolveDefaultMicDevice,
  startMicCapture,
  type MicCapture,
  type WavAudio,
} from "@speakeasy/speech-io";
import { ConversationLoop, type AudioSource } from "./loop.ts";
import {
  createConversationRuntime,
  type ResponderKind,
  type RuntimeConfig,
  type TtsEngine,
} from "./runtime.ts";
import type { ConversationEvent } from "./events.ts";
import { formatSessionSummary } from "./metrics.ts";

/**
 * Terminal speech-to-speech demo: the composition root that wires real engines
 * (kroko STT + rewrite, Cerebras LLM, sherpa TTS) into the conversation loop and
 * prints state changes, live partials, and the per-turn latency line.
 *
 * `--wav <path>` replaces the microphone with a recorded utterance (plus a
 * silence tail so eager endpointing closes the turn), which drives the mic-free
 * end-to-end smoke. Requires CEREBRAS_API_KEY in the environment; it fails fast
 * with a clear message if absent and never echoes the key.
 */

const FRAME_SAMPLES = (CAPTURE_SAMPLE_RATE * CAPTURE_FRAME_MS) / 1_000;
/** Silence appended after a --wav utterance so the endpoint detector fires. */
const WAV_SILENCE_TAIL_MS = 700;

type DemoArgs = RuntimeConfig & {
  system: string | undefined;
  wavPath: string | undefined;
  maxTurns: number | undefined;
  barge: boolean;
};

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  let runtime;
  try {
    runtime = await createConversationRuntime(args);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
    return;
  }

  const mic = args.wavPath
    ? new WavAudioSource(await readWavFrames(args.wavPath, CAPTURE_FRAME_MS))
    : new MicAudioSource((await resolveDefaultMicDevice()).spec);

  const loop = new ConversationLoop(
    {
      stt: runtime.stt,
      responder: runtime.responder,
      mic,
      createSink: (sampleRate) => createSegmentPlayer(sampleRate),
    },
    {
      systemPrompt: args.system,
      sttConfig: {
        sampleRate: CAPTURE_SAMPLE_RATE,
        endpoint: { mode: "eager" },
      },
      maxTurns: args.maxTurns,
      barge: args.barge,
      onEvent: printTerminalEvent,
    },
  );

  const onSigint = (): void => {
    console.log("\nstopping...");
    void loop.stop();
  };
  process.on("SIGINT", onSigint);

  // Interactive mic runs: any key interrupts the assistant, Ctrl+C quits. Raw
  // mode swallows SIGINT, so 0x03 is handled here instead.
  const interactive = process.stdin.isTTY && !args.wavPath;
  const restoreKeys = interactive ? setupKeys(loop) : undefined;

  console.log(
    `speak-easy convo | ${runtime.label} | source=${args.wavPath ? `wav ${args.wavPath}` : "microphone"}${args.barge ? " | barge-in on (use headphones)" : ""}`,
  );
  console.log(
    interactive
      ? "Speak, pause to send. Any key interrupts, Ctrl+C quits."
      : "Speak, pause to send. Ctrl+C to quit.",
  );

  await loop.start();
  await loop.done;

  restoreKeys?.();
  process.removeListener("SIGINT", onSigint);
  console.log(formatSessionSummary([...loop.metrics]));
}

function clearPartial(): void {
  if (process.stdout.isTTY) {
    process.stdout.write("\r\x1b[K");
  }
}

function printTerminalEvent(event: ConversationEvent): void {
  if (event.type === "transcript" && event.role === "user" && !event.final) {
    if (process.stdout.isTTY) {
      process.stdout.write(`\r\x1b[K… ${event.text}`);
    }
    return;
  }
  clearPartial();
  if (event.type === "state") {
    console.log(`[state] ${event.state}`);
  } else if (event.type === "notice") {
    const output = event.level === "error" ? console.error : console.log;
    output(event.message);
  }
}

/** Raw-mode keys: any key interrupts the assistant, Ctrl+C quits. */
function setupKeys(loop: ConversationLoop): () => void {
  process.stdin.setRawMode(true);
  process.stdin.resume();
  const onData = (chunk: Buffer): void => {
    for (const byte of chunk) {
      if (byte === 0x03) {
        console.log("\nstopping...");
        void loop.stop();
        return;
      }
      loop.interrupt();
    }
  };
  const restore = (): void => {
    if (process.stdin.isTTY) {
      try {
        process.stdin.setRawMode(false);
      } catch {
        // TTY may already be gone at exit.
      }
    }
  };
  process.stdin.on("data", onData);
  process.once("exit", restore);
  return () => {
    process.stdin.off("data", onData);
    restore();
    process.stdin.pause();
  };
}

/** Live microphone via the ffmpeg capture helper. */
class MicAudioSource implements AudioSource {
  readonly #device: string;
  #capture: MicCapture | undefined;

  constructor(device: string) {
    this.#device = device;
  }

  start(handlers: {
    onFrame: (frame: Float32Array) => void;
    onError: (error: Error) => void;
  }): void {
    this.#capture = startMicCapture({
      device: this.#device,
      onFrame: handlers.onFrame,
      onError: handlers.onError,
    });
  }

  async stop(): Promise<void> {
    await this.#capture?.stop();
  }
}

/** Replays a recorded utterance at real-time cadence, then a silence tail. */
class WavAudioSource implements AudioSource {
  readonly #frames: Float32Array[];
  #timer: NodeJS.Timeout | undefined;
  #stopped = false;

  constructor(wav: WavAudio) {
    this.#frames = wav.frames;
  }

  start(handlers: { onFrame: (frame: Float32Array) => void }): void {
    const silence = Array.from(
      { length: Math.round(WAV_SILENCE_TAIL_MS / CAPTURE_FRAME_MS) },
      () => new Float32Array(FRAME_SAMPLES),
    );
    const frames = [...this.#frames, ...silence];
    let index = 0;
    const tick = (): void => {
      if (this.#stopped || index >= frames.length) {
        return;
      }
      handlers.onFrame(frames[index]!);
      index += 1;
      this.#timer = setTimeout(tick, CAPTURE_FRAME_MS);
    };
    tick();
  }

  stop(): void {
    this.#stopped = true;
    if (this.#timer) {
      clearTimeout(this.#timer);
    }
  }
}

function parseArgs(argv: string[]): DemoArgs {
  const args: DemoArgs = {
    responder: "cascade",
    ttsEngine: "sherpa",
    ttsModel: undefined,
    voice: undefined,
    system: undefined,
    wavPath: undefined,
    maxTurns: undefined,
    barge: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = (): string => {
      const next = argv[index + 1];
      if (next === undefined || next.startsWith("--")) {
        throw new Error(`missing value for ${flag}`);
      }
      index += 1;
      return next;
    };
    switch (flag) {
      case "--": // pnpm forwards the separator verbatim; ignore it
        break;
      case "--barge":
        args.barge = true;
        break;
      case "--responder":
        args.responder = parseResponder(value());
        break;
      case "--model":
        args.llmModel = value();
        break;
      case "--tts":
        args.ttsEngine = parseTtsEngine(value());
        break;
      case "--tts-model":
        args.ttsModel = value();
        break;
      case "--voice":
        args.voice = value();
        break;
      case "--system":
        args.system = value();
        break;
      case "--wav":
        args.wavPath = value();
        break;
      case "--max-turns":
        args.maxTurns = Number(value());
        break;
      default:
        throw new Error(`unknown argument ${flag}`);
    }
  }
  return args;
}

function parseTtsEngine(value: string): TtsEngine {
  if (value !== "sherpa" && value !== "cartesia") {
    throw new Error(`--tts must be "sherpa" or "cartesia", got "${value}"`);
  }
  return value;
}

function parseResponder(value: string): ResponderKind {
  if (value !== "cascade" && value !== "realtime") {
    throw new Error(`--responder must be "cascade" or "realtime", got "${value}"`);
  }
  return value;
}

await main();
