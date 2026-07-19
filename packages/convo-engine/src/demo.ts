import {
  CAPTURE_FRAME_MS,
  CAPTURE_SAMPLE_RATE,
  CartesiaTextToSpeech,
  createSegmentPlayer,
  DEFAULT_CARTESIA_MODEL,
  DEFAULT_CARTESIA_VOICE,
  DEFAULT_RULES,
  parseTtsModelId,
  readWavFrames,
  resolveDefaultMicDevice,
  SherpaEngine,
  SherpaTextToSpeech,
  startMicCapture,
  withRewrite,
  type MicCapture,
  type TextToSpeech,
  type TTSConfig,
  type WavAudio,
} from "@speakeasy/speech-io";
import { CerebrasChatModel, DEFAULT_CEREBRAS_MODEL } from "@speakeasy/llm";
import { ConversationLoop, type AudioSource } from "./loop.ts";
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

type TtsEngine = "sherpa" | "cartesia";

type DemoArgs = {
  llmModel: string;
  ttsEngine: TtsEngine;
  ttsModel: string | undefined;
  voice: string | undefined;
  system: string | undefined;
  wavPath: string | undefined;
  maxTurns: number | undefined;
  barge: boolean;
};

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (!process.env.CEREBRAS_API_KEY) {
    console.error(
      "CEREBRAS_API_KEY is not set. Add it to a gitignored .env and export it before running the convo demo.",
    );
    process.exitCode = 1;
    return;
  }

  const engine = new SherpaEngine();
  await engine.prepare();
  const stt = withRewrite(engine, { rules: DEFAULT_RULES, numbers: "off" });
  const llm = new CerebrasChatModel({ config: { model: args.llmModel } });

  const voice = buildTts(args);
  if (!voice) {
    process.exitCode = 1;
    return;
  }
  const { tts, ttsConfig, ttsLabel } = voice;

  const mic = args.wavPath
    ? new WavAudioSource(await readWavFrames(args.wavPath, CAPTURE_FRAME_MS))
    : new MicAudioSource((await resolveDefaultMicDevice()).spec);

  const loop = new ConversationLoop(
    { stt, llm, tts, mic, createSink: (sampleRate) => createSegmentPlayer(sampleRate) },
    {
      systemPrompt: args.system,
      sttConfig: {
        sampleRate: CAPTURE_SAMPLE_RATE,
        endpoint: { mode: "eager" },
      },
      ttsConfig,
      maxTurns: args.maxTurns,
      barge: args.barge,
      log: (line) => {
        clearPartial();
        console.log(line);
      },
      onState: (state) => {
        clearPartial();
        console.log(`[state] ${state}`);
      },
      onPartial: (text) => {
        if (process.stdout.isTTY) {
          process.stdout.write(`\r\x1b[K… ${text}`);
        }
      },
      onInterrupt: clearPartial,
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
    `speak-easy convo | stt=${engine.label} | llm=${args.llmModel} | tts=${ttsLabel} | source=${args.wavPath ? `wav ${args.wavPath}` : "microphone"}${args.barge ? " | barge-in on (use headphones)" : ""}`,
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

/**
 * Select the TTS engine from --tts. Returns undefined (after printing a clear
 * message) when the chosen engine's key is missing, so main() can fail fast.
 */
function buildTts(
  args: DemoArgs,
): { tts: TextToSpeech; ttsConfig: TTSConfig; ttsLabel: string } | undefined {
  if (args.ttsEngine === "cartesia") {
    if (!process.env.CARTESIA_API_KEY) {
      console.error(
        "CARTESIA_API_KEY is not set. Add it to a gitignored .env and export it before using --tts cartesia.",
      );
      return undefined;
    }
    const voice = args.voice ?? DEFAULT_CARTESIA_VOICE;
    return {
      tts: new CartesiaTextToSpeech(),
      ttsConfig: { model: args.ttsModel, voice },
      ttsLabel: `cartesia ${args.ttsModel ?? DEFAULT_CARTESIA_MODEL} voice=${voice}`,
    };
  }
  const model = parseTtsModelId(args.ttsModel ?? "kokoro-v0.19");
  return {
    tts: new SherpaTextToSpeech(),
    ttsConfig: {
      model,
      voice: args.voice !== undefined ? Number(args.voice) : undefined,
    },
    ttsLabel: `${model}${args.voice !== undefined ? ` voice=${args.voice}` : ""}`,
  };
}

function clearPartial(): void {
  if (process.stdout.isTTY) {
    process.stdout.write("\r\x1b[K");
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
    llmModel: DEFAULT_CEREBRAS_MODEL,
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
      case "--barge":
        args.barge = true;
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

await main();
