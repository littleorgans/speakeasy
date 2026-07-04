import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { formatMs } from "../bench/format.ts";
import { parseTtsModelId, TTS_MODELS, type TtsModelId } from "./models.ts";
import {
  buildTimingReport,
  streamSpeech,
  type SpeechSegment,
} from "./stream.ts";
import { BUILD_STATUS_PARAGRAPH, TTS_RESULTS_DIR } from "./sweep.ts";
import { TtsSynth, writeWav } from "./synth.ts";

/** Silence (ms) written to the sink on open() so CoreAudio opens off the request path. */
const SINK_PRIMER_MS = 150;
/** Measured ffplay CoreAudio open latency; the cold-sink penalty in the honest TTFA. */
const DEVICE_OPEN_EST_MS = 500;

/**
 * Audible demo of sentence-pipelined streaming: pulls segments from
 * streamSpeech and feeds each into a single long-lived audio sink as it
 * arrives, then prints the timing report proving synthesis stayed ahead of
 * playback. Playback and the pull loop overlap on purpose; that overlap IS the
 * streaming.
 *
 * Playback uses one persistent ffplay process reading raw f32le PCM from its
 * stdin: segment i plays while segment i+1 synthesizes, and because RTF < 1 the
 * pipe never underruns, so sentences run together with a single controlled gap
 * (stream.ts trims the model's ragged end padding). This replaces the old
 * per-segment afplay chain, whose ~1.5s process-spawn latency per sentence was
 * the dominant inter-sentence pause. afplay is kept as a fallback when ffplay
 * is absent; that path still has the spawn gap.
 *
 * TTFA model: production runs a resident, warmed model and an already-open audio
 * device, so the honest time-to-first-audio is just the first chunk's synth. The
 * demo mirrors that by default: it loads + warms the model and pre-opens the
 * sink during a one-time STARTUP phase, then measures TTFA from the request. Run
 * --cold to skip warmup and sink pre-open and see the fresh-process penalty
 * (model load off the report, device-open added). stream.ts carves a short first
 * chunk (planSegments) so that first synth is small.
 */

const USAGE = [
  'usage: node src/tts/stream-demo.ts [--text "..."] [--model <id>] [--speed <rate>] [--no-play] [--cold]',
  `       --model ids: ${Object.keys(TTS_MODELS).join(", ")} (default piper-amy)`,
  "       --no-play: skip playback, just write wavs and print timings",
  "       --cold: skip model warmup + sink pre-open (fresh-process TTFA)",
].join("\n");

type DemoOptions = {
  text: string;
  model: TtsModelId;
  speed: number;
  play: boolean;
  warm: boolean;
};

const options = parseArgs(process.argv.slice(2));
await runDemo(options);

async function runDemo(options: DemoOptions): Promise<void> {
  console.log(
    `stream-demo model=${options.model} speed=${options.speed} play=${options.play} warm=${options.warm}`,
  );
  console.log(`text=${JSON.stringify(options.text)}`);

  // Startup: one-time cost a resident production process pays once. Load the
  // model, and when warm, pre-open the sink (device opens now) and warm the
  // onnxruntime session so the first real synth runs warm.
  const startup0 = performance.now();
  const synth = await TtsSynth.create(options.model);
  const loadMs = performance.now() - startup0;
  const player = options.play ? createPlayer(synth.sampleRate) : undefined;
  let warmupMs = 0;
  const preOpened = Boolean(player) && options.warm;
  if (options.warm) {
    player?.open(); // device-open overlaps the warmup synth below
    warmupMs = await synth.warmup();
  }
  const startupMs = performance.now() - startup0;
  console.log(
    `sink=${player?.kind ?? "none"} startup=${formatMs(startupMs)} (load=${formatMs(loadMs)} warmup=${options.warm ? formatMs(warmupMs) : "skipped"} sink-preopen=${preOpened ? "yes" : "no"})`,
  );

  // Request: production TTFA is measured from here (model resident + warm).
  const reqStart = performance.now();
  const segments: SpeechSegment[] = [];
  let firstChunkMs: number | undefined;
  for await (const segment of streamSpeech(options.text, {
    synth,
    speed: options.speed,
  })) {
    segments.push(segment);
    const wavPath = join(TTS_RESULTS_DIR, `stream-${segment.index}.wav`);
    await writeWav(wavPath, segment);
    player?.write(segment, wavPath); // lazy-opens the sink here if not pre-opened
    firstChunkMs ??= player ? performance.now() - reqStart : segment.readyAtMs;
    console.log(
      `segment=${segment.index} ready=${formatMs(segment.readyAtMs)} synth=${formatMs(segment.synthMs)} audio=${formatMs(segment.audioDurationMs)} text=${JSON.stringify(segment.sentence)}`,
    );
  }

  await player?.end();

  // Honest audible first sound: the first chunk reaching an OPEN device. When
  // the sink was pre-opened the device is already hot, so audible == first
  // chunk; a cold sink opens on that first write, adding ~DEVICE_OPEN_EST_MS.
  const deviceOpenPenalty = player && !preOpened ? DEVICE_OPEN_EST_MS : 0;
  const ttfaAudibleMs = (firstChunkMs ?? 0) + deviceOpenPenalty;
  const report = buildTimingReport(segments);
  console.log(
    `ttfa-audible=${formatMs(ttfaAudibleMs)} (first-chunk=${formatMs(firstChunkMs ?? 0)}${deviceOpenPenalty ? ` + device-open~${formatMs(deviceOpenPenalty)}` : preOpened ? " sink pre-opened" : ""}) ahead-of-playback=${report.aheadOfPlayback ? "yes" : "no"}`,
  );
  for (const row of report.segments) {
    console.log(
      `segment=${row.index} ready=${formatMs(row.readyAtMs)} play-slot=${formatMs(row.playStartMs)} margin=${formatMs(row.marginMs)}`,
    );
  }
}

/** A continuous audio sink fed segment-by-segment as synthesis produces them. */
type SegmentPlayer = {
  kind: string;
  /** Pre-open the device (off the request path) so it is hot by first write. */
  open(): void;
  write(segment: SpeechSegment, wavPath: string): void;
  end(): Promise<void>;
};

/** ffplay when available (gapless PCM stdin), else the afplay chain fallback. */
function createPlayer(sampleRate: number): SegmentPlayer {
  return hasFfplay() ? ffplayPlayer(sampleRate) : afplayPlayer();
}

function hasFfplay(): boolean {
  return spawnSync("ffplay", ["-version"], { stdio: "ignore" }).status === 0;
}

/**
 * One ffplay process for the whole reply: each segment's Float32 PCM is written
 * to its stdin as f32le, so playback is gapless across sentences. open() spawns
 * it and writes a short silence primer, forcing CoreAudio to open before the
 * first real chunk arrives; write() lazy-spawns too, for the --cold path.
 */
function ffplayPlayer(sampleRate: number): SegmentPlayer {
  let child: ChildProcess | undefined;
  let done: Promise<void> = Promise.resolve();
  const ensure = (): ChildProcess => {
    if (child) {
      return child;
    }
    child = spawn(
      "ffplay",
      // ffplay is a playback tool: channels come from -ch_layout, not the
      // ffmpeg output flag -ac (which it rejects). -nostats/-nodisp keep it
      // headless; -autoexit quits at stdin EOF once the queue is drained.
      // prettier-ignore
      [
        "-hide_banner", "-loglevel", "error", "-nostats", "-nodisp",
        "-autoexit", "-f", "f32le", "-ar", String(sampleRate),
        "-ch_layout", "mono", "-i", "pipe:0",
      ],
      { stdio: ["pipe", "ignore", "ignore"] },
    );
    child.stdin?.on("error", () => {}); // report via exit, not an EPIPE throw
    done = onExit(child, "ffplay");
    return child;
  };
  return {
    kind: "ffplay",
    open() {
      const frames = Math.round((sampleRate * SINK_PRIMER_MS) / 1000);
      ensure().stdin?.write(Buffer.alloc(frames * Float32Array.BYTES_PER_ELEMENT));
    },
    write(segment) {
      const { samples } = segment;
      ensure().stdin?.write(
        Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength),
      );
    },
    async end() {
      child?.stdin?.end();
      await done;
    },
  };
}

/** Fallback: chain one afplay per segment. Retains the per-spawn gap. */
function afplayPlayer(): SegmentPlayer {
  let queue: Promise<void> = Promise.resolve();
  return {
    kind: "afplay (fallback; per-segment spawn gap, no pre-open)",
    open() {}, // afplay plays whole wav files; nothing to pre-open
    write(_segment, wavPath) {
      queue = queue.then(() =>
        onExit(spawn("afplay", [wavPath], { stdio: "ignore" }), "afplay"),
      );
    },
    end: () => queue,
  };
}

/** Resolve when the child exits 0; reject on spawn error or nonzero exit. */
function onExit(child: ChildProcess, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("exit", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`${name} exited with code ${code ?? "unknown"}`)),
    );
  });
}

function parseArgs(args: string[]): DemoOptions {
  const options: DemoOptions = {
    text: BUILD_STATUS_PARAGRAPH,
    model: "piper-amy",
    speed: 1,
    play: true,
    warm: true,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--text") {
      options.text = requireValue(args, index);
      index += 1;
    } else if (arg === "--model") {
      options.model = parseTtsModelId(requireValue(args, index));
      index += 1;
    } else if (arg === "--speed") {
      options.speed = Number(requireValue(args, index));
      if (!Number.isFinite(options.speed) || options.speed <= 0) {
        throw new Error("--speed must be a positive number");
      }
      index += 1;
    } else if (arg === "--no-play") {
      options.play = false;
    } else if (arg === "--cold") {
      options.warm = false;
    } else {
      throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }

  return options;
}

function requireValue(args: string[], index: number): string {
  const value = args[index + 1];
  if (value === undefined) {
    throw new Error(`Missing value for ${args[index]}`);
  }
  return value;
}
