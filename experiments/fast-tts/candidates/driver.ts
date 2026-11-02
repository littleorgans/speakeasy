import { execFileSync, spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { arch, cpus, platform, release, totalmem } from "node:os";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";

/**
 * Shared driver for the TypeScript fast-tts candidates (kokoro-coreml,
 * supertonic3, piper-split): the CLI flags of render-baselines.ts, cold and warm
 * process shapes, the result JSON of CONTRACT.md, clips, and a client for native
 * workers that stream framed PCM over stdio. Each candidate supplies only an
 * `Adapter`; summarize.py validates and derives the metrics.
 */

export type Mode = "stream" | "full";
export type CohortText = { id: string; length: string; text: string };

/** Contract engine block: the required keys plus engine specific extras. */
export type EngineMeta = {
  id: string; label: string; runtime: string; provider: string; model: string;
  modelRevision: string; voice: string; sampleRate: number; [extra: string]: unknown;
};

/** One chunk of decoded samples as the caller receives it. */
export type AudioChunk = { samples: Float32Array; text: string | null };

/** A loaded engine. `stream` yields chunks as they become ready; `full` returns all audio at once. */
export type Adapter = {
  engine: EngineMeta;
  versions: Record<string, string>;
  loadDetail?: Record<string, unknown>;
  warmup(): Promise<void>;
  stream(text: string): AsyncIterable<AudioChunk>;
  full(text: string): Promise<Float32Array>;
  close(): Promise<void>;
};

export type Args = { engine: string; root: string; mode: Mode; temperature: "cold" | "warm" };

const HERE = import.meta.dirname;
const COHORT = join(HERE, "..", "cohort.json");
export const WARMUP_TEXT = "Warm up.";

/**
 * Parse the render-baselines.ts flags, load the engine (timed as `loadMs`), warm
 * up when warm, run the texts and write one result file. `open` gets the engine id
 * after it is checked against `engineIds`.
 */
export async function runCandidate(engineIds: readonly string[], open: (args: Args) => Promise<Adapter>): Promise<void> {
  const usage = "usage: node <candidate>/render.ts --engine <id> --temperature cold|warm --json <file> "
    + `[--root <models/tts>] [--mode stream|full] [--repeats N] [--text <id>]... [--clips <dir>] [--timed]\n  engines: ${engineIds.join(", ")}`;
  const { values } = parseArgs({
    options: {
      engine: { type: "string" }, temperature: { type: "string" }, json: { type: "string" },
      root: { type: "string", default: join(process.cwd(), "models", "tts") },
      mode: { type: "string", default: "stream" }, repeats: { type: "string", default: "1" },
      text: { type: "string", multiple: true }, clips: { type: "string" }, timed: { type: "boolean", default: false },
    },
  });
  const { engine, temperature, mode, json } = values;
  if (!engine || !engineIds.includes(engine) || (temperature !== "cold" && temperature !== "warm")
    || !json || (mode !== "stream" && mode !== "full")) {
    throw new Error(usage);
  }
  const repeats = temperature === "cold" ? 1 : Number(values.repeats);
  const cohort = (JSON.parse(await readFile(COHORT, "utf8")) as { texts: CohortText[] }).texts;
  const texts = (values.text ?? (temperature === "cold" ? ["status"] : cohort.map((item) => item.id)))
    .map((id) => cohort.find((item) => item.id === id) ?? fail(`unknown cohort id ${id}`));
  // Only the first request in a fresh process is cold, so a cold process runs one text.
  if (temperature === "cold" && texts.length !== 1) throw new Error("cold takes exactly one --text");

  const loadStart = performance.now();
  const adapter = await open({ engine, root: values.root, mode, temperature });
  const loadMs = performance.now() - loadStart;
  let warmupMs: number | null = null;
  if (temperature === "warm") {
    const start = performance.now();
    await adapter.warmup();
    warmupMs = performance.now() - start;
  }

  const rate = adapter.engine.sampleRate;
  const runs = [];
  for (const item of texts) {
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      const chunks: { index: number; readyMs: number; audioMs: number; text: string | null }[] = [];
      const pieces: Float32Array[] = [];
      const submit = performance.now();
      if (mode === "stream") {
        for await (const chunk of adapter.stream(item.text)) {
          chunks.push({ index: chunks.length, readyMs: round(performance.now() - submit), audioMs: round((chunk.samples.length / rate) * 1000), text: chunk.text });
          pieces.push(chunk.samples);
        }
      } else {
        const samples = await adapter.full(item.text);
        chunks.push({ index: 0, readyMs: round(performance.now() - submit), audioMs: round((samples.length / rate) * 1000), text: item.text });
        pieces.push(samples);
      }
      runs.push({ textId: item.id, mode, temperature, repeat, playbackStartMs: null, chunks });
      if (repeat === 0 && values.clips) {
        const suffix = mode === "full" ? "--full" : "";
        await writeWav(join(values.clips, `${item.id}--${engine}${suffix}.wav`), concat(pieces), rate);
      }
    }
  }
  await adapter.close();

  const document = {
    schema: "fast-tts.result.v1",
    engine: adapter.engine,
    environment: environment(adapter.versions),
    process: {
      temperature, timed: values.timed, repeats, startedAt: new Date().toISOString(),
      clock: "performance.now (monotonic)", playback: null, loadMs: round(loadMs),
      warmupMs: warmupMs === null ? null : round(warmupMs), loadDetail: adapter.loadDetail ?? {},
    },
    runs,
  };
  await mkdir(dirname(json), { recursive: true });
  await writeFile(json, `${JSON.stringify(document, null, 1)}\n`);
  console.log(`${engine} ${temperature} ${mode}: ${runs.length} runs, load ${loadMs.toFixed(0)} ms -> ${json}`);
}

/** Milliseconds to the microsecond, matching fasttts.py. */
export function round(ms: number): number {
  return Math.round(ms * 1000) / 1000;
}

/** Every item of an async iterable, in order (Array.fromAsync is past speak-easy's ES2023 lib). */
export async function collect<T>(items: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of items) out.push(item);
  return out;
}

export function concat(pieces: Float32Array[]): Float32Array {
  const out = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
  let offset = 0;
  for (const piece of pieces) {
    out.set(piece, offset);
    offset += piece.length;
  }
  return out;
}

/** Mono 16 bit PCM WAV, as fasttts.write_wav; no native addon in the measured process. */
export async function writeWav(path: string, samples: Float32Array, sampleRate: number): Promise<void> {
  const data = Buffer.alloc(44 + samples.length * 2);
  data.write("RIFF", 0);
  data.writeUInt32LE(36 + samples.length * 2, 4);
  data.write("WAVEfmt ", 8);
  data.writeUInt32LE(16, 16);
  data.writeUInt16LE(1, 20);
  data.writeUInt16LE(1, 22);
  data.writeUInt32LE(sampleRate, 24);
  data.writeUInt32LE(sampleRate * 2, 28);
  data.writeUInt16LE(2, 32);
  data.writeUInt16LE(16, 34);
  data.write("data", 36);
  data.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((sample, i) => data.writeInt16LE(Math.trunc(Math.max(-1, Math.min(1, sample)) * 32767), 44 + i * 2));
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, data);
}

function environment(versions: Record<string, string>) {
  const command = (file: string, args: string[]) => {
    try {
      return execFileSync(file, args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
    } catch {
      return null;
    }
  };
  const mac = platform() === "darwin" ? command("sw_vers", ["-productVersion"]) : null;
  return {
    machine: cpus()[0]?.model ?? "unknown", arch: arch(), cpuCount: cpus().length,
    memoryGiB: Math.round(totalmem() / 2 ** 30), os: mac ? `macOS ${mac}` : `${platform()} ${release()}`,
    versions: { node: process.versions.node, ...versions },
    harnessCommit: command("git", ["-C", HERE, "rev-parse", "--short=12", "HEAD"]),
  };
}

export function fail(message: string): never {
  throw new Error(message);
}

/**
 * Frame kinds of the native worker protocol. Every frame is a little endian
 * int32 kind, an int32 count, then the payload: `count` float32 samples for
 * AUDIO, `count` bytes of UTF-8 JSON otherwise. Requests are lines on stdin:
 * `S<TAB>part<US>part...` synthesizes the parts in order (one AUDIO frame or
 * more per part, then END); `Q` quits.
 */
export const FRAME = { READY: 1, AUDIO: 2, END: 3, ERROR: 4 } as const;
const PART_SEPARATOR = "\u001f";

export type Frame = { kind: number; samples?: Float32Array; info?: Record<string, unknown> };

/** A resident native worker: one request at a time, frames read as they arrive. */
export class NativeWorker {
  readonly ready: Record<string, unknown>;
  #child: ChildProcessWithoutNullStreams;
  #frames: AsyncIterator<Frame>;

  private constructor(child: ChildProcessWithoutNullStreams, frames: AsyncIterator<Frame>, ready: Record<string, unknown>) {
    this.#child = child;
    this.#frames = frames;
    this.ready = ready;
  }

  /** Spawn the worker and wait for its READY frame (the load is inside the caller's loadMs). */
  static async start(file: string, args: string[], env: Record<string, string> = {}): Promise<NativeWorker> {
    const child = spawn(file, args, { stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, ...env } });
    child.stderr.on("data", (data: Buffer) => process.stderr.write(data));
    const frames = readFrames(child.stdout)[Symbol.asyncIterator]();
    const first = await frames.next();
    if (first.done || first.value.kind !== FRAME.READY) throw new Error(`${file}: no READY frame (${JSON.stringify(first.value?.info)})`);
    return new NativeWorker(child, frames, first.value.info ?? {});
  }

  /** Submit the parts and yield each AUDIO frame's samples, then the END info as the return value. */
  async *synth(parts: string[]): AsyncGenerator<Float32Array, Record<string, unknown>> {
    if (parts.some((part) => /[\t\n\u001f]/.test(part))) throw new Error("part contains a protocol separator");
    this.#child.stdin.write(`S\t${parts.join(PART_SEPARATOR)}\n`);
    for (;;) {
      const next = await this.#frames.next();
      if (next.done) throw new Error("worker exited mid request");
      const frame = next.value;
      if (frame.kind === FRAME.AUDIO) yield frame.samples!;
      else if (frame.kind === FRAME.END) return frame.info ?? {};
      else throw new Error(`worker error: ${JSON.stringify(frame.info)}`);
    }
  }

  async close(): Promise<void> {
    const exited = new Promise((resolve) => this.#child.once("exit", resolve));
    this.#child.stdin.end("Q\n");
    await exited;
  }
}

async function* readFrames(stream: NodeJS.ReadableStream): AsyncGenerator<Frame> {
  let buffer: Buffer = Buffer.alloc(0);
  for await (const data of stream as AsyncIterable<Buffer>) {
    buffer = buffer.length ? Buffer.concat([buffer, data]) : data;
    for (;;) {
      if (buffer.length < 8) break;
      const kind = buffer.readInt32LE(0);
      const count = buffer.readInt32LE(4);
      const bytes = kind === FRAME.AUDIO ? count * 4 : count;
      if (buffer.length < 8 + bytes) break;
      const payload = buffer.subarray(8, 8 + bytes);
      buffer = buffer.subarray(8 + bytes);
      if (kind === FRAME.AUDIO) {
        // Copy: the payload may be unaligned and its buffer is reused.
        const samples = new Float32Array(count);
        Buffer.from(samples.buffer).set(payload);
        yield { kind, samples };
      } else {
        yield { kind, info: count ? (JSON.parse(payload.toString("utf8")) as Record<string, unknown>) : {} };
      }
    }
  }
}
