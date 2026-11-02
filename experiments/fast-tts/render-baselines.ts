import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { cpus, platform, release, totalmem, arch } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { performance } from "node:perf_hooks";
import { getLocalOutputReadiness, resolveLocalOutput, type LocalOutputModelId } from "../../packages/speech-io/src/tts/catalogue.ts";
import { LOCAL_TTS_MODELS } from "../../packages/speech-io/src/tts/models.ts";
import { SherpaTextToSpeech } from "../../packages/speech-io/src/tts/sherpa-tts.ts";
import { TtsSynth, writeWav } from "../../packages/speech-io/src/tts/synth.ts";

/**
 * fast-tts baseline engines through speak-easy's own path: TtsSynth (sherpa, cpu
 * provider, Piper Plus frontend) behind SherpaTextToSpeech, whose speak() runs the
 * shared sentence pipeline. No onProgress anywhere. Emits the shared result JSON
 * (CONTRACT.md) for one engine per process; summarize.py derives the metrics.
 *
 * cold: load, then one run of one --text (default status) with no warmup.
 * warm: load, one discarded warmup synth, then --repeats passes over the texts.
 * Adapted from experiments/audio8-tts/render-baselines.ts (audio8-tts thread).
 */

// threads unset: TtsSynth's product default (DEFAULT_NUM_THREADS in synth.ts).
const ENGINES = {
  "piper-kristin": { label: "Piper Kristin medium", model: "piper-kristin-medium", voice: "kristin" },
  "piper-kristin-t4": { label: "Piper Kristin medium", model: "piper-kristin-medium", voice: "kristin", threads: 4 },
  "kokoro-heart": { label: "Kokoro v1.0 · af_heart", model: "kokoro-v1.0", voice: "af_heart" },
  "kokoro-heart-t4": { label: "Kokoro v1.0 · af_heart", model: "kokoro-v1.0", voice: "af_heart", threads: 4 },
  "kokoro-michael": { label: "Kokoro v1.0 · am_michael", model: "kokoro-v1.0", voice: "am_michael" },
} as const satisfies Record<string, { label: string; model: LocalOutputModelId; voice: string; threads?: number }>;
type EngineId = keyof typeof ENGINES;

type CohortText = { id: string; length: string; text: string };
type Chunk = { index: number; readyMs: number; audioMs: number; text: string };
type Mode = "stream" | "full";

const USAGE = "usage: node experiments/fast-tts/render-baselines.ts --engine <id> --temperature cold|warm --json <file> "
  + `[--root <models/tts>] [--mode stream|full] [--repeats N] [--text <id>]... [--clips <dir>] [--timed]\n  engines: ${Object.keys(ENGINES).join(", ")}`;

const { values } = parseArgs({
  options: {
    engine: { type: "string" },
    temperature: { type: "string" },
    json: { type: "string" },
    root: { type: "string", default: join(process.cwd(), "models", "tts") },
    mode: { type: "string", default: "stream" },
    repeats: { type: "string", default: "1" },
    text: { type: "string", multiple: true },
    clips: { type: "string" },
    timed: { type: "boolean", default: false },
  },
});
const engineId = values.engine as EngineId;
const temperature = values.temperature;
const mode = values.mode as Mode;
if (!Object.hasOwn(ENGINES, engineId ?? "") || (temperature !== "cold" && temperature !== "warm")
  || !values.json || (mode !== "stream" && mode !== "full")) {
  throw new Error(USAGE);
}
const engine: { label: string; model: LocalOutputModelId; voice: string; threads?: number } = ENGINES[engineId];
const repeats = temperature === "cold" ? 1 : Number(values.repeats);
const cohort = (JSON.parse(await readFile(join(import.meta.dirname, "cohort.json"), "utf8")) as { texts: CohortText[] }).texts;
const texts = (values.text ?? (temperature === "cold" ? ["status"] : cohort.map((item) => item.id)))
  .map((id) => cohort.find((item) => item.id === id) ?? missing(id));
// Only the first request in a fresh process is cold, so a cold process runs one text.
if (temperature === "cold" && texts.length !== 1) throw new Error("cold takes exactly one --text");

// Integrity check off the clock; TtsSynth.create verifies the same digests again inside loadMs.
const readiness = await getLocalOutputReadiness(engine.model, values.root);
if (readiness.state !== "ready") throw new Error(`${engine.model} not ready: ${readiness.reason}`);
const { voice, speed } = resolveLocalOutput({ schema: 1, provider: { kind: "local" }, model: engine.model, voice: engine.voice, speed: 1 });

const loadStart = performance.now();
const synth = await TtsSynth.create(engine.model, engine.threads, { root: values.root, allowDownload: false });
const session = await new SherpaTextToSpeech(async () => synth).open({ model: engine.model, voice: voice.speakerId, speed });
const loadMs = performance.now() - loadStart;
const warmupMs = temperature === "warm" ? await synth.warmup() : null;

const runs = [];
for (const item of texts) {
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const { chunks, samples } = mode === "stream" ? await runStream(item.text) : await runFull(item.text);
    runs.push({ textId: item.id, mode, temperature, repeat, playbackStartMs: null, chunks });
    if (repeat === 0 && values.clips) {
      const suffix = mode === "full" ? "--full" : "";
      await writeWav(join(values.clips, `${item.id}--${engineId}${suffix}.wav`), { samples, sampleRate: synth.sampleRate });
    }
  }
}
await session.close();

const document = {
  schema: "fast-tts.result.v1",
  engine: {
    id: engineId,
    label: `${engine.label} · ${synth.numThreads} threads${engine.threads === undefined ? ", product default" : ""}`,
    runtime: `sherpa-onnx-node ${packageVersion("sherpa-onnx-node")}`, provider: "cpu",
    model: engine.model, modelRevision: `sha256:${LOCAL_TTS_MODELS[engine.model].files[0].sha256.slice(0, 12)}`,
    voice: engine.voice, sampleRate: synth.sampleRate, threads: synth.numThreads,
    streaming: mode === "stream" ? "speak-easy sentence pipeline (planSegments, depth 1, trimSilence)" : "one generateAsync call",
  },
  environment: environment(),
  process: {
    temperature, timed: values.timed, repeats, startedAt: new Date().toISOString(),
    clock: "performance.now (monotonic)", playback: null, loadMs: round(loadMs), warmupMs: warmupMs === null ? null : round(warmupMs),
    loadDetail: { nativeLoadMs: round(synth.loadMs), note: "loadMs is TtsSynth.create (digest verify, native load, frontend) plus session open" },
  },
  runs,
};
await mkdir(join(values.json, ".."), { recursive: true });
await writeFile(values.json, `${JSON.stringify(document, null, 1)}\n`);
console.log(`${engineId} ${temperature} ${mode}: ${runs.length} runs, load ${loadMs.toFixed(0)} ms -> ${values.json}`);

/** Text submit to each chunk reaching the caller, through the shared sentence pipeline. */
async function runStream(text: string) {
  const chunks: Chunk[] = [];
  const pieces: Float32Array[] = [];
  const submit = performance.now();
  for await (const segment of session.speak(text)) {
    chunks.push({ index: segment.index, readyMs: round(performance.now() - submit), audioMs: round(segment.audioDurationMs), text: segment.sentence });
    pieces.push(segment.samples);
  }
  return { chunks, samples: concat(pieces) };
}

/** Whole text in one synth call: one chunk at the end. */
async function runFull(text: string) {
  const submit = performance.now();
  const result = await synth.synth({ text, sid: voice.speakerId, speed });
  const readyMs = round(performance.now() - submit);
  return { chunks: [{ index: 0, readyMs, audioMs: round(result.audioDurationMs), text }], samples: result.samples };
}

/** Milliseconds to the microsecond, matching fasttts.py. */
function round(ms: number): number {
  return Math.round(ms * 1000) / 1000;
}

function concat(pieces: Float32Array[]): Float32Array {
  const out = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
  let offset = 0;
  for (const piece of pieces) {
    out.set(piece, offset);
    offset += piece.length;
  }
  return out;
}

function environment() {
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
    versions: { node: process.versions.node, "sherpa-onnx-node": packageVersion("sherpa-onnx-node") },
    harnessCommit: command("git", ["-C", import.meta.dirname, "rev-parse", "--short=12", "HEAD"]),
  };
}

function packageVersion(name: string): string {
  const path = join(import.meta.dirname, "../../packages/speech-io/node_modules", name, "package.json");
  return (JSON.parse(readFileSync(path, "utf8")) as { version: string }).version;
}

function missing(id: string): never {
  throw new Error(`unknown cohort id ${id}`);
}
