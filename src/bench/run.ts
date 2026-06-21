import { mkdir, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import type { EndpointConfig, STTSession, VoiceToText } from "../contract.ts";
import { StubEngine } from "../engines/stub.ts";
import { MoonshineEngine } from "../engines/moonshine.ts";
import { SherpaEngine } from "../engines/sherpa.ts";
import { readWavFrames, type WavAudio } from "./wav.ts";

const DEFAULT_RUNS = 5;
const DEFAULT_ENGINE = "stub";
const DEFAULT_FRAME_MS = 20;
const PASS_THRESHOLD_MS = 200;
const SOFT_THRESHOLD_MS = 300;
const SESSION_TIMEOUT_MS = 30_000;
const EXPECTED_JFK_TRANSCRIPT = "and so my fellow americans";
const RMS_WINDOW_MS = 20;
const RMS_HANGOVER_MS = 650;
const RMS_PEAK_RATIO = 0.04;
const RMS_NOISE_MULTIPLIER = 8;
const RMS_MIN_THRESHOLD = 0.016;
const RMS_TAIL_NOISE_MS = 500;
const SHERPA_SWEEP_PATH = "results/sherpa-sweep.txt";
const PARTIAL_ROOTCAUSE = "model-right-context";
const SHERPA_SWEEP: Required<EndpointConfig>[] = [
  { mode: "eager", minTrailingSilenceMs: 80, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 120, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 160, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 200, minUtteranceMs: 20_000 },
  { mode: "eager", minTrailingSilenceMs: 300, minUtteranceMs: 20_000 },
];

type EngineName = "stub" | "moonshine" | "sherpa";

type BenchEngine = VoiceToText & {
  label?: string;
  prepare?: () => Promise<void>;
};

type CliOptions = {
  engine: EngineName;
  wav?: string;
  runs: number;
  frameMs: number;
};

type SpeechProfile = {
  endMs: number;
  frameIndex: number;
  offsetWithinFrameMs: number;
  threshold: number;
  windowMs: number;
  hangoverMs: number;
};

type FinalObservation = {
  endpointAt: number;
  finalAt: number;
  text: string;
};

type RunResult = {
  run: number;
  firstPartialMs?: number;
  endpointToFinalMs: number;
  speechEndToFinalMs: number;
  endpointDelayMs: number;
  wallMs: number;
  finalText: string;
  finalizedAfterSpeechEnd: boolean;
  textCorrect: boolean;
};

type Summary = {
  engineLabel: string;
  endpoint?: Required<EndpointConfig>;
  results: RunResult[];
  endpointToFinalMedian: number;
  speechEndToFinalMedian: number;
  firstPartialMedian?: number;
  firstPartialColdMs?: number;
  firstPartialWarmMedian?: number;
  textCorrect: boolean;
  passFail: "PASS" | "FAIL";
  ok: boolean;
};

const options = parseArgs(process.argv.slice(2));
if (!options.wav) {
  throw new Error("Missing required --wav <path> argument");
}

const audio = await readWavFrames(options.wav, options.frameMs);
const speech = detectSpeechProfile(audio, options.frameMs);

console.log("speak-easy latency benchmark");
console.log(`wav=${options.wav}`);
console.log(
  `audio=${audio.durationMs.toFixed(0)}ms speech-end=${speech.endMs.toFixed(0)}ms trailing-silence=${(audio.durationMs - speech.endMs).toFixed(0)}ms`,
);
console.log(
  `speech-end-detector=rms-window window=${speech.windowMs}ms hangover=${speech.hangoverMs}ms threshold=${speech.threshold.toFixed(4)}`,
);
console.log(
  `expected=${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} comparison=case/punct-insensitive-exact`,
);
console.log(
  `format=${audio.sampleRate}Hz mono ${audio.bitsPerSample}-bit cadence=${options.frameMs}ms/frame frames=${audio.frames.length} runs=${options.runs}`,
);
console.log(
  options.engine === "sherpa"
    ? "feeding frames in real time; sherpa endpoint configs are swept for eager finalization"
    : "feeding frames in real time; endpoint is supplied by the engine or end-of-input",
);

const summaries = options.engine === "sherpa"
  ? await runSherpaSweep(options, audio, speech)
  : [await runEngineSummary(options, audio, speech, undefined)];

for (const summary of summaries) {
  printSummary(summary);
}

const selected = selectBestSummary(summaries) ?? selectLowestMeasuredSummary(summaries);
if (summaries.length > 1 && selected) {
  console.log(
    `selected=${formatEndpoint(selected.endpoint)} endpoint->final=${formatMs(selected.speechEndToFinalMedian)} event->final=${formatMs(selected.endpointToFinalMedian)} text-correct=${formatBoolean(selected.textCorrect)} first-partial-warm=${formatOptionalMs(selected.firstPartialWarmMedian)} partial-rootcause=${PARTIAL_ROOTCAUSE}`,
  );
}

if (options.engine === "sherpa") {
  await writeSherpaSweep(SHERPA_SWEEP_PATH, summaries, selected, speech);
  console.log(`file=${SHERPA_SWEEP_PATH}`);
}

async function writeSherpaSweep(
  path: string,
  summaries: Summary[],
  selected: Summary | undefined,
  speechProfile: SpeechProfile,
): Promise<void> {
  await mkdir("results", { recursive: true });
  const knee = selectBestSummary(summaries);
  const rows = summaries.map((summary) =>
    [
      formatEndpoint(summary.endpoint),
      formatMs(summary.speechEndToFinalMedian),
      formatBoolean(summary.textCorrect),
      JSON.stringify(mostCommonFinalText(summary.results)),
    ].join(" | "),
  );

  const content = [
    "# sherpa endpoint sweep",
    "",
    `engine: ${summaries[0]?.engineLabel ?? "unknown"}`,
    `speech-end-reference: rms-window ${speechProfile.windowMs}ms + ${speechProfile.hangoverMs}ms hangover, threshold=${speechProfile.threshold.toFixed(4)}, end=${speechProfile.endMs.toFixed(1)}ms`,
    `expected: ${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} (case/punct-insensitive exact)`,
    `knee: ${knee ? formatEndpoint(knee.endpoint) : "none"}`,
    `selected: ${selected ? formatEndpoint(selected.endpoint) : "none"}`,
    `partial-rootcause: ${PARTIAL_ROOTCAUSE}; decode is called whenever sherpa reports readiness per pushed frame, but this model emits no non-empty result until its chunk/right-context is satisfied`,
    "",
    "config | perceived endpoint->final median | text-correct | finalText",
    "--- | ---: | :---: | ---",
    ...rows,
    "",
  ].join("\n");

  await writeFile(path, content);
}

function mostCommonFinalText(results: RunResult[]): string {
  const counts = new Map<string, number>();
  for (const result of results) {
    counts.set(result.finalText, (counts.get(result.finalText) ?? 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? "";
}

async function runSherpaSweep(
  cli: CliOptions,
  wav: WavAudio,
  speechProfile: SpeechProfile,
): Promise<Summary[]> {
  const summaries: Summary[] = [];
  for (const endpoint of SHERPA_SWEEP) {
    summaries.push(await runEngineSummary(cli, wav, speechProfile, endpoint));
  }
  return summaries;
}

async function runEngineSummary(
  cli: CliOptions,
  wav: WavAudio,
  speechProfile: SpeechProfile,
  endpoint: Required<EndpointConfig> | undefined,
): Promise<Summary> {
  const engine = createEngine(cli.engine);
  await engine.prepare?.();
  const engineLabel = engine.label ?? cli.engine;
  const results: RunResult[] = [];

  for (let run = 1; run <= cli.runs; run += 1) {
    results.push(
      await runOnce({
        run,
        engine,
        endpoint,
        audio: wav,
        speech: speechProfile,
        frameMs: cli.frameMs,
      }),
    );
  }

  const endpointToFinalMedian = median(
    results.map((result) => result.endpointToFinalMs),
  );
  const speechEndToFinalMedian = median(
    results.map((result) => result.speechEndToFinalMs),
  );
  const firstPartialMedian = medianOptional(
    results.map((result) => result.firstPartialMs),
  );
  const firstPartialColdMs = results[0]?.firstPartialMs;
  const firstPartialWarmMedian = medianOptional(
    results.slice(1).map((result) => result.firstPartialMs),
  );
  const textCorrect = results.every((result) => result.textCorrect);
  const ok = results.every(
    (result) => result.finalizedAfterSpeechEnd && result.textCorrect,
  );

  return {
    engineLabel,
    endpoint,
    results,
    endpointToFinalMedian,
    speechEndToFinalMedian,
    firstPartialMedian,
    firstPartialColdMs,
    firstPartialWarmMedian,
    textCorrect,
    passFail:
      ok && speechEndToFinalMedian < PASS_THRESHOLD_MS ? "PASS" : "FAIL",
    ok,
  };
}

async function runOnce(input: {
  run: number;
  engine: VoiceToText;
  endpoint?: EndpointConfig;
  audio: WavAudio;
  speech: SpeechProfile;
  frameMs: number;
}): Promise<RunResult> {
  const session = await input.engine.open({
    sampleRate: input.audio.sampleRate,
    endpoint: input.endpoint,
  });
  const state = attachSessionObservers(session);
  const start = performance.now();
  let speechEndAt: number | undefined;

  for (const [frameIndex, frame] of input.audio.frames.entries()) {
    const frameSentAt = performance.now();
    session.pushAudio(frame);
    if (frameIndex === input.speech.frameIndex) {
      speechEndAt = frameSentAt + input.speech.offsetWithinFrameMs;
    }
    await delay(input.frameMs);
  }

  await withTimeout(session.end(), SESSION_TIMEOUT_MS);
  state.throwIfError();

  const wallMs = performance.now() - start;
  const observedSpeechEndAt = speechEndAt ?? start + input.speech.endMs;
  const final = chooseFinal(state.finals, observedSpeechEndAt) ?? {
    endpointAt: performance.now(),
    finalAt: performance.now(),
    text: "",
  };
  const speechEndToFinalMs = final.finalAt - observedSpeechEndAt;
  const textCorrect = isExpectedTranscript(final.text);

  return {
    run: input.run,
    firstPartialMs: state.firstPartialAt
      ? state.firstPartialAt - start
      : undefined,
    endpointToFinalMs: final.finalAt - final.endpointAt,
    speechEndToFinalMs,
    endpointDelayMs: final.endpointAt - observedSpeechEndAt,
    wallMs,
    finalText: final.text,
    finalizedAfterSpeechEnd: speechEndToFinalMs >= 0,
    textCorrect,
  };
}

function attachSessionObservers(session: STTSession): {
  firstPartialAt?: number;
  finals: FinalObservation[];
  throwIfError: () => void;
} {
  const endpointEvents: number[] = [];
  const finalTexts: string[] = [];
  const finals: FinalObservation[] = [];
  const state: { firstPartialAt?: number; error?: unknown } = {};

  session.on("partial", () => {
    state.firstPartialAt ??= performance.now();
  });
  session.on("endpoint", () => {
    endpointEvents.push(performance.now());
  });
  session.on("final", ({ text }: { text: string }) => {
    const finalAt = performance.now();
    const trimmed = text.trim();
    if (trimmed) {
      finalTexts.push(trimmed);
    }
    finals.push({
      endpointAt: lastEndpointBefore(endpointEvents, finalAt) ?? finalAt,
      finalAt,
      text: finalTexts.join(" ").trim(),
    });
  });
  session.on("error", ({ err }: { err: unknown }) => {
    state.error = err;
  });

  return {
    get firstPartialAt() {
      return state.firstPartialAt;
    },
    finals,
    throwIfError() {
      if (state.error) {
        throw state.error;
      }
    },
  };
}

function chooseFinal(
  finals: FinalObservation[],
  speechEndAt: number,
): FinalObservation | undefined {
  return (
    finals.find((final) => final.finalAt >= speechEndAt && final.text) ??
    finals.findLast((final) => final.text)
  );
}

function lastEndpointBefore(
  endpoints: number[],
  finalAt: number,
): number | undefined {
  return endpoints.findLast((endpointAt) => endpointAt <= finalAt);
}

function createEngine(name: EngineName): BenchEngine {
  if (name === "stub") {
    return new StubEngine();
  }
  if (name === "moonshine") {
    return new MoonshineEngine();
  }
  if (name === "sherpa") {
    return new SherpaEngine();
  }
  throw new Error(`Unsupported engine ${name}`);
}

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    engine: DEFAULT_ENGINE,
    runs: DEFAULT_RUNS,
    frameMs: DEFAULT_FRAME_MS,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--engine") {
      options.engine = parseEngine(requireValue(args, index));
      index += 1;
    } else if (arg === "--wav") {
      options.wav = requireValue(args, index);
      index += 1;
    } else if (arg === "--runs") {
      options.runs = parsePositiveInteger("--runs", requireValue(args, index));
      index += 1;
    } else if (arg === "--frame-ms") {
      options.frameMs = parsePositiveInteger(
        "--frame-ms",
        requireValue(args, index),
      );
      index += 1;
    } else {
      throw new Error(`Unknown argument ${arg}`);
    }
  }

  if (options.runs < DEFAULT_RUNS) {
    throw new Error(`--runs must be at least ${DEFAULT_RUNS}`);
  }

  return options;
}

function requireValue(args: string[], index: number): string {
  const value = args[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${args[index]}`);
  }
  return value;
}

function parseEngine(value: string): EngineName {
  if (value === "stub" || value === "moonshine" || value === "sherpa") {
    return value;
  }
  throw new Error(`--engine must be stub, moonshine, or sherpa; received ${value}`);
}

function parsePositiveInteger(flag: string, value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, received ${value}`);
  }
  return parsed;
}

function detectSpeechProfile(audio: WavAudio, frameMs: number): SpeechProfile {
  const windowSamples = Math.round((audio.sampleRate * RMS_WINDOW_MS) / 1_000);
  const windows = buildRmsWindows(audio.samples, audio.sampleRate, windowSamples);
  const peakRms = Math.max(...windows.map((window) => window.rms));
  const tailNoise = median(
    windows
      .filter((window) => window.endMs >= audio.durationMs - RMS_TAIL_NOISE_MS)
      .map((window) => window.rms),
  );
  const threshold = Math.max(
    RMS_MIN_THRESHOLD,
    peakRms * RMS_PEAK_RATIO,
    tailNoise * RMS_NOISE_MULTIPLIER,
  );
  const lastSpeechWindow = windows.findLast((window) => window.rms >= threshold);
  if (!lastSpeechWindow) {
    throw new Error("WAV does not contain detectable speech");
  }

  const endMs = Math.min(
    audio.durationMs,
    lastSpeechWindow.endMs + RMS_HANGOVER_MS,
  );
  const samplesPerFrame = Math.round((audio.sampleRate * frameMs) / 1_000);
  const endSample = Math.round((endMs / 1_000) * audio.sampleRate);
  const frameIndex = Math.floor(endSample / samplesPerFrame);
  const sampleOffset = endSample % samplesPerFrame;
  const offsetWithinFrameMs = (sampleOffset / audio.sampleRate) * 1_000;

  return {
    endMs,
    frameIndex,
    offsetWithinFrameMs,
    threshold,
    windowMs: RMS_WINDOW_MS,
    hangoverMs: RMS_HANGOVER_MS,
  };
}

function buildRmsWindows(
  samples: Float32Array,
  sampleRate: number,
  windowSamples: number,
): Array<{ endMs: number; rms: number }> {
  const windows: Array<{ endMs: number; rms: number }> = [];
  for (let start = 0; start < samples.length; start += windowSamples) {
    let sumSquares = 0;
    const end = Math.min(samples.length, start + windowSamples);
    for (let index = start; index < end; index += 1) {
      const sample = samples[index]!;
      sumSquares += sample * sample;
    }
    windows.push({
      endMs: (end / sampleRate) * 1_000,
      rms: Math.sqrt(sumSquares / Math.max(1, end - start)),
    });
  }
  return windows;
}

function selectBestSummary(summaries: Summary[]): Summary | undefined {
  return summaries
    .filter((summary) => summary.ok)
    .sort(compareEndpointSummaries)[0];
}

function selectLowestMeasuredSummary(summaries: Summary[]): Summary | undefined {
  return summaries
    .filter((summary) =>
      summary.results.every((result) => result.finalizedAfterSpeechEnd),
    )
    .sort(compareEndpointSummaries)[0] ?? summaries[0];
}

function compareEndpointSummaries(left: Summary, right: Summary): number {
  const trailingDelta =
    (left.endpoint?.minTrailingSilenceMs ?? Number.POSITIVE_INFINITY) -
    (right.endpoint?.minTrailingSilenceMs ?? Number.POSITIVE_INFINITY);
  return trailingDelta === 0
    ? left.speechEndToFinalMedian - right.speechEndToFinalMedian
    : trailingDelta;
}

function printSummary(summary: Summary): void {
  console.log(
    `engine=${summary.engineLabel}${summary.endpoint ? ` endpoint=${formatEndpoint(summary.endpoint)}` : ""}`,
  );
  for (const result of summary.results) {
    console.log(formatRun(result));
  }
  console.log(
    `median endpoint->final=${formatMs(summary.speechEndToFinalMedian)} event->final=${formatMs(summary.endpointToFinalMedian)} first-partial-cold=${formatOptionalMs(summary.firstPartialColdMs)} first-partial-warm=${formatOptionalMs(summary.firstPartialWarmMedian)} text-correct=${formatBoolean(summary.textCorrect)} ${summary.passFail} threshold=${PASS_THRESHOLD_MS}ms soft=${SOFT_THRESHOLD_MS}ms ok=${summary.ok}`,
  );
}

function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot compute median of an empty array");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle]!;
  }
  return (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function medianOptional(values: Array<number | undefined>): number | undefined {
  const present = values.filter((value): value is number => value !== undefined);
  return present.length > 0 ? median(present) : undefined;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Session timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function formatRun(result: RunResult): string {
  return [
    `run=${result.run}`,
    `endpoint->final=${formatMs(result.speechEndToFinalMs)}`,
    `endpoint-delay=${formatMs(result.endpointDelayMs)}`,
    `event->final=${formatMs(result.endpointToFinalMs)}`,
    `first-partial=${formatOptionalMs(result.firstPartialMs)}`,
    `wall=${formatMs(result.wallMs)}`,
    `text-correct=${formatBoolean(result.textCorrect)}`,
    `ok=${formatBoolean(result.finalizedAfterSpeechEnd && result.textCorrect)}`,
    `text=${JSON.stringify(result.finalText)}`,
  ].join(" ");
}

function isExpectedTranscript(text: string): boolean {
  return normalizeTranscript(text) === normalizeTranscript(EXPECTED_JFK_TRANSCRIPT);
}

function normalizeTranscript(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function formatEndpoint(endpoint: EndpointConfig | undefined): string {
  if (!endpoint) {
    return "default";
  }
  return `${endpoint.mode ?? "eager"}:trail=${endpoint.minTrailingSilenceMs ?? "default"}ms:minutt=${endpoint.minUtteranceMs ?? "default"}ms`;
}

function formatMs(value: number): string {
  return `${value.toFixed(1)}ms`;
}

function formatOptionalMs(value: number | undefined): string {
  return value === undefined ? "n/a" : formatMs(value);
}

function formatBoolean(value: boolean): "y" | "n" {
  return value ? "y" : "n";
}
