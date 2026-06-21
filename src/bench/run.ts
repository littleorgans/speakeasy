import { setTimeout as delay } from "node:timers/promises";
import { performance } from "node:perf_hooks";
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
const SPEECH_THRESHOLD = 0.01;
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
};

type Summary = {
  engineLabel: string;
  endpoint?: Required<EndpointConfig>;
  results: RunResult[];
  endpointToFinalMedian: number;
  speechEndToFinalMedian: number;
  firstPartialMedian?: number;
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

const best = selectBestSummary(summaries);
if (summaries.length > 1 && best) {
  console.log(
    `best=${formatEndpoint(best.endpoint)} endpoint->final=${formatMs(best.speechEndToFinalMedian)} event->final=${formatMs(best.endpointToFinalMedian)} first-partial=${formatOptionalMs(best.firstPartialMedian)}`,
  );
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
    const result = await runOnce({
      run,
      engine,
      endpoint,
      audio: wav,
      speech: speechProfile,
      frameMs: cli.frameMs,
    });
    results.push(result);
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
  const ok = results.every(
    (result) => result.finalizedAfterSpeechEnd && result.finalText.length > 0,
  );

  return {
    engineLabel,
    endpoint,
    results,
    endpointToFinalMedian,
    speechEndToFinalMedian,
    firstPartialMedian,
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

  const endPromise = session.end();
  const final = await withTimeout(state.final, SESSION_TIMEOUT_MS);
  await endPromise;
  const wallMs = performance.now() - start;
  const observedSpeechEndAt = speechEndAt ?? start + input.speech.endMs;
  const speechEndToFinalMs = final.finalAt - observedSpeechEndAt;

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
  };
}

function attachSessionObservers(session: STTSession): {
  firstPartialAt?: number;
  final: Promise<{ endpointAt: number; finalAt: number; text: string }>;
} {
  const state: {
    firstPartialAt?: number;
    endpointAt?: number;
    finalResolve?: (value: {
      endpointAt: number;
      finalAt: number;
      text: string;
    }) => void;
    finalReject?: (error: unknown) => void;
  } = {};

  const final = new Promise<{
    endpointAt: number;
    finalAt: number;
    text: string;
  }>((resolve, reject) => {
    state.finalResolve = resolve;
    state.finalReject = reject;
  });

  session.on("partial", () => {
    state.firstPartialAt ??= performance.now();
  });
  session.on("endpoint", () => {
    state.endpointAt = performance.now();
  });
  session.on("final", ({ text }: { text: string }) => {
    const finalAt = performance.now();
    state.finalResolve?.({
      endpointAt: state.endpointAt ?? finalAt,
      finalAt,
      text,
    });
  });
  session.on("error", ({ err }: { err: unknown }) => {
    state.finalReject?.(err);
  });

  return {
    get firstPartialAt() {
      return state.firstPartialAt;
    },
    final,
  };
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
  let lastSpeechSample = -1;
  for (let index = 0; index < audio.samples.length; index += 1) {
    if (Math.abs(audio.samples[index]!) >= SPEECH_THRESHOLD) {
      lastSpeechSample = index;
    }
  }
  if (lastSpeechSample < 0) {
    throw new Error("WAV does not contain detectable speech");
  }

  const samplesPerFrame = Math.round((audio.sampleRate * frameMs) / 1_000);
  const endMs = (lastSpeechSample / audio.sampleRate) * 1_000;
  const frameIndex = Math.floor(lastSpeechSample / samplesPerFrame);
  const sampleOffset = lastSpeechSample % samplesPerFrame;
  const offsetWithinFrameMs = (sampleOffset / audio.sampleRate) * 1_000;

  return { endMs, frameIndex, offsetWithinFrameMs };
}

function selectBestSummary(summaries: Summary[]): Summary | undefined {
  return summaries
    .filter((summary) => summary.ok)
    .sort((left, right) => {
      const trailingDelta =
        (left.endpoint?.minTrailingSilenceMs ?? Number.POSITIVE_INFINITY) -
        (right.endpoint?.minTrailingSilenceMs ?? Number.POSITIVE_INFINITY);
      return trailingDelta === 0
        ? left.speechEndToFinalMedian - right.speechEndToFinalMedian
        : trailingDelta;
    })[0];
}

function printSummary(summary: Summary): void {
  console.log(
    `engine=${summary.engineLabel}${summary.endpoint ? ` endpoint=${formatEndpoint(summary.endpoint)}` : ""}`,
  );
  for (const result of summary.results) {
    console.log(formatRun(result));
  }
  console.log(
    `median endpoint->final=${formatMs(summary.speechEndToFinalMedian)} event->final=${formatMs(summary.endpointToFinalMedian)} first-partial=${formatOptionalMs(summary.firstPartialMedian)} ${summary.passFail} threshold=${PASS_THRESHOLD_MS}ms soft=${SOFT_THRESHOLD_MS}ms ok=${summary.ok}`,
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
    `ok=${result.finalizedAfterSpeechEnd && result.finalText.length > 0}`,
    `text=${JSON.stringify(result.finalText)}`,
  ].join(" ");
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
