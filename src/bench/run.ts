import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import type { EndpointConfig, VoiceToText } from "../contract.ts";
import {
  DEFAULT_ENGINE,
  DEFAULT_FRAME_MS,
  DEFAULT_RUNS,
  EXPECTED_JFK_TRANSCRIPT,
  PARTIAL_ROOTCAUSE,
  PASS_THRESHOLD_MS,
  PTT_MIN_RUNS,
  SESSION_TIMEOUT_MS,
  SHERPA_SWEEP,
  SHERPA_SWEEP_PATH,
  SOFT_THRESHOLD_MS,
} from "./config.ts";
import { runCorpusBench } from "./corpus.ts";
import { parseSherpaModelId, SHERPA_MODELS } from "../engines/sherpa-models.ts";
import {
  formatBoolean,
  formatEndpoint,
  formatMs,
  formatOptionalMs,
} from "./format.ts";
import {
  attachSessionObservers,
  createEngine,
  runPttOnce,
  withTimeout,
} from "./harness.ts";
import { writePttReport, writeSherpaSweep } from "./report.ts";
import { detectSpeechProfile, type SpeechProfile } from "./speech.ts";
import { median, medianOptional } from "./stats.ts";
import {
  DEFAULT_MAX_WORD_ERRORS,
  isExactTranscript,
} from "./transcript.ts";
import type {
  BenchMode,
  CliOptions,
  EngineName,
  FinalObservation,
  PttRunResult,
  PttSummary,
  PttVariant,
  RunResult,
  Summary,
} from "./types.ts";
import { readWavFrames, type WavAudio } from "./wav.ts";

const USAGE = [
  "usage: pnpm bench --wav <path> [--engine stub|moonshine|sherpa] [--model <id>] [--mode sweep|ptt] [--runs <n>] [--frame-ms <n>]",
  "       pnpm bench --corpus <dir> [--engine stub|moonshine|sherpa] [--model <id>] [--frame-ms <n>]  (WER scorer over labeled demo recordings)",
  `       --model ids (sherpa only): ${Object.keys(SHERPA_MODELS).join(", ")}`,
].join("\n");

const options = parseArgs(process.argv.slice(2));
if (options.corpus) {
  await runCorpusBench(options);
} else {
  await runWavBench(options);
}

async function runWavBench(options: CliOptions): Promise<void> {
  if (!options.wav) {
    throw new Error(`Missing required --wav <path> argument\n${USAGE}`);
  }
  if (options.mode === "ptt") {
    options.runs = Math.max(options.runs, PTT_MIN_RUNS);
  }

  const audio = await readWavFrames(options.wav, options.frameMs);
  const speech = detectSpeechProfile(audio, options.frameMs);

  console.log("speak-easy latency benchmark");
  console.log(`wav=${options.wav} mode=${options.mode}`);
  console.log(
    `audio=${audio.durationMs.toFixed(0)}ms speech-end=${speech.endMs.toFixed(0)}ms trailing-silence=${(audio.durationMs - speech.endMs).toFixed(0)}ms`,
  );
  console.log(
    `speech-end-detector=rms-window window=${speech.windowMs}ms hangover=${speech.hangoverMs}ms threshold=${speech.threshold.toFixed(4)}`,
  );
  console.log(
    options.mode === "ptt"
      ? `expected=${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} comparison=word-tolerant(word-errors<=${DEFAULT_MAX_WORD_ERRORS})`
      : `expected=${JSON.stringify(EXPECTED_JFK_TRANSCRIPT)} comparison=case/punct-insensitive-exact`,
  );
  console.log(
    `format=${audio.sampleRate}Hz mono ${audio.bitsPerSample}-bit cadence=${options.frameMs}ms/frame frames=${audio.frames.length} runs=${options.runs}`,
  );
  console.log(describeFeedStrategy(options));

  if (options.mode === "ptt") {
    const summaries: PttSummary[] = [];
    for (const variant of ["strict", "loose"] as const) {
      const summary = await runPttSummary(options, audio, speech, variant);
      printPttSummary(summary);
      summaries.push(summary);
    }
    if (options.engine === "sherpa") {
      await writePttReport(SHERPA_SWEEP_PATH, summaries, speech);
      console.log(`file=${SHERPA_SWEEP_PATH}`);
    }
  } else {
    const summaries = options.engine === "sherpa"
      ? await runSherpaSweep(options, audio, speech)
      : [await runEngineSummary(options, audio, speech, undefined)];

    for (const summary of summaries) {
      printSummary(summary);
    }

    const knee = selectBestSummary(summaries);
    const selected = knee ?? selectLowestMeasuredSummary(summaries);
    if (summaries.length > 1 && selected) {
      console.log(
        `selected=${formatEndpoint(selected.endpoint)} endpoint->final=${formatMs(selected.speechEndToFinalMedian)} event->final=${formatMs(selected.endpointToFinalMedian)} text-correct=${formatBoolean(selected.textCorrect)} first-partial-warm=${formatOptionalMs(selected.firstPartialWarmMedian)} partial-rootcause=${PARTIAL_ROOTCAUSE}`,
      );
    }

    if (options.engine === "sherpa") {
      await writeSherpaSweep(SHERPA_SWEEP_PATH, summaries, knee, selected, speech);
      console.log(`file=${SHERPA_SWEEP_PATH}`);
    }
  }
}

function describeFeedStrategy(cli: CliOptions): string {
  if (cli.mode === "ptt") {
    return "ptt mode: feeding frames in real time up to the release point (strict=last voiced sample, loose=rms end incl. hangover), then flush() immediately; measuring flush->final on the final event";
  }
  return cli.engine === "sherpa"
    ? "feeding frames in real time; sherpa endpoint configs are swept for eager finalization"
    : "feeding frames in real time; endpoint is supplied by the engine or end-of-input";
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
  const engine = createEngine(cli.engine, cli.model);
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
  const textCorrect = isExactTranscript(final.text, EXPECTED_JFK_TRANSCRIPT);

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

async function runPttSummary(
  cli: CliOptions,
  wav: WavAudio,
  speechProfile: SpeechProfile,
  variant: PttVariant,
): Promise<PttSummary> {
  const engine = createEngine(cli.engine, cli.model);
  await engine.prepare?.();
  const engineLabel = engine.label ?? cli.engine;
  const releaseMs =
    variant === "strict" ? speechProfile.voicedEndMs : speechProfile.endMs;
  const releaseFrame = Math.min(
    variant === "strict"
      ? speechProfile.voicedFrameIndex
      : speechProfile.frameIndex,
    wav.frames.length - 1,
  );
  const results: PttRunResult[] = [];

  for (let run = 1; run <= cli.runs; run += 1) {
    results.push(
      await runPttOnce({
        run,
        engine,
        audio: wav,
        releaseFrame,
        frameMs: cli.frameMs,
        expected: EXPECTED_JFK_TRANSCRIPT,
      }),
    );
  }

  const coldFlushToFinalMs = results[0]!.flushToFinalMs;
  const warmValues = results
    .slice(1)
    .map((result) => result.flushToFinalMs)
    .filter((value) => Number.isFinite(value));
  const warmFlushToFinalMedianMs =
    warmValues.length > 0 ? median(warmValues) : Number.NaN;
  const textCorrect = results.every((result) => result.textCorrect);

  return {
    engineLabel,
    variant,
    releaseMs,
    results,
    coldFlushToFinalMs,
    warmFlushToFinalMedianMs,
    textCorrect,
    passFail:
      textCorrect && warmFlushToFinalMedianMs < PASS_THRESHOLD_MS
        ? "PASS"
        : "FAIL",
  };
}

function printPttSummary(summary: PttSummary): void {
  console.log(
    `engine=${summary.engineLabel} mode=ptt variant=${summary.variant} release=${formatMs(summary.releaseMs)} endpoint=manual`,
  );
  for (const result of summary.results) {
    console.log(
      [
        `run=${result.run}${result.run === 1 ? "(cold)" : ""}`,
        `flush->final=${formatMs(result.flushToFinalMs)}`,
        `first-partial=${formatOptionalMs(result.firstPartialMs)}`,
        `word-errors=${result.wordErrors}`,
        `text-correct=${formatBoolean(result.textCorrect)}`,
        `text=${JSON.stringify(result.finalText)}`,
      ].join(" "),
    );
  }
  console.log(
    `variant=${summary.variant} flush->final cold=${formatMs(summary.coldFlushToFinalMs)} warm-median=${formatMs(summary.warmFlushToFinalMedianMs)} text-correct=${formatBoolean(summary.textCorrect)} ${summary.passFail} threshold=${PASS_THRESHOLD_MS}ms`,
  );
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

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    engine: DEFAULT_ENGINE,
    runs: DEFAULT_RUNS,
    frameMs: DEFAULT_FRAME_MS,
    mode: "sweep",
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--engine") {
      options.engine = parseEngine(requireValue(args, index));
      index += 1;
    } else if (arg === "--model") {
      options.model = parseSherpaModelId(requireValue(args, index));
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
    } else if (arg === "--mode") {
      options.mode = parseMode(requireValue(args, index));
      index += 1;
    } else if (arg === "--corpus") {
      options.corpus = requireValue(args, index);
      index += 1;
    } else {
      throw new Error(`Unknown argument ${arg}\n${USAGE}`);
    }
  }

  if (options.corpus && options.wav) {
    throw new Error(`Use --wav or --corpus, not both\n${USAGE}`);
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

function parseMode(value: string): BenchMode {
  if (value === "sweep" || value === "ptt") {
    return value;
  }
  throw new Error(`--mode must be sweep or ptt; received ${value}`);
}

function parsePositiveInteger(flag: string, value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, received ${value}`);
  }
  return parsed;
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

