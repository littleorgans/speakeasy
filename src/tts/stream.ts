import { performance } from "node:perf_hooks";
import { TtsSynth } from "./synth.ts";
import type { TtsModelId } from "./models.ts";

/**
 * Sentence-pipelined TTS streaming: the workaround for the broken native
 * onProgress callback (see synth.ts). Text is split into sentences and
 * synthesized sequentially with the stable no-callback generateAsync; each
 * segment is yielded as soon as it is ready, so playback of sentence 1 starts
 * after ~one sentence of synth time instead of the whole reply. With piper-amy
 * (RTF ~0.35) synthesis stays ahead of playback, so segments queue gaplessly.
 *
 * Pipeline depth is 1: the next sentence starts synthesizing while the
 * consumer plays the current one. generateAsync runs on a native worker
 * thread, so the overlap is real parallelism, not event-loop interleaving.
 */

export type SpeechSegment = {
  index: number;
  sentence: string;
  samples: Float32Array;
  sampleRate: number;
  /** Wall-clock ms from stream start until this segment was ready. */
  readyAtMs: number;
  /** Synth time for this segment alone. */
  synthMs: number;
  audioDurationMs: number;
};

export type StreamSpeechOptions = {
  model?: TtsModelId;
  speed?: number;
  /** Reuse a loaded synth (skips model load); overrides `model`. */
  synth?: TtsSynth;
};

/**
 * Split text into speakable sentences. Terminators (.!?) plus following
 * quotes/brackets stay attached; whitespace-only pieces are dropped. Text
 * without a terminator is one sentence.
 */
export function splitSentences(text: string): string[] {
  const matches = text.match(/[^.!?]*[.!?]+[)\]"'”’]*\s*|[^.!?]+$/g) ?? [];
  return matches.map((sentence) => sentence.trim()).filter(Boolean);
}

/**
 * Word cap for the first streamed chunk. The opening chunk is carved short so
 * first audio lands ASAP (kokoro's fixed per-synth cost makes a full first
 * sentence the dominant TTFA term); later chunks stay whole sentences for
 * natural prosody. Tune here.
 */
export const FIRST_CHUNK_MAX_WORDS = 4;

/**
 * Plan streaming chunks: an aggressively short first chunk, then whole
 * sentences. The first sentence is cut at the earliest of its first clause
 * boundary (comma/semicolon/colon/dash) or FIRST_CHUNK_MAX_WORDS words; a first
 * sentence already within the cap is left whole. Splitting only the opener
 * keeps the prosody cost off every later sentence.
 */
export function planSegments(text: string): string[] {
  const sentences = splitSentences(text);
  if (sentences.length === 0) {
    return [];
  }
  const [head, tail] = carveFirstChunk(sentences[0]!);
  const rest = sentences.slice(1);
  return tail ? [head, tail, ...rest] : [head, ...rest];
}

/** Split the opening sentence into [short head, remainder]; tail "" = no split. */
function carveFirstChunk(sentence: string): [string, string] {
  const words = sentence.split(/\s+/);
  const clauseCut = firstClauseWordCount(words);
  const cut = Math.min(clauseCut, FIRST_CHUNK_MAX_WORDS);
  if (cut >= words.length) {
    return [sentence, ""];
  }
  return [words.slice(0, cut).join(" "), words.slice(cut).join(" ")];
}

/** Word count up to and including the first word ending a clause, else Infinity. */
function firstClauseWordCount(words: string[]): number {
  for (const [index, word] of words.entries()) {
    if (/[,;:—–-]$/.test(word)) {
      return index + 1;
    }
  }
  return Number.POSITIVE_INFINITY;
}

/** Below this absolute amplitude (fraction of full scale) a sample is silence. */
const SILENCE_AMPLITUDE = 0.015;
/** Speech guard kept either side of the trim so onsets/tails are not clipped. */
const EDGE_KEEP_MS = 8;
/**
 * Controlled pause appended to each segment in place of the model's ragged
 * end padding, so back-to-back sentences carry a natural speech rhythm instead
 * of dead air. Piper pads utterance ends ~190ms and Kokoro pads both ends
 * ~120-195ms; those vary per sentence, this does not.
 */
export const INTER_SENTENCE_GAP_MS = 150;

/**
 * Trim leading and trailing near-silence from a segment, then append a fixed
 * inter-sentence gap. Pure: takes/returns samples, so it is unit-testable and
 * reused by any consumer that plays segments continuously. A fully silent input
 * collapses to just the gap.
 */
export function trimSilence(
  samples: Float32Array,
  sampleRate: number,
  gapMs: number = INTER_SENTENCE_GAP_MS,
): Float32Array {
  const gap = Math.round((sampleRate * gapMs) / 1000);
  let start = 0;
  while (start < samples.length && Math.abs(samples[start]!) < SILENCE_AMPLITUDE) {
    start += 1;
  }
  let end = samples.length;
  while (end > start && Math.abs(samples[end - 1]!) < SILENCE_AMPLITUDE) {
    end -= 1;
  }
  if (start >= end) {
    return new Float32Array(gap);
  }
  const keep = Math.round((sampleRate * EDGE_KEEP_MS) / 1000);
  start = Math.max(0, start - keep);
  end = Math.min(samples.length, end + keep);
  const out = new Float32Array(end - start + gap);
  out.set(samples.subarray(start, end), 0);
  return out;
}

/** Synthesize text sentence-by-sentence, yielding segments as they complete. */
export async function* streamSpeech(
  text: string,
  options: StreamSpeechOptions = {},
): AsyncGenerator<SpeechSegment> {
  const sentences = planSegments(text);
  if (sentences.length === 0) {
    return;
  }
  const synth =
    options.synth ?? (await TtsSynth.create(options.model ?? "piper-amy"));
  const speed = options.speed ?? 1;
  const start = performance.now();

  const synthOne = (sentence: string) =>
    synth.synth({ text: sentence, speed });

  // 1-deep pipeline: while the consumer handles segment i, segment i+1 is
  // already synthesizing on the native worker thread.
  let pending = synthOne(sentences[0]!);
  for (const [index, sentence] of sentences.entries()) {
    const result = await pending;
    const readyAtMs = performance.now() - start;
    const next = sentences[index + 1];
    if (next !== undefined) {
      pending = synthOne(next);
    }
    // Trim the model's ragged silence and append one controlled gap so
    // continuous playback sounds like natural sentence rhythm. audioDurationMs
    // reflects the trimmed+gapped samples, keeping the ahead-of-playback math
    // honest against what actually plays.
    const samples = trimSilence(result.samples, result.sampleRate);
    yield {
      index,
      sentence,
      samples,
      sampleRate: result.sampleRate,
      readyAtMs,
      synthMs: result.totalSynthMs,
      audioDurationMs: (samples.length / result.sampleRate) * 1_000,
    };
  }
}

export type StreamTimingReport = {
  timeToFirstAudioMs: number;
  /** True iff every segment was ready before its gapless play slot started. */
  aheadOfPlayback: boolean;
  segments: Array<{
    index: number;
    sentence: string;
    readyAtMs: number;
    /** When gapless playback would need this segment (readyAt of segment 0 + prior audio). */
    playStartMs: number;
    /** playStartMs - readyAtMs; negative means an underrun (gap). */
    marginMs: number;
    synthMs: number;
    audioDurationMs: number;
  }>;
};

/**
 * Prove (or disprove) gapless playback from segment timings: segment i's play
 * slot starts when segment 0 was ready plus all prior audio; it must be ready
 * by then. Pure so the invariant is unit-testable.
 */
export function buildTimingReport(
  segments: Array<
    Pick<
      SpeechSegment,
      "index" | "sentence" | "readyAtMs" | "synthMs" | "audioDurationMs"
    >
  >,
): StreamTimingReport {
  if (segments.length === 0) {
    throw new Error("Cannot build a timing report from zero segments");
  }
  const timeToFirstAudioMs = segments[0]!.readyAtMs;
  let priorAudioMs = 0;
  const rows = segments.map((segment) => {
    const playStartMs = timeToFirstAudioMs + priorAudioMs;
    priorAudioMs += segment.audioDurationMs;
    return {
      index: segment.index,
      sentence: segment.sentence,
      readyAtMs: segment.readyAtMs,
      playStartMs,
      marginMs: playStartMs - segment.readyAtMs,
      synthMs: segment.synthMs,
      audioDurationMs: segment.audioDurationMs,
    };
  });
  return {
    timeToFirstAudioMs,
    aheadOfPlayback: rows.every((row) => row.marginMs >= 0),
    segments: rows,
  };
}
