import { performance } from "node:perf_hooks";
import { TtsSynth, type SynthRequest, type SynthResult } from "./synth.ts";
import type { TtsModelId } from "./models.ts";
import type { AudioSegment } from "./contract.ts";

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

/**
 * The pipeline yields the engine-agnostic AudioSegment defined by the TTS
 * contract; SpeechSegment is kept as the internal name used across the demos and
 * sweeps. One shape, one source of truth.
 */
export type SpeechSegment = AudioSegment;

/** The synth capability the pipeline needs, structural so tests can inject a fake. */
export type SegmentSynth = {
  synth(request: SynthRequest): Promise<SynthResult>;
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
    if (isClauseEnd(word)) {
      return index + 1;
    }
  }
  return Number.POSITIVE_INFINITY;
}

/** A word ending a clause (comma/semicolon/colon/dash) — the first-chunk cut cues. */
function isClauseEnd(word: string): boolean {
  return /[,;:—–-]$/.test(word);
}

/** A word ending a sentence: a terminator plus any trailing closing quotes. */
function isSentenceEnd(word: string): boolean {
  return /[.!?][)\]"'”’]*$/.test(word);
}

/**
 * Incremental counterpart to planSegments for the LLM feed: consume text deltas
 * and yield the same speakable chunks planSegments would, but as early as each
 * cut is provable rather than waiting for the whole reply. Feeding the complete
 * text as a single delta yields exactly planSegments(text); token-by-token it
 * emits the aggressive first chunk the moment its boundary is settled, then whole
 * sentences as their terminators arrive, then flushes any trailing fragment.
 */
export async function* planSegmentsStream(
  tokens: AsyncIterable<string>,
): AsyncGenerator<string> {
  let buffer = "";
  let headEmitted = false;
  for await (const delta of tokens) {
    buffer += delta;
    if (!headEmitted) {
      const carved = carveStreamingHead(buffer, false);
      if (!carved) {
        continue;
      }
      yield carved.head;
      buffer = carved.rest;
      headEmitted = true;
    }
    const { sentences, remainder } = drainSentences(buffer, false);
    yield* sentences;
    buffer = remainder;
  }
  if (!headEmitted) {
    const carved = carveStreamingHead(buffer, true);
    if (!carved) {
      return;
    }
    yield carved.head;
    buffer = carved.rest;
  }
  yield* drainSentences(buffer, true).sentences;
}

type WordSpan = { text: string; start: number; end: number };

/** Whitespace-delimited tokens with their raw offsets, so heads slice cleanly. */
function wordSpans(buffer: string): WordSpan[] {
  const spans: WordSpan[] = [];
  const re = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(buffer)) !== null) {
    spans.push({ text: match[0], start: match.index, end: re.lastIndex });
  }
  return spans;
}

/**
 * Words we can safely reason about: all of them once the stream ends, otherwise
 * all but a trailing token that has no whitespace or punctuation after it (it may
 * still grow with the next delta).
 */
function settledWordSpans(buffer: string, ended: boolean): WordSpan[] {
  const spans = wordSpans(buffer);
  if (ended || spans.length === 0) {
    return spans;
  }
  const last = spans[spans.length - 1]!;
  const followedByWhitespace = last.end < buffer.length;
  const punctuationTerminated = /[.!?,;:—–\-)\]"'”’]$/.test(last.text);
  return followedByWhitespace || punctuationTerminated ? spans : spans.slice(0, -1);
}

/**
 * Decide the first-chunk cut from the settled prefix, matching carveFirstChunk:
 * the first clause boundary within the opening words, else the word cap when the
 * first sentence runs longer, else the whole first sentence once it terminates.
 * Returns null while the cut is not yet provable.
 */
function carveStreamingHead(
  buffer: string,
  ended: boolean,
): { head: string; rest: string } | null {
  const settled = settledWordSpans(buffer, ended);
  if (settled.length === 0) {
    return null;
  }
  const terminatedAt = settled.findIndex((span) => isSentenceEnd(span.text));
  const sentenceLength = terminatedAt === -1 ? Infinity : terminatedAt + 1;
  const scanLimit = Math.min(FIRST_CHUNK_MAX_WORDS, sentenceLength, settled.length);
  let clauseCut = Infinity;
  for (let index = 0; index < scanLimit; index += 1) {
    if (isClauseEnd(settled[index]!.text)) {
      clauseCut = index + 1;
      break;
    }
  }

  let cut: number;
  if (clauseCut !== Infinity) {
    cut = clauseCut;
  } else if (Number.isFinite(sentenceLength)) {
    cut = Math.min(FIRST_CHUNK_MAX_WORDS, sentenceLength);
  } else if (settled.length > FIRST_CHUNK_MAX_WORDS) {
    cut = FIRST_CHUNK_MAX_WORDS;
  } else if (ended) {
    cut = settled.length;
  } else {
    return null;
  }

  const head = settled
    .slice(0, cut)
    .map((span) => span.text)
    .join(" ");
  return { head, rest: buffer.slice(settled[cut - 1]!.end) };
}

/**
 * Pull every fully terminated sentence out of the buffer, returning the raw
 * unterminated tail to keep accumulating. When the stream has ended the tail is
 * emitted too, matching splitSentences exactly.
 */
function drainSentences(
  buffer: string,
  ended: boolean,
): { sentences: string[]; remainder: string } {
  if (ended) {
    return { sentences: splitSentences(buffer), remainder: "" };
  }
  const re = /[^.!?]*[.!?]+[)\]"'”’]*\s*|[^.!?]+$/g;
  const sentences: string[] = [];
  let consumed = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(buffer)) !== null) {
    const isTerminated = /[.!?][)\]"'”’]*\s*$/.test(match[0]);
    if (!isTerminated) {
      break;
    }
    const trimmed = match[0].trim();
    if (trimmed) {
      sentences.push(trimmed);
    }
    consumed = re.lastIndex;
  }
  return { sentences, remainder: buffer.slice(consumed) };
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

/**
 * Synthesize a fixed string sentence-by-sentence. Thin wrapper over the shared
 * synthPipeline: plan the chunks up front, then stream them. The incremental
 * (LLM feed) path lives behind the TTS contract and drives synthPipeline with
 * planSegmentsStream instead, so both share one pipeline.
 */
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
  yield* synthPipeline(fromArray(sentences), synth, options.speed ?? 1);
}

/** Adapt a fixed list of chunks to the async stream synthPipeline consumes. */
export async function* fromArray(items: string[]): AsyncGenerator<string> {
  yield* items;
}

/**
 * The 1-deep synth pipeline: consume a stream of planned sentence chunks and
 * yield an AudioSegment per chunk. While the consumer handles segment i, segment
 * i+1 is already synthesizing on the native worker thread; because RTF < 1 with
 * piper/kokoro, synthesis stays ahead and playback runs gapless. Timing is
 * measured from the first chunk pulled, so readyAtMs is honest for both the
 * fixed-string and the incremental (token-fed) callers.
 */
export async function* synthPipeline(
  segments: AsyncIterable<string>,
  synth: SegmentSynth,
  speed: number = 1,
): AsyncGenerator<SpeechSegment> {
  const iterator = segments[Symbol.asyncIterator]();
  let current = await iterator.next();
  if (current.done) {
    return;
  }
  const start = performance.now();
  let pending = synth.synth({ text: current.value, speed });
  let index = 0;
  try {
    while (!current.done) {
      const result = await pending;
      const readyAtMs = performance.now() - start;
      const next = await iterator.next();
      if (!next.done) {
        pending = synth.synth({ text: next.value, speed });
      }
      // Trim the model's ragged silence and append one controlled gap so
      // continuous playback sounds like natural sentence rhythm. audioDurationMs
      // reflects the trimmed+gapped samples, keeping the ahead-of-playback math
      // honest against what actually plays.
      const samples = trimSilence(result.samples, result.sampleRate);
      yield {
        index,
        sentence: current.value,
        samples,
        sampleRate: result.sampleRate,
        readyAtMs,
        synthMs: result.totalSynthMs,
        audioDurationMs: (samples.length / result.sampleRate) * 1_000,
      };
      index += 1;
      current = next;
    }
  } finally {
    // On an early return (barge-in breaks the consumer's loop) a prefetched
    // synth may still be in flight; swallow it so it never surfaces as an
    // unhandled rejection.
    void Promise.resolve(pending).catch(() => {});
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
