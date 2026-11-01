import type { Hotwords } from "./sherpa-api.ts";
import type { NemoTransducerModel } from "./sherpa-models.ts";

export type OfflineHotwords = Readonly<{
  phrases: readonly string[];
  score: number;
}>;

/** The host's file is streaming compatibility data; offline owns this snapshot. */
export function snapshotHotwords(hotwords?: Hotwords): OfflineHotwords | undefined {
  return hotwords?.phrases.length
    ? Object.freeze({ phrases: Object.freeze([...hotwords.phrases]), score: hotwords.score })
    : undefined;
}

/** Reconstruct only a descriptor's proven export scores, never infer them by family. */
export function prepareOfflineHotwords(
  model: NemoTransducerModel,
  tokenTable: string,
  hotwords: OfflineHotwords,
): { vocabulary: string; encodings: string[][] } {
  const layout = model.hotwordTokenizer;
  if (!layout) throw new Error(`${model.name}: unsupported hotword tokenizer`);
  if (!Number.isFinite(hotwords.score)) {
    throw new Error(`${model.name}: hotword score must be finite`);
  }
  const pieces = parseTokens(model.name, tokenTable, layout.firstScoredPieceId);
  const scores = new Map(pieces.slice(0, -1).map((piece, id) =>
    [piece, id < layout.firstScoredPieceId ? 0 : layout.firstScoredPieceId - id]));
  const vocabulary = [...scores].map(([piece, score], id) =>
    `${piece}\t${id === layout.firstScoredPieceId ? "-0" : score}\n`).join("");
  const maxBytes = Math.max(...pieces.map((piece) => Buffer.byteLength(piece)));
  const encodings = hotwords.phrases.map((phrase) => {
    const fail = (reason: string): never => {
      throw new Error(`${model.name}: hotword ${JSON.stringify(phrase)}: ${reason}`);
    };
    if (!phrase.trim() || /[\x00-\x08\x0a-\x1f\x7f]/.test(phrase)) {
      fail("empty phrase or native line/control syntax");
    }
    // Native EncodeHotwords reserves a word starting with ':' for a score.
    const words = phrase.split(/[ \t\r\n\v\f]+/).filter(Boolean);
    if (words.some((word) => word.startsWith(":"))) fail("reserved native score syntax");
    if (words.some((word) => /^<[^>]*>$/.test(word))) fail("reserved control or blank token");
    const normalized = words.map((word) => `▁${word}`).join("");
    const encoded = encode(normalized, scores, maxBytes, fail);
    if (!encoded.length || encoded.some((piece) => /^<[^>]*>$/.test(piece))) {
      fail("reserved unknown, control or blank piece");
    }
    const roundtrip = encoded.join("").replaceAll("▁", " ").slice(1);
    if (roundtrip !== words.join(" ")) fail("incomplete text roundtrip");
    return encoded;
  });
  return { vocabulary, encodings };
}

function parseTokens(model: string, table: string, firstScoredPieceId: number): string[] {
  const fail = (): never => { throw new Error(`${model}: malformed tokens table`); };
  const rows = table.trimEnd().split(/\r?\n/);
  const pieces = rows.map((row, id) => {
    const match = /^(\S+)[ \t]+(\d+)$/.exec(row);
    if (!match || Number(match[2]) !== id || match[1]!.includes("\ufffd")) return fail();
    return match[1]!;
  });
  if (new Set(pieces).size !== pieces.length || !pieces.includes("<unk>") ||
      pieces.at(-1) !== "<blk>" || firstScoredPieceId >= pieces.length - 1) fail();
  return pieces;
}

/** simple-sentencepiece 0.7: UTF8 suffix DP, float scores, shortest end on ties. */
function encode(
  text: string,
  scores: ReadonlyMap<string, number>,
  maxBytes: number,
  fail: (reason: string) => never,
): string[] {
  const bytes = Buffer.from(text);
  const best = new Float32Array(bytes.length + 1);
  const next = new Int32Array(bytes.length).fill(-1);
  for (let start = bytes.length - 1; start >= 0; start -= 1) {
    let bestScore = -Infinity;
    for (let end = start + 1; end <= Math.min(bytes.length, start + maxBytes); end += 1) {
      const score = scores.get(bytes.subarray(start, end).toString("utf8"));
      if (score === undefined) continue;
      const candidate = Math.fround(score + best[end]!);
      if (candidate > bestScore) {
        bestScore = candidate;
        next[start] = end;
      }
    }
    // Native fallback has score zero. Preserve its route, then reject it below.
    best[start] = bestScore === -Infinity ? 0 : bestScore;
  }
  const pieces: string[] = [];
  for (let start = 0; start < bytes.length;) {
    const end = next[start]!;
    if (end === -1) fail(`unencodable UTF8 coverage at byte ${start} (unknown or byte fallback)`);
    pieces.push(bytes.subarray(start, end).toString("utf8"));
    start = end;
  }
  return pieces;
}
