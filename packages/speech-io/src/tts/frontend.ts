import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { TtsRequest } from "sherpa-onnx-node";
import { ensureAsset, type FileSetSpec } from "../engines/assets.ts";
import type { ModelLicence } from "../engines/sherpa-models.ts";
import { TTS_ROOT } from "./models.ts";
import { EnglishG2P } from "./vendor/piper-plus-g2p/en/index.js";

/**
 * Text frontend for every sherpa voice, Piper and Kokoro alike: the one swap
 * point for the phonemizer. sherpa-onnx 2.x dropped its built in espeak-ng
 * (GPL) and ignores `dataDir`, so text reaches phonemes only through caller
 * supplied phonemes or a lexicon.
 *
 * Piper Plus's English G2P (CMU dictionary, vendored at commit 4fd95c24)
 * turns text into IPA, adapted to the espeak voice the model was trained
 * with, and sherpa receives it as phoneme codepoints. Picked by the Space
 * lead from voice-out research (2026-11-01). Against espeak-ng: Kristin MOS
 * 4.253 to 4.194, WER 14.0% to 13.2%; Kokoro v1.0 af_heart MOS 4.161 to
 * 4.294, WER 7.9% to 7.0%. The text rules and the British mapping port that
 * research's adapter.
 */

export type TtsFrontend = {
  /**
   * The generateAsync request for one synth call, or undefined when the text
   * has nothing to speak. Never sets onProgress.
   */
  request(text: string, sid: number, speed: number): TtsRequest | undefined;
};

const PIPER_PLUS_COMMIT = "4fd95c24d364fcaeba200ee606c0902e9bb70768";

/** CMU Pronouncing Dictionary 0.7b as JSON, from the vendored commit. */
export const CMU_DICT = {
  url: `https://raw.githubusercontent.com/ayutaz/piper-plus/${PIPER_PLUS_COMMIT}/src/wasm/g2p/data/cmudict_data.json`,
  sha256: "f78ed3bfa40146b70b9de78a7c67ef3be6b6c11671bb2a050442f232252a3f01", // computed 2026-11-01
  bytes: 3_748_042,
  licence: {
    id: "LicenseRef-CMUdict",
    terms:
      "CMU Pronouncing Dictionary 0.7b, BSD style: keep the copyright notice and acknowledgement, no CMU endorsement or naming. Checked 2026-11-01.",
    commercialUse: true,
    redistribution: true,
    attribution:
      "This product includes data from the Carnegie Mellon Pronouncing Dictionary, which is freely available at <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>.",
    source: `https://github.com/ayutaz/piper-plus/blob/${PIPER_PLUS_COMMIT}/src/wasm/g2p/THIRD_PARTY_LICENSES.md`,
  } satisfies ModelLicence,
} as const;

/** Espeak voices whose training phonemes are British (Received Pronunciation). */
const BRITISH_VOICES = new Set(["en", "en-gb", "en-gb-x-rp"]);

/**
 * Letter names for spelled initialisms. CMU lists every other single letter
 * under its name ("h" is EY1 CH) but reads "a" as a schwa.
 */
const LETTER_NAMES: Record<string, string> = { a: "ay" };

/** Not followed by an (optionally stressed) vowel: a coda position. */
const CODA = "(?![ˈˌ]?[aeiouæɑɒɔəɛɜɪʊʌ])";

/** voice-out's ordered US -> RP rules; START is parked in U+E000 meanwhile. */
const RP_RULES: readonly [RegExp, string][] = (
  [
    [`ɑːɹ${CODA}`, "\uE000"],
    [`[ɔo]ːɹ${CODA}`, "ɔː"],
    [`ɜːɹ${CODA}`, "ɜː"],
    [`ɪɹ${CODA}`, "ɪə"],
    [`ɛɹ${CODA}`, "eə"],
    [`ʊɹ${CODA}`, "ʊə"],
    ["ɑː", "ɒ"],
    ["oʊ", "əʊ"],
    ["ɝ", "ɜː"],
    ["ɚ", "ə"],
    ["ɾ", "t"],
    ["ᵻ", "ɪ"],
    ["\uE000", "ɑː"],
    [`(?<=[ːəɪʊaeɛæʌɒ])ɹ${CODA}`, ""],
  ] as const
).map(([pattern, replacement]) => [new RegExp(pattern, "gu"), replacement]);

/**
 * Text rules applied before G2P: clock times, dotted names, hyphens, am/pm
 * after a number, and initialisms spelled by letter name. Piper Plus spells
 * the remaining numbers itself.
 */
export function normalizeText(text: string, british: boolean): string {
  const letters: Record<string, string> = british ? { ...LETTER_NAMES, z: "zed" } : LETTER_NAMES;
  return text
    .replace(/(\d+):(\d+)/g, (_match, hours: string, minutes: string) =>
      `${Number(hours)} ${Number(minutes) < 10 ? "oh " : ""}${Number(minutes)}`)
    .replace(/(?<=\d)\s*([ap])\.?m(?![A-Za-z])/gi, (_match, half: string) => ` ${half.toLowerCase()} m`)
    .replace(/(?<=[A-Za-z])\.(?=[A-Za-z])/g, " dot ")
    .replace(/-/g, " ")
    .replace(/\b[A-Za-z]{1,8}\b/g, (word) =>
      /^[A-Z]{2,8}$/.test(word) || (/^[bcdfghjklmnpqrstvwxz]+$/i.test(word) && word.toLowerCase() !== "a")
        ? [...word.toLowerCase()].map((c) => letters[c] ?? c).join(" ")
        : word)
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Adapt Piper Plus IPA to the voice's espeak convention: canonical symbols
 * for both, then voice-out's RP rules for British voices, or espeak's long
 * U.S. LOT vowel for American ones.
 */
export function adaptIpa(ipa: string, british: boolean): string {
  const canonical = ipa.replace(/ɫ/g, "l").replace(/g/g, "ɡ").replace(/͡/g, "");
  if (!british) {
    return canonical.replace(/ɹ̩/g, "ɚ").replace(/ɑ(?!ː)/g, "ɑː");
  }
  let rp = canonical.replace(/ɑ(?!ː)/g, "ɒ").replace(/ɹ̩/g, "ə");
  for (const [pattern, replacement] of RP_RULES) {
    rp = rp.replace(pattern, replacement);
  }
  return rp.replace(/æ/g, "a");
}

/** One codepoint array per sentence, keeping only symbols the voice knows. */
export function toPhonemeCodepoints(ipa: string, symbols: ReadonlySet<string>): number[][] {
  return (ipa.match(/[^.!?]+[.!?]?/g) ?? [])
    .map((sentence) => [...sentence.trim()].filter((c) => symbols.has(c)))
    .filter((sentence) => sentence.length > 0)
    .map((sentence) => sentence.map((c) => c.codePointAt(0)!));
}

/** Symbols in a sherpa tokens.txt (`<symbol> <id>` per line; the symbol may be a space). */
export function parseTokenSymbols(tokens: string): Set<string> {
  const symbols = new Set<string>();
  for (const line of tokens.split("\n")) {
    const split = line.lastIndexOf(" ");
    if (split > 0) symbols.add(line.slice(0, split));
  }
  return symbols;
}

let english: Promise<EnglishG2P> | undefined;

/** The G2P over the pinned CMU dictionary, installed and parsed once per process. */
function loadEnglishG2P(root: string): Promise<EnglishG2P> {
  english ??= (async () => {
    const dict: FileSetSpec = {
      dir: join(root, `piper-plus-g2p-${PIPER_PLUS_COMMIT.slice(0, 8)}`),
      files: [{ url: CMU_DICT.url, sha256: CMU_DICT.sha256, name: "cmudict_data.json", bytes: CMU_DICT.bytes }],
    };
    await ensureAsset(dict);
    const cmuDict = JSON.parse(await readFile(join(dict.dir, "cmudict_data.json"), "utf8")) as Record<string, string>;
    return new EnglishG2P({ cmuDict });
  })().catch((err: unknown) => {
    english = undefined;
    throw err;
  });
  return english;
}

/**
 * The frontend for one loaded voice: `tokens` is its tokens.txt, `voice` the
 * espeak voice its metadata names (`OfflineTts.lang`, "en-us" for the
 * registry voices).
 */
export async function createTtsFrontend(
  voice: { tokens: string; lang: string },
  root: string = TTS_ROOT,
): Promise<TtsFrontend> {
  const [g2p, tokens] = await Promise.all([loadEnglishG2P(root), readFile(voice.tokens, "utf8")]);
  const symbols = parseTokenSymbols(tokens);
  const british = BRITISH_VOICES.has(voice.lang);
  return {
    request: (text, sid, speed) => {
      const ipa = adaptIpa(g2p.phonemize(normalizeText(text, british)).tokens.join(""), british);
      const phonemeCodepoints = toPhonemeCodepoints(ipa, symbols);
      return phonemeCodepoints.length > 0
        ? { text: "", sid, speed, generationConfig: { phonemeCodepoints, sid, speed } }
        : undefined;
    },
  };
}
