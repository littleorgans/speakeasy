import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { COMPLETE_MARKER } from "../engines/assets.ts";
import {
  adaptIpa,
  CMU_DICT,
  createTtsFrontend,
  normalizeText,
  parseTokenSymbols,
  toPhonemeCodepoints,
  toMisaki,
} from "./frontend.ts";
import { EnglishG2P } from "./vendor/piper-plus-g2p/en/index.js";

const g2p = new EnglishG2P({
  cmuDict: { march: "M AA1 R CH", stuart: "S T UW1 ER0 T", car: "K AA1 R", h: "EY1 CH" },
});

test("the CMU dictionary is pinned at the vendored commit", () => {
  assert.match(CMU_DICT.sha256, /^[0-9a-f]{64}$/);
  assert.ok(CMU_DICT.bytes > 0);
  assert.match(CMU_DICT.url, /\/ayutaz\/piper-plus\/4fd95c24d364fcaeba200ee606c0902e9bb70768\//);
  assert.match(CMU_DICT.licence.attribution, /Carnegie Mellon Pronouncing Dictionary/);
});

test("normalizeText reads clocks, am/pm, dotted names, hyphens and initialisms", () => {
  assert.equal(normalizeText("Leave at 7:05 pm.", false), "Leave at 7 oh 5 p m.");
  assert.equal(normalizeText("Back by 9 a.m. today", false), "Back by 9 ay m. today");
  assert.equal(normalizeText("Open notes.md now", false), "Open notes dot m d now");
  assert.equal(normalizeText("a well-known API", false), "a well known ay p i");
  assert.equal(normalizeText("I am here", false), "I am here");
  assert.equal(normalizeText("the BBC zx", true), "the b b c zed x");
  assert.equal(normalizeText("the BBC zx", false), "the b b c z x");
});

test("adaptIpa keeps U.S. rhotics and lengthens LOT for American voices", () => {
  const ipa = g2p.phonemize("march car").tokens.join("");
  assert.equal(adaptIpa(ipa, false), "mˈɑːɹtʃ kˈɑːɹ");
});

test("morning times use the letter A rather than the article", () => {
  const clockG2p = new EnglishG2P({ cmuDict: { a: "AH0", ay: "EY1", m: "EH1 M" } });
  const spoken = clockG2p.phonemize(normalizeText("9 am", false)).tokens.join("");
  assert.ok(spoken.includes("ˈeɪ"), spoken);
  assert.equal(normalizeText("9 am", false), "9 ay m");
});

test("version numbers retain every separator through G2P", () => {
  const versionG2p = new EnglishG2P({ cmuDict: {
    two: "T UW1", dot: "D AA1 T", zero: "Z IH1 R OW0", one: "W AH1 N", point: "P OY1 N T",
  } });
  assert.deepEqual(
    versionG2p.phonemize(normalizeText("2.0.1", false)).tokens,
    versionG2p.phonemize("two point zero point one").tokens,
  );
  assert.equal(normalizeText("Version 2.0.1, decimal 2.5.", false), "Version two point zero point one, decimal 2.5.");
});

test("adaptIpa applies the RP rules for British voices", () => {
  const ipa = g2p.phonemize("march car stuart").tokens.join("");
  assert.equal(adaptIpa(ipa, true), "mˈɑːtʃ kˈɑː stˈuːət");
});

test("toPhonemeCodepoints rejects meaningful missing phonemes and permits absent separators", () => {
  const symbols = new Set(["a", "b", " ", "."]);
  assert.throws(() => toPhonemeCodepoints("ab x. a?", symbols), /Unknown TTS phonemes.*x/);
  assert.deepEqual(toPhonemeCodepoints("ab. a?", symbols), [[97, 98, 46], [97]]);
  assert.deepEqual(toPhonemeCodepoints("... ?", symbols), []);
  assert.throws(() => toPhonemeCodepoints("aˈ", symbols), /ˈ/);
});

test("Kokoro mapping matches research diphthongs, rhotics, affricates and syllabic consonants", () => {
  assert.equal(toMisaki("ɹ̩ ɚ ɝ ɜːɹ ɜː oʊ əʊ eɪ aɪ aʊ ɔɪ t͡ʃ d^ʒ ɫ g r ɐ n̩ ɑː"),
    "əɹ əɹ ɜɹ ɜɹ ɜɹ O Q A I W Y ʧ ʤ l ɡ ɹ ə ᵊn ɑ");
  const raw = g2p.phonemize("march car stuart").tokens.join("");
  assert.equal(toMisaki(raw), "mˈɑɹʧ kˈɑɹ stˈuəɹt");
});

test("createTtsFrontend asks for no synthesis when nothing is speakable", async () => {
  const dir = await mkdtemp(join(tmpdir(), "speak-easy-frontend-"));
  try {
    const tokens = join(dir, "tokens.txt");
    await writeFile(tokens, "_ 0\n^ 1\n$ 2\n  3\n. 4\nm 5\nˈ 6\nɑ 7\nɹ 8\nʧ 9\nt 10\nʃ 11\nː 12\n");
    const g2pDir = join(dir, "piper-plus-g2p-4fd95c24");
    await mkdir(g2pDir);
    await writeFile(join(g2pDir, "cmudict_data.json"), JSON.stringify({ march: "M AA1 R CH" }));
    await writeFile(join(g2pDir, COMPLETE_MARKER), `${CMU_DICT.sha256}  cmudict_data.json\n`);
    const frontend = await createTtsFrontend({ tokens, lang: "en-us" }, dir);
    assert.equal(frontend.request("...", 0, 1), undefined);
    assert.equal(frontend.request("   ", 0, 1), undefined);
    const kokoro = await createTtsFrontend({ tokens, lang: "en-us", family: "kokoro" }, dir);
    const kristin = await createTtsFrontend({ tokens, lang: "en", family: "vits" }, dir);
    const ipa = (f: typeof frontend) => f.request("march", 0, 1)!.generationConfig!.phonemeCodepoints!.map((s) => String.fromCodePoint(...s)).join("");
    assert.equal(ipa(kokoro), "mˈɑɹʧ");
    assert.equal(ipa(kristin), "mˈɑːtʃ");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("parseTokenSymbols keeps the space token", () => {
  assert.deepEqual([...parseTokenSymbols("_ 0\n^ 1\n  3\nɑ 50\n")], ["_", "^", " ", "ɑ"]);
});
