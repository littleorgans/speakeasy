# Sherpa model-quality sweep

Streaming-zipformer model comparison for the speak-easy STT spike. Goal:
lowest corpus WER with worst-case (max) flush->final under the strict 200ms
budget (`PASS_THRESHOLD_MS`).

## Method

- Corpus: 13 hand-labeled director-command utterances in `corpus/` (wav +
  json sidecar with confirmed `expected`).
- Scorer: `node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --model <id>`
  re-decodes each wav through the push-to-talk path (endpoint=manual, release
  at end-of-capture) and scores the fresh hypothesis vs `expected`. Comparison
  is case- and punctuation-insensitive.
- Latency: flush->final per utterance; we report the median and the **max**
  (the tail is what the 200ms budget must survive, not the median).
- All models use the int8 encoder where shipped; kroko ships fp32 only.
- Scoring runs on the shared normalized form (`packages/speech-io/src/bench/normalize.ts`:
  lowercase, strip punctuation, spelled numbers canonicalized to digits so
  "10" and "ten" collapse). Raw WER (case/punct only, no number canon) is shown
  alongside so the digit-normalization delta is visible. Latency is unaffected
  by normalization; small median/max drift between runs is machine variance.

## Results

| Model id | Encoder | Raw WER | Normalized WER | median flush->final | max flush->final | Under 200ms? | Notes |
|---|---|---|---|---|---|---|---|
| `en-2023-06-26` (prior default) | ~66M, chunk-16 int8, bpe | 36.2% (17/47) | 34.0% (16/47) | 52.9ms | 101.0ms | yes | previous baseline |
| `en-2023-02-21` | LibriSpeech-only int8 | 27.7% (13/47) | 25.5% (12/47) | 59.5ms | 73.5ms | yes | weaker on varied acoustics / proper nouns |
| `en-2023-06-21` | 187MB int8 (large, Libri+Giga) | 21.3% (10/47) | 19.1% (9/47) | 54.9ms | **300.2ms** | **no** | best full-model WER, but tail blows the budget |
| `en-kroko-2025-08-06` (**default**) | 70MB fp32 (Banafo Kroko) | **17.0% (8/47)** | **12.8% (6/47)** | **29.3ms** | **37.4ms** | yes | **winner: lowest WER and lowest latency** |

## Findings

- **Winner: `en-kroko-2025-08-06`.** Lowest WER (12.8% normalized, a 21.2-point
  drop from the 34.0% baseline; 17.0% -> 12.8% raw -> normalized) and lowest
  latency simultaneously. Max flush->final ~37-52ms leaves ~150ms of headroom
  under the 200ms budget. It also finalizes mid-stream on most utterances
  (flush->final often <3ms), so the flush decode is near-free. Newer (2025)
  training generalizes better to the short, proper-noun-heavy command style
  than the 2023 icefall models.

- **`en-2023-06-21` fails the latency budget on the tail.** It reaches 19.1%
  normalized WER (second best), but the longest utterance ("navigate to
  littleorgans project") took ~300-319ms flush->final on the 187MB encoder,
  exceeding the strict 200ms budget. Good average WER does not save a model
  whose worst case blows the budget. Flagged, not dropped.

- **Digit normalization now applied (implemented).** The corpus `expected`
  transcripts spell numbers as digits ("10"), but models emit words
  ("ten"/"tan"/"tone"). The shared normalizer canonicalizes both sides to
  digits, so legitimate "10"<->"ten" matches no longer count as errors while
  genuine mishears ("tone", "tan") still do. Kroko gains the most (17.0% ->
  12.8%): more of its residual errors were digit-spelling, not acoustics. The
  normalizer runs on every WER path (corpus scorer and jfk bench), see
  `packages/speech-io/src/bench/normalize.ts`, covered by `transcript.test.ts`.

## Hotwords experiment (contextual biasing)

Experiment (not a feature): can hotword biasing recover the default model's
proper-noun misses (littleorgans, chrome, pane, codex)? `SherpaEngine`
auto-loads `./hotwords.txt` if present (one term per line, `#` comments and
blanks ignored), feeds it to sherpa's `hotwordsFile` at `hotwordsScore=2`, and
switches the decoder to `modified_beam_search` (hotwords are ignored under
greedy). Activate with `cp hotwords.sample.txt hotwords.txt`.

**Finding: hotwords are a no-op on the kroko default.** sherpa tokenizes each
hotword line into the model's modeling units. BPE models do this with a
SentencePiece `bpe.model`/`bpe.vocab`; kroko ships neither, and its tokens are
sub-word BPE pieces (not char-level), so sherpa falls back to a literal
tokens.txt lookup and fails on every multi-piece term:
`Cannot find ID for token littleorgans ... Check the tokens.txt`. The context
graph ends up empty.

A/B over the corpus (kroko default):

| | WER | median flush->final | max flush->final | residuals flipped | new errors |
|---|---|---|---|---|---|
| no hotwords (greedy) | 12.8% (6/47) | 26.9ms | 36.5ms | — | — |
| hotwords (modified_beam_search) | 12.8% (6/47) | 32.6ms | 48.8ms | none | none |

Identical WER; `littleorgans` still decodes to "little organs". The switch to
beam search costs ~6ms median / ~12ms max — negligible, still far under the
200ms budget. So latency is not the blocker; **tokenization is**.

**The mechanism works given a bpe.vocab** (proven, so this is a data gap not a
code bug). Regenerating the text `bpe.vocab` for the bpe model
`en-2023-06-26` and biasing with a `CHROME` hotword flipped its decode of the
"Open Chrome browser" clip from `"OPEN GROAM BROWSER"` (greedy) to
`"OPEN CHROME BROWSER"` (beam + hotword).

**Path A investigated (obtain kroko's SentencePiece) — BLOCKED.** Real hotword
biasing needs kroko's *own* `bpe.model` so the exported `bpe.vocab` aligns 1:1
with its `tokens.txt` IDs (a generic English spm mis-tokenizes and mis-biases).
It is not publicly obtainable:

- The k2-fsa release tarball ships only encoder/decoder/joiner onnx + tokens.txt
  (no spm). Upstream `Banafo/Kroko-ASR` and `Banafo/test-onnx` (both public)
  ship only proprietary `.data` community bundles + a `decode_file.py`; no
  `bpe.model`, `bpe.vocab`, or SentencePiece artifact anywhere. `decode_file.py`
  even references `{lang}_encoder.onnx` files that are not actually published.
- Banafo's own `decode_file.py` wires `hotwords_file` but passes no `bpe_vocab`
  and no `modeling_unit`, so it relies on the default — which cannot tokenize
  multi-piece English words either.
- `modelingUnit="cjkchar"` does NOT char-split English (it treats Latin runs as
  whole words), so it fails identically: `Cannot find ID for token littleorgans`.
  There is no English word->subword segmentation without the spm.
- Fabricating a `bpe.vocab` from `tokens.txt` is rejected: `tokens.txt` has the
  pieces and IDs but not SentencePiece merge scores, so the segmentation would
  be wrong and silently mis-bias.

**Recommendation: Path B (FST replacement).** Use `ruleFsts`/`ruleFars`
(post-decode text rewriting / ITN, model-vocab-independent) to fix systematic
mis-splits like "little organs" -> "littleorgans" and "ten" -> "10". This needs
no SentencePiece model and sidesteps the tokenization gate entirely.

The hotwords wiring stays inert by default: no `hotwords.txt` means greedy
baseline, unchanged. It remains useful for any future model that bundles a
`bpe.vocab`.

## Path B: post-decode rewrite (in-house map vs sherpa ruleFsts)

Since hotword biasing is blocked on kroko (Path A), fix its *systematic*
residuals after decoding instead. This was run as a head-to-head of two
encodings of ONE shared ruleset (`packages/speech-io/src/rewrite/rules.json`): littleorgans<-"little
organs", chrome<-"crown", pane<-"pain" (plus a number rule ten->10 that has since
moved to its own stage — see "Graduated" below).

- **Arm 1 — in-house map**: whole-word, case-insensitive regex replacement
  applied to the committed hypothesis. Deliberately separate from normalize.ts
  (that canonicalizes both sides for WER; this transforms the engine output only,
  so WER reflects what a consumer receives). This arm won and was graduated.
- **Arm 2 — sherpa ruleFsts**: a byte-level OpenFst rewrite compiled from the
  same rules by kaldifst (build-time only) and applied inside the engine via
  `ruleFsts`. Archived under `experiments/ruleFsts/` — not a production path.

**Gate (Arm 2): ruleFsts DO fire on the streaming/online recognizer.** Proven on
kroko: with the FST wired, `"Open crown browser"` -> `"Open chrome browser"`;
without it, `"Open crown browser"`. So this is not offline-only.

A/B/C over the corpus (kroko default):

| arm | WER | residuals fixed | new errors | median flush->final | max flush->final |
|---|---|---|---|---|---|
| none (baseline) | 12.8% (6/47) | — | — | 28.8ms | 38.8ms |
| in-house map | 4.3% (2/47) | littleorgans, pane, chrome | none | 30.5ms | 46.5ms |
| ruleFsts | 4.3% (2/47) | littleorgans, pane, chrome | none | 31.3ms | 45.7ms |

Both fix 4 of the 6 residuals (the littleorgans mis-split counts as 2). The 2
left are not in the ruleset: "spawn 10 agents" -> "tone" (a genuine mishear, not
"ten") and "go to sleep" -> "go to" (a dropped word). No new errors from either.

**Comparison.** They AGREE on WER — but only after the FST build was made to
compensate for two byte-level limitations the regex map handles for free:

- *Casing.* The recognizer capitalizes words ("Little Organs"), and a byte FST
  is case-sensitive, so `build-fst.py` emits lowercase AND title-case variants
  (4 rules -> 8 paths). The map gets this from one `i` flag.
- *Boundaries.* The map matches whole words (`\b`); the FST matches substrings,
  so in general text it would over-fire ("often" -> "of10"). Safe on this corpus
  (no such substrings) but a real divergence, on top of the shared over-trigger
  risk (crown/pain/ten are legitimate words; flagged `overTrigger` in rules.json).
- *Latency.* FST composition adds a few ms and scales with text length; on these
  short commands it is within run-to-run noise, and both stay far under 200ms.
- *Build/dependency cost.* The FST needs kaldifst (a compiled pip dep) at build
  time and a committed binary `.fst` that must be rebuilt whenever rules change.
  The map is plain TypeScript reading a human-readable JSON — no build step, no
  binary, trivially diffable and unit-tested.

## Graduated: the `withRewrite` decorator (production)

The in-house map won and shipped as a post-processor **decorator** at the
contract layer. `withRewrite(engine, config)` wraps any `VoiceToText`, rewrites
committed **final** events (partials are left untouched to avoid flicker), and
re-emits. The engine stays pure — no rewrite logic in `SherpaEngine`.

`packages/speech-io/src/rewrite/` modules, one job each:

- `rules.json` — user-editable domain rules (data).
- `rules.ts` — stage 1: whole-word, case-insensitive rule application.
- `numbers.ts` — stage 2: number normalization, words<->digits for 0..999,
  config-gated (`--numbers digits|words|off`, default off). This is the
  digits-vs-words requirement; a spoken counting list stays separate
  ("one two three" -> "1 2 3") while "twenty three" compounds to 23.
- `pipeline.ts` — composes rules then numbers (both idempotent, passthrough).
- `decorator.ts` — `withRewrite` + the session wrapper.

Wiring: the **demo wraps by default** (domain rules on; `--no-rewrite` for raw,
`--numbers` for the number stage). The **bench toggles** `--rewrite on|off` and
re-scores the corpus: raw reproduces **12.8% (6/47)**, wrapped hits **4.3%
(2/47)**, decorator latency within noise of raw. All `packages/speech-io/src/rewrite/` modules are
unit-tested (rule application, number normalization, decorator final-vs-partial).

## Reproduce

```
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa                          # default (en-kroko-2025-08-06)
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-06-26     # prior default
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-06-21
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-02-21

# Post-decode rewrite (corpus only): raw vs wrapped decorator
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --rewrite off            # raw, 12.8%
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --rewrite on             # wrapped, 4.3%
node packages/speech-io/src/bench/run.ts --corpus corpus --engine sherpa --rewrite on --numbers digits

# Archived ruleFsts experiment (not production):
python3 -m venv .venv && ./.venv/bin/pip install kaldifst
./.venv/bin/python experiments/ruleFsts/build-fst.py
```

Model registry: `packages/speech-io/src/engines/sherpa-models.ts`. Adding a candidate is
config-only (one descriptor); swapping is `--model <id>`. Per Stuart's verdict
(2026-07-04) the default is now `en-kroko-2025-08-06`; the no-flag run above
loads it.
