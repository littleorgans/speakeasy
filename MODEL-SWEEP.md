# Sherpa model-quality sweep

Streaming-zipformer model comparison for the speak-easy STT spike. Goal:
lowest corpus WER with worst-case (max) flush->final under the strict 200ms
budget (`PASS_THRESHOLD_MS`).

## Method

- Corpus: 13 hand-labeled director-command utterances in `corpus/` (wav +
  json sidecar with confirmed `expected`).
- Scorer: `node src/bench/run.ts --corpus corpus --engine sherpa --model <id>`
  re-decodes each wav through the push-to-talk path (endpoint=manual, release
  at end-of-capture) and scores the fresh hypothesis vs `expected`. Comparison
  is case- and punctuation-insensitive.
- Latency: flush->final per utterance; we report the median and the **max**
  (the tail is what the 200ms budget must survive, not the median).
- All models use the int8 encoder where shipped; kroko ships fp32 only.
- Scoring runs on the shared normalized form (`src/bench/normalize.ts`:
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
  `src/bench/normalize.ts`, covered by `transcript.test.ts`.

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

**Paths forward** (kroko biasing, if pursued): (a) obtain Banafo's SentencePiece
`bpe.model` for kroko from HuggingFace and generate its `bpe.vocab`; (b) FST
rewriting via `ruleFsts`/`ruleFars` (post-decode text replacement / ITN, not
decode-time biasing) to fix systematic mis-splits like "little organs" ->
"littleorgans" and "ten" -> "10"; (c) hand-tokenized hotwords using kroko's
existing token pieces (brittle, needs the segmentation). The wiring stays inert
by default: no `hotwords.txt` means greedy baseline, unchanged.

## Reproduce

```
node src/bench/run.ts --corpus corpus --engine sherpa                          # default (en-kroko-2025-08-06)
node src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-06-26     # prior default
node src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-06-21
node src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-02-21
```

Model registry: `src/engines/sherpa-models.ts`. Adding a candidate is
config-only (one descriptor); swapping is `--model <id>`. Per Stuart's verdict
(2026-07-04) the default is now `en-kroko-2025-08-06`; the no-flag run above
loads it.
