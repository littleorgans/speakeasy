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

## Results

| Model id | Encoder | Corpus WER | median flush->final | max flush->final | Under 200ms? | Notes |
|---|---|---|---|---|---|---|
| `en-2023-06-26` (current default) | ~66M, chunk-16 int8, bpe | 36.2% (17/47) | 51.0ms | 64.7ms | yes | baseline |
| `en-2023-02-21` | LibriSpeech-only int8 | 27.7% (13/47) | 51.9ms | 62.4ms | yes | weaker on varied acoustics / proper nouns |
| `en-2023-06-21` | 187MB int8 (large, Libri+Giga) | 21.3% (10/47) | 58.2ms | **319.3ms** | **no** | best full-model WER, but tail blows the budget |
| `en-kroko-2025-08-06` | 70MB fp32 (Banafo Kroko) | **17.0% (8/47)** | **29.9ms** | **52.4ms** | yes | **winner: lowest WER and lowest latency** |

## Findings

- **Winner: `en-kroko-2025-08-06`.** Lowest WER (17.0%, a 19.2-point absolute
  drop from the 36.2% baseline) and lowest latency simultaneously. Max
  flush->final 52.4ms leaves ~148ms of headroom under the 200ms budget. It also
  finalizes mid-stream on most utterances (flush->final often <3ms), so the
  flush decode is near-free. Newer (2025) training generalizes better to the
  short, proper-noun-heavy command style than the 2023 icefall models.

- **`en-2023-06-21` fails the latency budget on the tail.** It reaches 21.3%
  WER (second best), but the longest utterance ("navigate to littleorgans
  project") took 319ms flush->final on the 187MB encoder, exceeding the strict
  200ms budget. Good average WER does not save a model whose worst case blows
  the budget. Flagged, not dropped.

- **Digit normalization inflates every model equally.** The corpus `expected`
  transcripts spell numbers as digits ("10"), but all models emit words
  ("ten"/"tan"/"tone"). "10" appears 3 times, so ~6.4 WER points are pure
  text-normalization mismatch affecting all four models identically. Kroko's
  acoustic-only WER is closer to ~10.6% (5/47) once digit-normalization is set
  aside. Consider a digit<->word normalization pass in the scorer, or labeling
  numbers as words, before treating absolute WER as final.

## Reproduce

```
node src/bench/run.ts --corpus corpus --engine sherpa                          # baseline (en-2023-06-26)
node src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-06-21
node src/bench/run.ts --corpus corpus --engine sherpa --model en-2023-02-21
node src/bench/run.ts --corpus corpus --engine sherpa --model en-kroko-2025-08-06
```

Model registry: `src/engines/sherpa-models.ts`. Adding a candidate is
config-only (one descriptor); swapping is `--model <id>`. The default model is
unchanged pending the orchestrator's verdict.
