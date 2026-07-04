# TTS head-to-head sweep

Offline TTS comparison for the speak-easy spike, mirroring the STT sweep in
MODEL-SWEEP.md. Goal: hear the quality difference between a fast VITS voice
and a modern StyleTTS2-based voice, expose the config surface of
sherpa-onnx-node's `OfflineTts`, and measure synthesis latency. TTS is a
separate bounded context from the STT `VoiceToText` contract: everything lives
in `src/tts/`, nothing touches `src/contract.ts`.

## Models compared

Both ship as tarballs on the same k2-fsa release host as the STT models
(release tag `tts-models`), downloaded and extracted under `models/tts/` via
the shared `ensureAsset` in `src/engines/assets.ts` (also used by the STT
registry after this spike's refactor).

| Model id | Family | Tarball | Extracted | Sample rate | Voices | Load time |
|---|---|---|---|---|---|---|
| `piper-amy` | vits (Piper `vits-piper-en_US-amy-low`, low) | 64MB | 79MB | 16kHz | 1 | ~0.6s |
| `piper-ryan-high` | vits (Piper `vits-piper-en_US-ryan-high`) | 110MB | 147MB | 22kHz | 1 | ~0.5s |
| `piper-lessac-high` | vits (Piper `vits-piper-en_US-lessac-high`) | 110MB | 131MB | 22kHz | 1 | ~0.6s |
| `piper-libritts` | vits (Piper `vits-piper-en_US-libritts_r-medium`) | 78MB | 98MB | 22kHz | 904 | ~0.4s |
| `kokoro-v0.19` | kokoro (`kokoro-en-v0_19`, 82M StyleTTS2-based) | 305MB | 361MB | 24kHz | 11 | ~0.5s |
| `kokoro-int8` | kokoro (`kokoro-int8-en-v0_19`, v0.19 int8) | 98MB | 153MB | 24kHz | 11 | ~0.5s |

The first two rows are the original head-to-head; the middle three high/medium Piper
voices and the int8 Kokoro were added in the 2026-07-04 streamable-quality pass (below).

## Config surface

`OfflineTts` config (per family, see `src/tts/synth.ts`):

- **vits**: `model`, `tokens`, `dataDir` (espeak-ng phoneme data), plus
  `noiseScale`, `noiseScaleW`, `lengthScale` knobs.
- **kokoro**: `model`, `voices` (speaker embedding bank), `tokens`, `dataDir`,
  plus `lengthScale`, `lexicon`, `lang`.
- **shared**: `numThreads`, `provider`, `debug` (under `model`), and top-level
  `maxNumSentences`, `silenceScale`.

Per-request (`generate`/`generateAsync`): `text`, `sid` (speaker id, 0..10 for
kokoro, 0 for single-voice piper), `speed` (rate multiplier). Sweep flags:
`--model <id>` narrows to one model, `--speed <rate>` passes through.

## Streaming: exposed but unusable in 1.13.3

`generateAsync` accepts an `onProgress` callback and the native addon exports
the async paths (`offlineTtsGenerateAsync`, `...WithConfig`), so the API
surface says streaming exists. Two findings, both reproduced:

- **Granularity is per sentence, not per chunk.** The callback fires once per
  generated sentence with that sentence's samples. On a 3-sentence paragraph
  the first callback landed at 696ms vs 3465ms total (piper), so
  sentence-level streaming is real and worthwhile for long text. For
  single-sentence utterances first-audio equals total-synth.
- **The callback path aborts the process.** The TypedThreadSafeFunction that
  delivers callbacks has a use-after-free race: any item still queued when
  generation completes is delivered against freed `TtsCallbackData`, and
  `napi_create_arraybuffer` with the garbage length dies with
  `FATAL ERROR: v8::ArrayBuffer::New Allocation failed`. Not catchable from
  JS. Two sequential `generateAsync` calls with a callback crash
  deterministically; `enableExternalBuffer: false` only shifts the timing.

So the sweep measures **total-synth latency only** (`firstAudioMs` is always
undefined). To get first audio out early today, split text into sentences at
the app level and synth them serially; revisit the callback on a newer
binding.

## Streamable quality (2026-07-04)

The original sweep shipped `piper-amy` (low) as the interactive default. Road
testing found it grating, worst on long sentences and paragraphs, so this pass
hunts the highest-fidelity voice that still **streams on CPU**: RTF below ~0.8
so the sentence pipeline (`stream.ts`) stays ahead of playback. Every registry
model is measured on the long build-status paragraph (the exact long-span case
the user flagged) at 2 and 4 threads, then the streamable ones are driven
through the real pipeline to time first audio.

Machine: Apple Silicon, cpu provider, back-to-back in one session. RTF is
`synth / audio` on the paragraph; first-audio and ahead-of-playback come from
`src/tts/qual.ts` running `streamSpeech` at 4 threads. Reproduce:
`node src/tts/qual.ts` (all models) or `--model <id>` to narrow.

| Model | Family | RTF@2thr | RTF@4thr | Streamable? | First-audio (stream, 4thr) | Download | Sample rate | Voices |
|---|---|---|---|---|---|---|---|---|
| `piper-libritts` | vits medium | 0.059 | **0.037** | yes | 67ms | 78MB | 22kHz | 904 |
| `piper-amy` | vits low | 0.043 | **0.033** | yes | 78ms | 64MB | 16kHz | 1 |
| `piper-lessac-high` | vits high | 0.392 | **0.225** | yes | 442ms | 110MB | 22kHz | 1 |
| `piper-ryan-high` | vits high | 0.430 | **0.249** | yes | 461ms | 110MB | 22kHz | 1 |
| `kokoro-v0.19` | kokoro fp32 | 0.627 | **0.429** | yes | 830ms | 305MB | 24kHz | 11 |
| `kokoro-int8` | kokoro int8 | 0.989 | **0.907** | no | n/a | 98MB | 24kHz | 11 |

Every model marked streamable reported `ahead-of-playback=yes`: synthesis of
sentence _i+1_ finishes before playback of sentence _i_ ends, so the queue
never underruns.

Two results overturn the priors:

- **int8 Kokoro is slower than fp32 Kokoro on this CPU, not faster.**
  `kokoro-int8` (the hypothesized prize) runs RTF 0.91 at 4 threads versus
  `kokoro-v0.19` fp32 at 0.43 and misses the streaming bar. Apple Silicon has
  very fast fp32/NEON throughput and onnxruntime's ARM int8 kernels do not beat
  it, so quantization buys nothing here and loses fidelity. int8 is a dead end
  on this hardware; it would only help on a box where fp32 is the bottleneck.
- **fp32 Kokoro already streams on the paragraph.** The earlier RTF 2.44 was a
  median across five utterances dominated by short sentences, where Kokoro's
  fixed per-inference cost wrecks RTF. Amortized over the ~5.7s paragraph it is
  0.43 at 4 threads, and the sentence pipeline confirmed ahead-of-playback with
  first audio at ~0.8s. (RTF here is paragraph-only and this machine ran the
  whole fleet faster than the historical box, so treat these as relative
  rankings measured together, not absolutes comparable to the old table.)

### Recommendation, ranked by likely naturalness among the streamable

By ear is the tiebreak; this is the expected order by architecture and tier:

1. **`kokoro-v0.19` (fp32)** — fidelity winner. StyleTTS2-based, 24kHz, 11
   voices, audibly the richest and most natural on long spans. Streams at RTF
   0.43 (4thr) with comfortable margin. Costs: ~830ms first audio (vs ~450ms
   for Piper high), 305MB, and worse RTF on very short one-line replies where
   its fixed cost dominates. This is the prize the pass was chasing, reached
   with fp32 rather than int8.
2. **`piper-lessac-high` / `piper-ryan-high`** — high-tier Piper VITS, 22kHz,
   single voice each. A large step up from `piper-amy` on long-sentence prosody,
   with low first audio (~450ms), fat streaming margin (RTF ~0.23), and half
   Kokoro's download. `lessac` is the smoother neutral read; `ryan` is a warmer
   male voice. The safe live-loop pick. Pick one by ear.
3. **`piper-libritts`** (`libritts_r-medium`, 904 voices) — medium tier, so a
   touch less polished than the `-high` Piper voices, but still clearly above
   `amy-low`, blazing fast (RTF 0.037), and multi-speaker for variety.
4. **`piper-amy` (low)** — the incumbent baseline the user found grating.
   Fastest and smallest, 16kHz, kept only as the reference point.

Practical call: ship `piper-lessac-high` (or `ryan-high`) as the new
interactive default (low latency, big margin, 22kHz, 110MB), and offer
`kokoro-v0.19` as the max-fidelity option where ~0.8s first-audio and 305MB are
acceptable. Drop `kokoro-int8`: on this hardware it is strictly worse than fp32.

### Listen (A/B the same two lines across every voice)

Two wavs per model in `results/tts/` (gitignored), same text across all so the
voice is the only variable:

- `qual-<model>-0.wav` — short pangram: _"The quick brown fox jumps over the
  lazy dog."_
- `qual-<model>-1.wav` — the flagged long paragraph: _"The build finished on
  the staging channel. Two checks are still running. Say the word and I will
  promote it."_

Models: `piper-amy`, `piper-ryan-high`, `piper-lessac-high`, `piper-libritts`,
`kokoro-v0.19`, `kokoro-int8` (12 files). Start with the `-1` paragraph files:
that is where `amy-low` grates and the high-tier voices pull ahead.

### Skipped, and why

- **Multi-lang int8 Kokoro (`kokoro-int8-multi-lang-v1_0` / `v1_1`).** They ship
  a `dict/` (jieba) plus `.fst` rule files and require a `dict-dir` the 1.13.3
  Kokoro config does not expose (only `model`, `voices`, `tokens`, `dataDir`,
  `lengthScale`, `lexicon`, `lang`). Chinese-oriented wiring for no English
  fidelity gain over v0.19, and int8 is slower than fp32 here regardless.
- **Matcha (flow-matching).** Needs a companion vocoder onnx
  (`vocos`/`hifigan`) wired as a second model. Fiddly for this pass; the two
  streamable high-fidelity candidates above already answer the question.

## Results (original 2-model sweep)

Machine: Apple Silicon, cpu provider, 2 threads. Per-utterance timings from
`node src/tts/run.ts` over 5 fixed domain-flavored sentences (littleorgans
domain line, digits, director command, neutral, 3-sentence paragraph), after
one unmeasured warmup per model. RTF = synth time / audio time; below 1 is
faster than realtime.

| Model | Cold synth | Median total-synth | Median RTF | RTF range |
|---|---|---|---|---|
| `piper-amy` | 111ms | 1194ms | **0.346** | 0.25 to 0.39 |
| `kokoro-v0.19` | 5335ms | 7978ms | **2.443** | 1.55 to 3.43 |

- **piper-amy is ~3x faster than realtime.** A ~3.4s utterance synthesizes in
  ~1.0 to 1.3s. With app-level sentence chunking that puts first audio for a
  short reply around a second.
- **kokoro is 2.4x slower than realtime at 2 threads.** A ~3s utterance takes
  ~8 to 11s. At 4 threads it improves to RTF 1.84 (5.9s for 3.2s of audio),
  still slower than realtime, so no thread count on this box makes it
  interactive on CPU.
- Voice variety: kokoro sids 1, 5, 9 of the domain sentence are written
  alongside sid 0 so the 11-voice range is audible; piper has a single voice.

Listen: 13 wavs in `results/tts/` (`<model>-<idx>.wav` for the sentence set,
`kokoro-v0.19-voice<sid>.wav` for variety).

## Recommendation

> Superseded for voice selection by the Streamable quality section above, which
> adds high-tier Piper and streamable fp32 Kokoro. Kept for the original
> two-model context. Note the RTF here is a 5-sentence median (short utterances
> included) on a slower run, so it reads higher than the paragraph-only numbers
> above.

- **Latency pick: `piper-amy`.** The only model that can serve an interactive
  voice loop on CPU (RTF 0.35). 16kHz single voice, robotic edge expected of
  Piper low, but responses start fast and the whole model is 64MB.
- **Quality pick: `kokoro-v0.19`.** 24kHz, 11 voices, audibly richer
  StyleTTS2-based synthesis (confirm by ear from `results/tts/`). Usable for
  offline or non-interactive rendering (pre-generated prompts, long-form
  narration) where a 2.4x RTF does not block a conversation.
- For a live assistant reply path, ship piper-amy now; kokoro only becomes a
  live candidate with GPU execution or a faster kokoro build.

## Reproduce

```
node src/tts/qual.ts                      # streamable-quality sweep, all models
node src/tts/qual.ts --model kokoro-v0.19 # one model (repeat --model to select several)
node src/tts/run.ts                       # original per-sentence sweep + variety wavs
node src/tts/run.ts --model piper-amy     # one model
node src/tts/run.ts --speed 1.2           # faster speaking rate
```

Model registry: `src/tts/models.ts`. Adding a candidate is config-only (one
descriptor); swapping is `--model <id>`. Downloads land in `models/tts/`,
wavs in `results/tts/` (both gitignored). `qual.ts` writes the A/B
`qual-<model>-<idx>.wav` samples and the RTF/streamable table; `run.ts` writes
the broader `<model>-<idx>.wav` sentence set.
