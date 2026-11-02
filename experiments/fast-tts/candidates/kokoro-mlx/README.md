# Candidate: Kokoro v1.0 on MLX

Kokoro 82M v1.0, the mlx-community bf16 conversion, run by mlx-audio on the Apple GPU (Metal). Engines `kokoro-mlx-heart` (af_heart) and `kokoro-mlx-michael` (am_michael). `render.py` follows [CONTRACT.md](../../CONTRACT.md) and imports `fasttts` from `experiments/fast-tts/`.

## Pins

| What | Pin |
| --- | --- |
| Weights | `mlx-community/Kokoro-82M-bf16@a71e4d38b236d968966a2002c4c895dbd12b1c3c`: `config.json`, `kokoro-v1_0.safetensors` (327,115,152 bytes), `voices/af_heart.safetensors`, `voices/am_michael.safetensors`; sha256 checked by `render.py` before load |
| Runtime | `mlx-audio==0.6.3`, `mlx==0.32.4` (resolved), Python 3.12 |
| Frontend | `misaki[en]==0.9.4` (unpinned by mlx-audio; resolved 2026-11-03), `spacy==3.8.18`, `en_core_web_sm==3.8.0`, `phonemizer-fork==3.3.2`, `espeakng-loader==0.2.4` (bundles libespeak-ng 1.52) |

## Licences

- Code: mlx-audio MIT, mlx MIT, misaki Apache 2.0, spaCy and en_core_web_sm MIT.
- Weights: Apache 2.0 (conversion card; base model hexgrad/Kokoro-82M).
- GPL boundary: misaki's English out of vocabulary fallback loads espeak-ng through `phonemizer-fork` (GPL 3.0 per its metadata) and `espeakng-loader` (no licence in its metadata; it ships libespeak-ng, GPL 3.0 or later upstream). Same boundary as voice-out kok-F5. No voice cloning.

## Setup

From the speak-easy repo root.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
uv venv -p 3.12 $TMPDIR/venv-hf && uv pip install --python $TMPDIR/venv-hf/bin/python huggingface_hub
$TMPDIR/venv-hf/bin/hf download mlx-community/Kokoro-82M-bf16 config.json kokoro-v1_0.safetensors \
  voices/af_heart.safetensors voices/am_michael.safetensors \
  --revision a71e4d38b236d968966a2002c4c895dbd12b1c3c --local-dir $MODELS/kokoro-82m-mlx-bf16-a71e4d38
uv venv -p 3.12 $TMPDIR/venv-kokoro-mlx
uv pip install --python $TMPDIR/venv-kokoro-mlx/bin/python 'mlx-audio==0.6.3' 'misaki[en]==0.9.4' \
  'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
```

`en_core_web_sm` is installed up front because misaki otherwise tries to download it with pip at first use. `render.py` sets `HF_HUB_OFFLINE=1`: in one untimed online try the first `generate` call took 39 s, against 3.7 s for the offline pipeline build (cause not isolated).

## Run

```bash
PY=$TMPDIR/venv-kokoro-mlx/bin/python
$PY -I experiments/fast-tts/candidates/kokoro-mlx/render.py --root $MODELS --engine kokoro-mlx-heart \
  --temperature warm --repeats 1 --json $TMPDIR/kokoro-mlx-heart--warm--stream.json --clips $ASSETS/clips
```

Flags match `render-baselines.ts`: `--temperature cold|warm`, `--mode stream|full`, `--repeats N`, `--text <id>` (repeatable; cold takes one, default `status`), `--clips <dir>`, `--timed`.

Timed pass block for `run-timed.sh`, same file names as the baselines:

```bash
for engine in kokoro-mlx-heart kokoro-mlx-michael; do
  for text in $COLD_TEXTS; do for i in $(seq 1 "$RUNS"); do
    $PY -I experiments/fast-tts/candidates/kokoro-mlx/render.py --root "$MODELS" --engine $engine --temperature cold \
      --text $text $TIMED --json "$OUT/$engine--cold--$text--$i.json" >> "$OUT/$engine--cold.log" 2>&1
  done; done
  for mode in $MODES; do
    $PY -I experiments/fast-tts/candidates/kokoro-mlx/render.py --root "$MODELS" --engine $engine --temperature warm \
      --mode $mode --repeats "$RUNS" $TIMED ${CLIPS:+--clips "$CLIPS"} --json "$OUT/$engine--warm--$mode.json" \
      > "$OUT/$engine--warm--$mode.log" 2>&1
  done
done
```

## Stream policy

- Segments: a port of speak-easy `planSegments` from `packages/speech-io/src/tts/stream.ts` (first chunk is the first clause or 4 words of sentence 1, whichever is shorter; then whole sentences). Checked identical to the TypeScript on all ten cohort texts.
- Each segment is a complete waveform: one `Model.generate` call per text with the segments newline separated (`split_pattern="\n+"`), one yield per segment. No incremental decoding within a segment.
- Each chunk passes through a port of `trimSilence` (edge trim at 0.015, 8 ms guard, 150 ms gap appended), as the baselines do, so first chunk audio and RTF compare.
- Cancellation checkpoint: between segments only. The host stops pulling the generator; an active segment's MLX inference runs to completion.
- Load (`process.loadMs`): `load_model` on the local snapshot plus the misaki G2P pipeline, which mlx-audio otherwise builds inside the first `generate`. mlx-audio reloads the voice pack inside every `generate` call; that cost is inside every run.
- Full mode: the whole text in one `generate` call without a split pattern (misaki still splits past 510 phonemes); one chunk at the end.
