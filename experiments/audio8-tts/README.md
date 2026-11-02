# Audio8 TTS beside Piper Kristin and Kokoro

Local experiment: run [Audio8 TTS](https://github.com/Edge0-AI/Audio8_TTS) on an Apple Silicon Mac, render the same short agent lines with Audio8, Piper Kristin and Kokoro v1.0, and time them. Not a product path; nothing here is imported by the packages.

## Files

- `cohort.json`: seven agent to owner lines plus the reference line spoken to make cloning references.
- `render-baselines.ts`: Kristin and Kokoro `af_heart` through speak-easy's own path (`TtsSynth` behind `SherpaTextToSpeech`, sentence pipeline, Piper Plus frontend, `cpu` provider, no `onProgress`).
- `audio8_bench.py`: registers cloned voices and renders or times the cohort through upstream's ONNX CPU runtimes (Preview 0.6B INT4, Preview 0.1B INT8), full and streamed.
- `audio8_torch.py`: renders or times Realistic Air 0.6B through PyTorch (no ONNX export exists), mirroring upstream `audio8_tts_infer.py`.
- `build_listen.py`: writes `listen.html` beside a `clips/` directory.
- `run-timed.sh`: the timed pass, one fresh process per engine.
- `summarize.py`: merges timing JSON into one results file and a markdown table.

## Pins

| Item | Pin | Licence |
| --- | --- | --- |
| Audio8 code | `Edge0-AI/Audio8_TTS` at `1d64afcc2da56222b41ddcf53de7b8bc54be1fb6` | Apache 2.0 |
| Preview 0.6B ONNX INT4 | `Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4` at `818569c6b832118ad68d61bbd873abe250fcd68a` | Apache 2.0 |
| Preview 0.1B ONNX INT8 | `Edge0/audio8-TTS-0.1B-ONNX-INT8` at `317c12d4e0da83847b594fcf8bd74bf2c76615ec` | Card says Apache 2.0; its base checkpoint uses the Audio8 Community License v1.0 (commercial use only below US$2M annual revenue) |
| Realistic Air 0.6B | `Edge0/Audio8-TTS-Realistic-Air` at `23c305cc013fde378edb459f538317043c993a74` | Apache 2.0 |
| Kristin, Kokoro v1.0 | speak-easy `LOCAL_TTS_MODELS` payload digests | see `packages/speech-io/src/tts/catalogue.ts` |

## Setup

From the speak-easy repo root after `pnpm install`. Weights live outside git, in `MODELS`; Python envs live in `$TMPDIR`.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
CLONE=~/Dev/research-clones/Audio8_TTS-1d64afcc

# Code, pinned shallow clone
git init -q $CLONE && git -C $CLONE remote add origin https://github.com/Edge0-AI/Audio8_TTS.git
git -C $CLONE fetch -q --depth 1 origin 1d64afcc2da56222b41ddcf53de7b8bc54be1fb6 && git -C $CLONE checkout -q FETCH_HEAD

# Weights
hf download Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4 --revision 818569c6b832118ad68d61bbd873abe250fcd68a --local-dir $MODELS/audio8-preview-0.6b-onnx-int4
hf download Edge0/audio8-TTS-0.1B-ONNX-INT8 --revision 317c12d4e0da83847b594fcf8bd74bf2c76615ec --local-dir $MODELS/audio8-0.1b-onnx-int8
hf download Edge0/Audio8-TTS-Realistic-Air --revision 23c305cc013fde378edb459f538317043c993a74 --local-dir $MODELS/audio8-realistic-air-0.6b

# Python envs (the two ONNX runtimes pin incompatible onnxruntime ranges)
uv venv --python 3.12 $TMPDIR/a8-env-06b && uv pip install --python $TMPDIR/a8-env-06b/bin/python -r $CLONE/onnx_runtime/requirements.txt tokenizers==0.22.2
uv venv --python 3.12 $TMPDIR/a8-env-01b && uv pip install --python $TMPDIR/a8-env-01b/bin/python -r $CLONE/onnx_runtime_0_1b_int8/requirements.txt
uv venv --python 3.12 $TMPDIR/a8-env-air && uv pip install --python $TMPDIR/a8-env-air/bin/python -r $CLONE/requirements.txt
```

Kristin and Kokoro v1.0 must sit in `$MODELS` as `vits-piper-en_US-kristin-medium/`, `kokoro-multi-lang-v1_0/` and `piper-plus-g2p-4fd95c24/`; `render-baselines.ts` refuses to run unless speak-easy's readiness check verifies their digests.

## Render

```bash
CLIPS=<assets>/audio8-tts/clips; mkdir -p $CLIPS
node experiments/audio8-tts/render-baselines.ts --root $MODELS --out $CLIPS --json $TMPDIR/baselines.json

# Cloned voices: Kristin and Heart from the baseline reference renders, demo from upstream's demo page
REF="$(python3 -I -c 'import json;print(json.load(open("experiments/audio8-tts/cohort.json"))["reference"]["text"])')"
DEMO="Our meeting has moved to Thursday afternoon, so please update the calendar before lunch."
for spec in "a8-env-06b onnx_runtime audio8-preview-0.6b-onnx-int4 preview-0.6b-int4" "a8-env-01b onnx_runtime_0_1b_int8 audio8-0.1b-onnx-int8 0.1b-int8"; do
  set -- $spec; A="--runtime-dir $CLONE/$2 --model-dir $MODELS/$3 --voices-dir $MODELS/audio8-voices/$4"
  $TMPDIR/$1/bin/python -I experiments/audio8-tts/audio8_bench.py register $A --name kristin --audio $CLIPS/reference--kristin.wav --text "$REF"
  $TMPDIR/$1/bin/python -I experiments/audio8-tts/audio8_bench.py register $A --name heart --audio $CLIPS/reference--kokoro-heart.wav --text "$REF"
  $TMPDIR/$1/bin/python -I experiments/audio8-tts/audio8_bench.py register $A --name demo --audio $CLONE/docs/audio/cases/reference/cross_enref_zh_04.mp3 --text "$DEMO"
done
$TMPDIR/a8-env-01b/bin/python -I experiments/audio8-tts/audio8_bench.py bundled --runtime-dir $CLONE/onnx_runtime_0_1b_int8 --model-dir $MODELS/audio8-0.1b-onnx-int8 --voices-dir $MODELS/audio8-voices/0.1b-int8 --name bundled

R06="--runtime-dir $CLONE/onnx_runtime --model-dir $MODELS/audio8-preview-0.6b-onnx-int4 --voices-dir $MODELS/audio8-voices/preview-0.6b-int4 --label a8-06b --out $CLIPS"
$TMPDIR/a8-env-06b/bin/python -I experiments/audio8-tts/audio8_bench.py render $R06 --voice demo kristin heart --mode full --json $TMPDIR/a8-06b-full.json
$TMPDIR/a8-env-06b/bin/python -I experiments/audio8-tts/audio8_bench.py render $R06 --voice demo --mode stream --json $TMPDIR/a8-06b-stream.json
R01="--runtime-dir $CLONE/onnx_runtime_0_1b_int8 --model-dir $MODELS/audio8-0.1b-onnx-int8 --voices-dir $MODELS/audio8-voices/0.1b-int8 --label a8-01b --out $CLIPS"
$TMPDIR/a8-env-01b/bin/python -I experiments/audio8-tts/audio8_bench.py render $R01 --voice demo kristin bundled --mode full --json $TMPDIR/a8-01b-full.json
$TMPDIR/a8-env-01b/bin/python -I experiments/audio8-tts/audio8_bench.py render $R01 --voice demo --mode stream --json $TMPDIR/a8-01b-stream.json
HF_HOME=$TMPDIR/a8-hf HF_HUB_OFFLINE=1 $TMPDIR/a8-env-air/bin/python -I experiments/audio8-tts/audio8_torch.py --model-dir $MODELS/audio8-realistic-air-0.6b \
  --voice demo $CLONE/docs/audio/cases/reference/cross_enref_zh_04.mp3 "$DEMO" --voice kristin $CLIPS/reference--kristin.wav "$REF" --device mps --out $CLIPS --json $TMPDIR/a8-air.json

cp $CLONE/docs/audio/cases/reference/cross_enref_zh_04.mp3 $CLIPS/reference--upstream-demo.mp3
python3 -I experiments/audio8-tts/build_listen.py $CLIPS/..
```

## Timing

On a quiet machine: `OUT=$TMPDIR/a8-timed RUNS=3 experiments/audio8-tts/run-timed.sh`, then `python3 -I experiments/audio8-tts/summarize.py results.json $TMPDIR/a8-timed/*.json`.

Definitions. Warm runs follow one warmup synth in the same process; cold load is reported separately. TTFA is request to first audio ready: the first sentence segment for the sherpa baselines, the first stream chunk for Audio8 streaming, and the whole synthesis for Audio8 full mode. RTF is request to last audio ready over audio duration. Audio8 samples with a fixed seed (42); upstream ONNX runtimes use 5 threads, speak-easy sherpa uses 2.
