# fast-tts: shared harness and baselines

Measurement harness for fast local TTS candidates (MLX, CoreML, llama.cpp, PyTorch MPS and others) beside speak-easy's two local voices, Piper Kristin medium and Kokoro v1.0. Not a product path; nothing here is imported by the packages. Every engine follows [CONTRACT.md](CONTRACT.md).

Adapted from the audio8-tts experiment (branch `tm/audio8-tts-research--75e44b1b`, `experiments/audio8-tts/`): cohort wording, the baseline renderer, the timed runner, the summarizer and the listening page builder started there.

## Files

- `CONTRACT.md`: metrics, endpoints, cold and warm, the result JSON schema, clip names.
- `cohort.json`: ten agent to owner texts: five short, three medium, two long.
- `fasttts.py`: Python helper for candidates (stdlib; `DevicePlayer` needs numpy and sounddevice). Records runs, captures the environment, validates and writes result JSON, writes WAV, and holds `derive`, the one implementation of RTF and underruns. `selftest` exercises it with a synthetic engine.
- `render-baselines.ts`: baselines through speak-easy's own path (`TtsSynth` behind `SherpaTextToSpeech`, sentence pipeline, Piper Plus frontend, `cpu` provider, no `onProgress`). Engines `piper-kristin`, `kokoro-heart`, `kokoro-michael`; modes `stream` (sentence pipeline) and `full` (one synth call).
- `run-timed.sh`: the timed pass, one fresh process per engine and temperature.
- `summarize.py`: validates and merges result files into a summary JSON and a markdown table.
- `build_listen.py`: writes the owner listening page from the summary and the clips.

## Setup

From the speak-easy repo root after `pnpm install`. Kristin, Kokoro v1.0 and the Piper Plus frontend must sit in `MODELS` as `vits-piper-en_US-kristin-medium/`, `kokoro-multi-lang-v1_0/` and `piper-plus-g2p-4fd95c24/`; the renderer refuses to run unless speak-easy's readiness check verifies their digests.

```bash
MODELS=~/Dev/LLM/DEV/helioy/speak-easy/models/tts
ASSETS=~/.transport-matters-preview/spaces/e97f8349-0357-4a5b-95ee-60908561af3f/assets/fast-tts
TEMPLATE=<owner-pages skill dir>/template.html
mkdir -p $ASSETS/clips
# Only for DevicePlayer (playback start endpoint) or the selftest with --play:
uv venv --python 3.13 $TMPDIR/ftts-env && uv pip install --python $TMPDIR/ftts-env/bin/python numpy sounddevice
```

## Smoke (untimed, any time)

```bash
SMOKE=1 RUNS=1 COLD_TEXTS=status OUT=$TMPDIR/ftts-smoke CLIPS=$ASSETS/clips experiments/fast-tts/run-timed.sh
python3 -I experiments/fast-tts/summarize.py $TMPDIR/ftts-smoke-summary.json $TMPDIR/ftts-smoke/*.json
python3 -I experiments/fast-tts/build_listen.py $ASSETS --summary $TMPDIR/ftts-smoke-summary.json --template $TEMPLATE
python3 -I experiments/fast-tts/fasttts.py selftest --out $TMPDIR/ftts-selftest
```

One engine by hand: `node experiments/fast-tts/render-baselines.ts --root $MODELS --engine kokoro-heart --temperature warm --repeats 1 --json $TMPDIR/k.json --clips $ASSETS/clips`.

## Timed pass (quiet machine slot only)

```bash
OUT=$TMPDIR/ftts-timed CLIPS=$ASSETS/clips experiments/fast-tts/run-timed.sh
python3 -I experiments/fast-tts/summarize.py $ASSETS/results.json $TMPDIR/ftts-timed/*.json
python3 -I experiments/fast-tts/build_listen.py $ASSETS --summary $ASSETS/results.json --template $TEMPLATE
```

Defaults: `RUNS=5`, engines `piper-kristin kokoro-heart kokoro-michael`, modes `stream full`, cold texts `status summary handover` (one per length class). The untimed smoke with `RUNS=1` took about 7 minutes; expect roughly 35 for the default timed pass.

## Notes

- First audio for the baselines is the first sentence pipeline segment reaching the caller; the pipeline carves a short first chunk (`planSegments`) and trims segment silence (`trimSilence`), so streamed clips have a fixed 150 ms gap between segments.
- The baselines do not play audio, so their playback start is `null`; `fasttts.DevicePlayer` measures it for Python engines.
- speak-easy sherpa uses 2 threads; `loadMs` includes speak-easy's payload digest check, with the native load in `loadDetail.nativeLoadMs`.
