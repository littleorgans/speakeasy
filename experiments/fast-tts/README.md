# fast-tts: shared harness and baselines

Measurement harness for fast local TTS candidates (MLX, CoreML, llama.cpp, PyTorch MPS and others) beside speak-easy's two local voices, Piper Kristin medium and Kokoro v1.0. Not a product path; nothing here is imported by the packages. Every engine follows [CONTRACT.md](CONTRACT.md).

Adapted from the audio8-tts experiment (branch `tm/audio8-tts-research--75e44b1b`, `experiments/audio8-tts/`): cohort wording, the baseline renderer, the timed runner, the summarizer and the listening page builder started there.

## Files

- `CONTRACT.md`: metrics, endpoints, cold and warm, the result JSON schema, clip names.
- `cohort.json`: ten agent to owner texts: five short, three medium, two long.
- `fasttts.py`: Python helper for candidates (stdlib; `DevicePlayer` needs numpy and sounddevice). `run_engine` is the shared engine script runner (the flags of `render-baselines.ts`, load timing, warmup, cold and warm loops, clips): a candidate implements `Engine` and calls it. Below it, `Result` and `Run` record runs; the module also captures the environment, verifies weight digests, validates and writes result JSON, writes WAV, and holds `derive`, the one implementation of RTF and underruns. `selftest` exercises it with a synthetic engine.
- `driver.ts`: the TypeScript twin of `fasttts.run_engine` (`runCandidate`: flags, cold and warm loops, clips, result JSON) and a client for native workers that stream PCM over stdio; `render-baselines.ts` and the native candidates supply only an `Adapter`.
- `render-baselines.ts`: baselines through speak-easy's own path (`TtsSynth` behind `SherpaTextToSpeech`, sentence pipeline, Piper Plus frontend, `cpu` provider, no `onProgress`). Engines `piper-kristin`, `kokoro-heart`, `kokoro-michael` at the product default threads, plus `piper-kristin-t4` and `kokoro-heart-t4` at 4 threads; modes `stream` (sentence pipeline) and `full` (one synth call).
- `candidates/`: one directory per candidate with its `render.py` and a README (pins, licences, setup, stream policy): `kokoro-mlx` (engines `kokoro-mlx-heart`, `kokoro-mlx-michael`) and `pocket-18m` (`pocket-18m-bill`) in Python; `kokoro-coreml`, `supertonic3` and `piper-split` in TypeScript ([candidates/NATIVE.md](candidates/NATIVE.md)).
- `run-timed.sh`: the timed pass, one fresh process per engine and temperature; its `render` function maps each engine to its script.
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

Candidates need their weights in `MODELS` and their Python envs, `$TMPDIR/venv-kokoro-mlx` and `$TMPDIR/venv-pocket` (override with `KOKORO_MLX_PY`, `POCKET_PY`), and the native artefacts in `$TMPDIR` named in NATIVE.md; the commands are in each candidate README.

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

Defaults: `RUNS=5`, engines `piper-kristin piper-kristin-t4 kokoro-heart kokoro-heart-t4 kokoro-michael kokoro-mlx-heart kokoro-mlx-michael pocket-18m-bill kokoro-coreml-heart kokoro-coreml-michael supertonic3-f1 supertonic3-m1 supertonic3-f1-t4 piper-split-t4 piper-split-carved-t4`, modes `stream full`, cold texts `status summary handover` (one per length class). Engines without `-t4` use speak-easy's product default thread count (`DEFAULT_NUM_THREADS` in `synth.ts`, 2); `-t4` engines use 4 threads, and each label names its count. Estimate, from smoke RTFs and about 145 s of cohort audio per pass: about 45 minutes, up to an hour if the machine is busy. Kokoro dominates: each 2 thread Kokoro engine takes about 12 minutes for both modes. With the ten candidate engines, the full pass is an estimated 80 minutes, up to 100 if the machine is busy: the `RUNS=1` smoke of all fifteen engines with one cold text took 11.4 minutes, and `RUNS=5` with three cold texts multiplies the warm synthesis by five and the cold processes by fifteen.

## Baseline first audio versus voice-out g2p-F10

Voice-out finding g2p-F10 (research `tracks/voice-out`, `experiments/g2p-bench.cjs`) reports Kristin warm first PCM of 22.1 ms for u01 ("Done.") and 43.0 ms for u10's first sentence ("I stopped the deploy."), at 4 threads; the same finding gives 32.5 and 66.5 ms at 2 threads. The first E1 smoke gave Kristin a 115 ms short median. A probe (untimed, same M2 Max, sherpa-onnx-node 2.0.1, Piper Plus frontend 4fd95c24, load average 8 to 23 from other work) explains the gap. The harness itself adds no defect:

| Factor | Effect | Evidence |
| --- | --- | --- |
| First chunk text | Largest structural factor. Synth time scales with audio length (about 0.05 x audio at 2 threads). "Done." is 0.63 s of audio; the harness's first chunks are `planSegments` heads of up to 4 words, such as "The deploy finished and" at 1.35 s, the size of u10's first sentence. | Direct `TtsSynth.synth`, 2 threads, median of 9: "Done." 31.8 ms, "I stopped the deploy." 65.5 ms, "The deploy finished and" 69.8 ms |
| Threads | speak-easy's product default is 2; the g2p-F10 headline is 4. | The 2 thread direct synth reproduces g2p-F10's own 2 thread figures within 1 ms |
| Machine load | The first smoke ran under heavy load. A rerun of the same engine and texts later gave Kristin 74 ms (was 115) and Kokoro af_heart 688 ms (was 1178). | Smoke reruns, untimed |
| Frontend per call | Negligible: the G2P loads once per synth, as in g2p-bench. | `frontend.request` 0.07 to 0.18 ms |
| Pipeline | Under 1 ms. | Segment `readyAtMs` 85.4 ms vs caller receive 85.8 ms; dispatching the next chunk 0.2 ms |
| Clock endpoint | The same: monotonic clock, submit to PCM returned by `generateAsync`. The harness adds `trimSilence` and one generator hop, both inside the pipeline figure. | Code read of both |
| sherpa, frontend | The same: 2.0.1, Piper Plus 4fd95c24. Node 25.9.0 here vs 26.13.0 there, not isolated. | Result environments |

So the short class median (five texts, first chunks of 2 to 4 words) sits above u01's single word by design, and the thread comparison needs the quiet slot: under this load 4 threads flipped between runs, slower than 2 in the probe ("Done." 74 vs 32 ms) and faster in the smoke rerun (61 vs 74 ms). RTF also differs by definition: g2p-F10 divides one call's time by raw audio including model padding, while this contract divides submit to last chunk by trimmed audio with 150 ms gaps.

## Notes

- First audio for the baselines is the first sentence pipeline segment reaching the caller; the pipeline carves a short first chunk (`planSegments`) and trims segment silence (`trimSilence`), so streamed clips have a fixed 150 ms gap between segments.
- A synth started while another runs on the same `OfflineTts` is slower (one probe: 77 to 103 ms). The harness drains every chunk, so its runs never overlap; a cancelled stream does leave its lookahead synth running.
- The baselines do not play audio, so their playback start is `null`; `fasttts.DevicePlayer` measures it for Python engines.
- `loadMs` includes speak-easy's payload digest check, with the native load in `loadDetail.nativeLoadMs`.
