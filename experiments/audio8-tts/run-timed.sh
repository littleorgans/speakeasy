#!/usr/bin/env bash
# Timed runs: each engine in a fresh process, one warmup, RUNS passes per text.
# Run from the speak-easy repo root on a quiet machine. Writes timing JSON and
# logs to OUT (clips there are throwaway; the listening clips stay untouched).
# STREAM_ONLY=1: only the two ONNX routes, streamed, on the longest and one short text.
set -euo pipefail

RUNS="${RUNS:-3}"
OUT="${OUT:?set OUT to a scratch directory}"
MODELS="${MODELS:-$HOME/Dev/LLM/DEV/helioy/speak-easy/models/tts}"
CLONE="${CLONE:-$HOME/Dev/research-clones/Audio8_TTS-1d64afcc}"
ENVS="${ENVS:-$TMPDIR}"
HERE=experiments/audio8-tts
DEMO_TEXT="Our meeting has moved to Thursday afternoon, so please update the calendar before lunch."
mkdir -p "$OUT"
A8_ARGS=(--voice demo --mode full stream --runs "$RUNS" --out "$OUT")
[[ "${STREAM_ONLY:-}" == 1 ]] && A8_ARGS=(--voice demo --mode stream --text paragraph status --runs "$RUNS" --out "$OUT")

[[ "${STREAM_ONLY:-}" == 1 ]] || node "$HERE/render-baselines.ts" --root "$MODELS" --out "$OUT" --json "$OUT/baselines.json" --runs "$RUNS" \
  > "$OUT/baselines.log" 2>&1

"$ENVS/a8-env-06b/bin/python" -I "$HERE/audio8_bench.py" render --runtime-dir "$CLONE/onnx_runtime" \
  --model-dir "$MODELS/audio8-preview-0.6b-onnx-int4" --voices-dir "$MODELS/audio8-voices/preview-0.6b-int4" \
  --label a8-06b "${A8_ARGS[@]}" --json "$OUT/a8-06b.json" \
  > "$OUT/a8-06b.log" 2>&1

"$ENVS/a8-env-01b/bin/python" -I "$HERE/audio8_bench.py" render --runtime-dir "$CLONE/onnx_runtime_0_1b_int8" \
  --model-dir "$MODELS/audio8-0.1b-onnx-int8" --voices-dir "$MODELS/audio8-voices/0.1b-int8" \
  --label a8-01b "${A8_ARGS[@]}" --json "$OUT/a8-01b.json" \
  > "$OUT/a8-01b.log" 2>&1

[[ "${STREAM_ONLY:-}" == 1 ]] || HF_HOME="$ENVS/a8-hf" HF_HUB_OFFLINE=1 "$ENVS/a8-env-air/bin/python" -I "$HERE/audio8_torch.py" \
  --model-dir "$MODELS/audio8-realistic-air-0.6b" \
  --voice demo "$CLONE/docs/audio/cases/reference/cross_enref_zh_04.mp3" "$DEMO_TEXT" \
  --device mps --runs "$RUNS" --out "$OUT" --json "$OUT/a8-air.json" > "$OUT/a8-air.log" 2>&1

echo "timed runs done: $OUT"
