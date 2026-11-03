#!/usr/bin/env bash
# kokoro-ane timed pass, ane-check's method. Needs a quiet slot granted by the Space lead.
# Idle, then under continuous Llama 3.2 3B generation (mlx_lm, started 20 s before the load pass):
#   run-timed.sh, stream, RUNS=5: cold status/summary/handover x5 and warm cohort x5 per engine;
#   KokoroAneProbe hybrid, folded and cpu: whole text per call with process CPU, 8 warm reps.
# Primes each engine once first (untimed): Core ML compiles for the ANE once per binary, and a fresh
# venv compiles its bytecode on first import (README.md).
# Samples the top 6 processes by CPU each minute; phases.txt stamps each engine's start for LLM tokens/s per phase. No playback; clips stay unwritten.
# Run from the speak-easy repo root. Output: $BASE/timed/{idle,load,llm.jsonl,background.txt}
set -euo pipefail
BASE="${BASE:-$TMPDIR/kokoro-ane}"
MODELS="${MODELS:-$HOME/Dev/LLM/DEV/helioy/speak-easy/models/tts}"
ENGINES="${ENGINES:-kokoro-mlx-heart kokoro-coreml-heart kokoro-ane-heart kokoro-ane-folded-heart}"
LLM_PY=$(head -1 "$HOME/.local/bin/mlx_lm.generate" | cut -c3-)
LLM_MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/*)
HERE=experiments/kokoro-ane
export KOKORO_MLX_PY="$BASE/venv-kokoro-mlx/bin/python"
export KOKORO_ANE_WORKER="$BASE/build-fluid/release/KokoroAneWorker"
export KOKORO_COREML_WORKER="$BASE/build-worker/release/KokoroCoreMLWorker"
PROBE="$BASE/build-worker/release/KokoroAneProbe"
OUT="$BASE/timed"
mkdir -p "$OUT"

for engine in $ENGINES; do
  OUT="$OUT/prime/$engine" RUNS=1 MODES=stream COLD_TEXTS=status SMOKE=1 ENGINES="$engine" MODELS="$MODELS" \
    bash experiments/fast-tts/run-timed.sh > /dev/null
done
for v in hybrid folded cpu; do
  "$PROBE" "$MODELS/kokoro-coreml-mattmireles-034f7741" "$v" 0 experiments/fast-tts/cohort.json "$OUT/prime/probe-$v.json" status 2> /dev/null
done

(while :; do { date -u +%FT%TZ; uptime; ps -Aceo pcpu,pmem,comm -r | head -7; echo; } >> "$OUT/background.txt"; sleep 60; done) &
SAMPLER=$!
trap 'kill $SAMPLER ${LLM:-} 2>/dev/null || true' EXIT

pass() {
  local out="$OUT/$1"
  mkdir -p "$out/probe"
  uptime > "$out/uptime.txt"
  for engine in $ENGINES; do
    echo "$(date -u +%FT%TZ) start $engine" >> "$OUT/phases.txt"
    RUNS=5 MODES=stream ENGINES="$engine" OUT="$out/runs" MODELS="$MODELS" \
      bash experiments/fast-tts/run-timed.sh >> "$out/run-timed.log" 2>&1
  done
  for v in hybrid folded cpu; do
    echo "$(date -u +%FT%TZ) start probe-$v" >> "$OUT/phases.txt"
    "$PROBE" "$MODELS/kokoro-coreml-mattmireles-034f7741" "$v" 8 experiments/fast-tts/cohort.json "$out/probe/$v.json" \
      status summary > "$out/probe/$v.out" 2> "$out/probe/$v.err"
  done
  echo "$(date -u +%FT%TZ) end $1" >> "$OUT/phases.txt"
  uptime >> "$out/uptime.txt"
  python3 -I experiments/fast-tts/summarize.py "$out/summary.json" "$out"/runs/*.json > "$out/table.md"
}
pass idle
HF_HUB_OFFLINE=1 "$LLM_PY" -I "$HERE/tools/llm_load.py" "$LLM_MODEL" 3600 "$OUT/llm.jsonl" > "$OUT/llm.err" 2>&1 &
LLM=$!
echo "$(date -u +%FT%TZ) start llm" >> "$OUT/phases.txt"
sleep 20  # model load and first generations
pass load
echo "kokoro-ane timed done: $OUT"
