#!/usr/bin/env bash
# Timed pass (CONTRACT.md): every process is fresh and runs one engine.
#   cold: RUNS processes per cold text, each loads the model and runs one text, no warmup.
#   warm: one process per engine and mode, one discarded warmup, RUNS passes over the cohort.
# Run from the speak-easy repo root on a quiet machine. Writes result JSON and logs to OUT;
# CLIPS, when set, receives the warm clips. SMOKE=1 marks results untimed (timed: false). Then: summarize.py <summary.json> $OUT/*.json
# Adapted from experiments/audio8-tts/run-timed.sh (audio8-tts thread).
set -euo pipefail

RUNS="${RUNS:-5}"
OUT="${OUT:?set OUT to a scratch directory}"
MODELS="${MODELS:-$HOME/Dev/LLM/DEV/helioy/speak-easy/models/tts}"
ENGINES="${ENGINES:-piper-kristin piper-kristin-t4 kokoro-heart kokoro-heart-t4 kokoro-michael kokoro-mlx-heart kokoro-mlx-michael pocket-18m-bill}"
KOKORO_MLX_PY="${KOKORO_MLX_PY:-$TMPDIR/venv-kokoro-mlx/bin/python}"  # candidate envs: candidates/*/README.md
POCKET_PY="${POCKET_PY:-$TMPDIR/venv-pocket/bin/python}"
MODES="${MODES:-stream full}"
COLD_TEXTS="${COLD_TEXTS:-status summary handover}"  # one per length class
HERE=experiments/fast-tts
TIMED=--timed; [ "${SMOKE:-0}" = 1 ] && TIMED=
mkdir -p "$OUT"

# render <engine> <flags>: one fresh process of the engine's script. Every script takes the same flags.
render() {
  local engine=$1
  shift
  case "$engine" in
    kokoro-mlx-*) "$KOKORO_MLX_PY" -I "$HERE/candidates/kokoro-mlx/render.py" --engine "$engine" "$@" ;;
    pocket-18m-*) "$POCKET_PY" -I "$HERE/candidates/pocket-18m/render.py" --engine "$engine" "$@" ;;
    *) node "$HERE/render-baselines.ts" --engine "$engine" "$@" ;;
  esac
}

for engine in $ENGINES; do
  for text in $COLD_TEXTS; do
    for i in $(seq 1 "$RUNS"); do
      render "$engine" --root "$MODELS" --temperature cold --text "$text" \
        $TIMED --json "$OUT/$engine--cold--$text--$i.json" >> "$OUT/$engine--cold.log" 2>&1
    done
  done
  for mode in $MODES; do
    render "$engine" --root "$MODELS" --temperature warm --mode "$mode" \
      --repeats "$RUNS" $TIMED ${CLIPS:+--clips "$CLIPS"} --json "$OUT/$engine--warm--$mode.json" \
      > "$OUT/$engine--warm--$mode.log" 2>&1
  done
done

echo "timed runs done: $OUT"
