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
ENGINES="${ENGINES:-piper-kristin piper-kristin-t4 kokoro-heart kokoro-heart-t4 kokoro-michael}"
MODES="${MODES:-stream full}"
COLD_TEXTS="${COLD_TEXTS:-status summary handover}"  # one per length class
HERE=experiments/fast-tts
TIMED=--timed; [ "${SMOKE:-0}" = 1 ] && TIMED=
mkdir -p "$OUT"

for engine in $ENGINES; do
  for text in $COLD_TEXTS; do
    for i in $(seq 1 "$RUNS"); do
      node "$HERE/render-baselines.ts" --root "$MODELS" --engine "$engine" --temperature cold --text "$text" \
        $TIMED --json "$OUT/$engine--cold--$text--$i.json" >> "$OUT/$engine--cold.log" 2>&1
    done
  done
  for mode in $MODES; do
    node "$HERE/render-baselines.ts" --root "$MODELS" --engine "$engine" --temperature warm --mode "$mode" \
      --repeats "$RUNS" $TIMED ${CLIPS:+--clips "$CLIPS"} --json "$OUT/$engine--warm--$mode.json" \
      > "$OUT/$engine--warm--$mode.log" 2>&1
  done
done

# Candidate engines: add one block per engine here, same shape and file names
# (<engine>--cold--<text>--<i>.json, <engine>--warm--<mode>.json), each a fresh process.

echo "timed runs done: $OUT"
