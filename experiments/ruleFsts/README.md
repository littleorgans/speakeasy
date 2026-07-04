# Archived experiment: sherpa ruleFsts rewrite arm

This is the **Path B Arm 2** experiment, kept for reference. It is **not** a
production path. The shipped rewrite is the `withRewrite` decorator over the
in-house map + number pipeline in `src/rewrite/` (see MODEL-SWEEP.md for why the
map won the head-to-head).

`build-fst.py` compiles a byte-level OpenFst rewrite (`replace.fst`) from the
production ruleset (`src/rewrite/rules.json`) using `kaldifst` (build-time only).
The `.fst` was applied inside the engine via sherpa's `ruleFsts` config to
rewrite recognized text.

The experiment proved `ruleFsts` fire on the streaming/online recognizer
(`"Open crown browser"` -> `"Open chrome browser"` on kroko) and matched the map's
corpus WER (4.3%), but only after emitting case variants to beat byte-level
case-sensitivity, and it carries a build dependency plus a committed binary. The
production decorator gets whole-word + case-insensitive handling for free.

Rebuild (if ever revisited):

    python3 -m venv .venv && ./.venv/bin/pip install kaldifst
    ./.venv/bin/python experiments/ruleFsts/build-fst.py

Its edge (runs in-engine, can stack with sherpa number/date ITN FSTs) is the
reason to keep it documented rather than deleted.
