# Hex Parakeet Unified research runners

Lifted unchanged from speak-easy `tm/stt-ux-research--30b495b8` (bd75011), research #83; this directory and `experiments/stt-ux-timing/` keep the runnable subset. Known limits: `run.py` hardcodes private inputs (corpus index, scorer and model paths on the research Mac), and `evaluate.mjs` needs `scorer.mjs` and `io.mjs` from `tm/known-words-fix--8588000c`, which are not on main. The road test's Hex worker is `packages/browser-voice/runtime/hex/`.
