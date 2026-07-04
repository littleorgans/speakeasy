#!/usr/bin/env python3
"""Build src/rewrite/replace.fst from the shared ruleset (src/rewrite/rules.json).

Path B "fst" arm of the post-decode rewrite experiment. Build-time only:
kaldifst is a dev dependency, the runtime (sherpa-onnx-node) just consumes the
compiled binary .fst via ruleFsts. Reproducible:

    python3 -m venv .venv && ./.venv/bin/pip install kaldifst
    ./.venv/bin/python scripts/build-fst.py

The FST is a byte-level transducer: a passthrough self-loop over printable
ASCII (weight 1) plus one zero-weight path per rule, so shortest-path prefers
rewrites. Two byte-level limitations vs the in-house regex map, handled/noted
here: (1) case sensitivity — we emit lowercase AND title-case variants of each
`from` because the recognizer capitalizes words ("Little Organs"); (2) no word
boundaries — matches are substring-level (fine on this corpus; a general
deployment would over-fire, same brittleness the map flags via overTrigger).
"""

import json
from pathlib import Path

import kaldifst

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "src" / "rewrite" / "rules.json"
OUT = ROOT / "src" / "rewrite" / "replace.fst"


def case_variants(text: str) -> list[str]:
    """Lowercase as-authored plus title-case per word (recognizer capitals)."""
    title = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    return list(dict.fromkeys([text, title]))


def rule_path(state0: int, next_free: int, src: str, dst: str) -> tuple[list[str], int]:
    """Zero-weight path consuming bytes(src), emitting bytes(dst), back to state0."""
    si = [ord(c) for c in src]
    di = [ord(c) for c in dst]
    n = max(len(si), len(di))
    si += [0] * (n - len(si))  # 0 = epsilon
    di += [0] * (n - len(di))
    lines: list[str] = []
    cur = state0
    ns = next_free
    for k in range(n):
        nxt = state0 if k == n - 1 else ns
        lines.append(f"{cur} {nxt} {si[k]} {di[k]} 0")
        if k != n - 1:
            ns += 1
        cur = nxt
    return lines, ns


def main() -> None:
    rules = json.loads(RULES.read_text())
    lines = [f"0 0 {b} {b} 1" for b in range(32, 127)]  # printable passthrough
    next_free = 1
    pairs = 0
    for rule in rules:
        for variant in case_variants(rule["from"]):
            path, next_free = rule_path(0, next_free, variant, rule["to"])
            lines.extend(path)
            pairs += 1
    lines.append("0 0")  # state 0 is final
    fst = kaldifst.compile(s="\n".join(lines) + "\n", acceptor=False)
    fst.write(str(OUT))
    print(f"wrote {OUT} from {len(rules)} rules ({pairs} case variants)")


if __name__ == "__main__":
    main()
