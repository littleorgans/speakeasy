"""Write listen.html beside a clips/ directory: one row per cohort text, engines side by side.

Usage: python3 -I experiments/audio8-tts/build_listen.py <assets/audio8-tts>
Lists only clips that exist. No scores: the owner judges by ear.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

COLUMNS = [
    ("Piper Kristin", "Speak-easy default local voice, sherpa cpu, Piper Plus frontend",
     [("kristin", "Kristin")]),
    ("Kokoro v1.0", "Speak-easy local voice, sherpa cpu, Piper Plus frontend",
     [("kokoro-heart", "af_heart")]),
    ("Audio8", "Realistic Air on PyTorch MPS; Preview builds on upstream ONNX CPU runtimes. Voices are cloned from the reference clips above", [
        ("a8-air-demo", "Realistic Air 0.6B · demo voice"),
        ("a8-air-kristin", "Realistic Air 0.6B · cloned Kristin"),
        ("a8-06b-demo", "Preview 0.6B INT4 · demo voice"),
        ("a8-06b-demo-stream", "Preview 0.6B INT4 · demo voice · streamed"),
        ("a8-06b-kristin", "Preview 0.6B INT4 · cloned Kristin"),
        ("a8-06b-heart", "Preview 0.6B INT4 · cloned Heart"),
        ("a8-01b-demo", "Preview 0.1B INT8 · demo voice"),
        ("a8-01b-demo-stream", "Preview 0.1B INT8 · demo voice · streamed"),
        ("a8-01b-kristin", "Preview 0.1B INT8 · cloned Kristin"),
        ("a8-01b-bundled", "Preview 0.1B INT8 · bundled Chinese reference"),
    ]),
]

REFERENCES = [
    ("reference--kristin.wav", "Kristin reference (Piper render)"),
    ("reference--kokoro-heart.wav", "Heart reference (Kokoro render)"),
    ("reference--upstream-demo.mp3", "Demo reference (upstream Audio8 demo page, cross_enref_zh_04)"),
]

STYLE = """
body{background:#111418;color:#dde3ea;font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}p.note{color:#8b96a3;margin:0 0 20px}
table{border-collapse:collapse;width:100%}th,td{border-top:1px solid #262c33;padding:10px;vertical-align:top;text-align:left}
th{color:#aab4bf;font-weight:600}th small{display:block;color:#6f7a86;font-weight:400}
td.text{width:24%;color:#f2f5f8}td.text small{display:block;color:#6f7a86;margin-top:4px}
.clip{margin-bottom:8px}.clip span{display:block;font-size:12px;color:#8b96a3}audio{width:100%;max-width:320px;height:32px}
.refs{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:24px}
"""


def clip(path: Path, name: str, label: str) -> str:
    if not (path / name).is_file():
        return ""
    return (f'<div class="clip"><span>{html.escape(label)}</span>'
            f'<audio controls preload="none" src="clips/{html.escape(name)}"></audio></div>')


def main() -> None:
    root = Path(sys.argv[1])
    clips = root / "clips"
    cohort = json.loads((Path(__file__).parent / "cohort.json").read_text())
    head = "".join(f"<th>{html.escape(t)}<small>{html.escape(n)}</small></th>" for t, n, _ in COLUMNS)
    rows = []
    for item in cohort["texts"]:
        cells = "".join(
            "<td>" + "".join(clip(clips, f"{item['id']}--{key}.wav", label) for key, label in variants) + "</td>"
            for _, _, variants in COLUMNS
        )
        rows.append(f'<tr><td class="text">{html.escape(item["text"])}<small>{html.escape(item["kind"])}</small></td>{cells}</tr>')
    refs = "".join(clip(clips, name, label) for name, label in REFERENCES)
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Audio8 listening page</title>'
        f"<style>{STYLE}</style></head><body><h1>Audio8 beside Piper Kristin and Kokoro</h1>"
        '<p class="note">Same texts for every engine. Rendered locally on this Mac. No scores; judge by ear.</p>'
        f'<h2 style="font-size:15px">Cloning references</h2><div class="refs">{refs}</div>'
        f'<table><tr><th>Text</th>{head}</tr>{"".join(rows)}</table></body></html>\n'
    )
    (root / "listen.html").write_text(page)
    print(f"wrote {root / 'listen.html'}")


if __name__ == "__main__":
    main()
