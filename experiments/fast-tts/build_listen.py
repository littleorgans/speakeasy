"""Write the owner listening page: same cohort texts across engines, side by side.

Usage: python3 -I experiments/fast-tts/build_listen.py <assets/fast-tts> --summary <summary.json> --template <owner-pages template.html> [--slot-log <slot.log>]
Clips sit in <assets/fast-tts>/clips as <textId>--<engineId>.wav (stream) or
<textId>--<engineId>--full.wav. Each clip shows warm median first audio and RTF
from the summary. The page reuses the owner pages template's stylesheet and
renderer verbatim, with no decisions. No quality scores: the owner judges by ear.
Adapted from experiments/audio8-tts/build_listen.py (audio8-tts thread).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fasttts as ft  # noqa: E402

BASELINES = ("piper-kristin", "piper-kristin-t4", "kokoro-heart", "kokoro-heart-t4", "kokoro-michael")
POLICY_TITLES = {"streaming": "Streaming", "chunkPolicy": "Chunk policy", "cancellation": "Cancellation"}
LENGTHS = (("short", "Short replies", "One sentence"), ("medium", "Medium replies", "Two or three sentences"),
           ("long", "Long replies", "One paragraph, six or more sentences"))

STYLE = """
.clips { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr)); gap: 16px; }
.clip { display: grid; gap: 8px; align-content: start; padding: 14px 16px; border: 1px solid var(--rule); border-radius: 4px; background: var(--tile); }
.clip b { font-size: 14px; }
.clip audio { width: 100%; height: 32px; }
.say { max-width: var(--measure); color: var(--ink); }
td.num, th.num { white-space: nowrap; }
td:not(.num) { overflow-wrap: anywhere; min-width: 20ch; }
tbody th { min-width: 16ch; }
"""


def slot_status(log: Path) -> str:
    """The timed status line from a slot log's `## start|sample|end <UTC> load { 1m 5m 15m }` lines."""
    marks = re.findall(r"^## (start|sample|end) (\S+) load \{ ([\d.]+)", log.read_text(), re.M)
    start = next(at for kind, at, _ in marks if kind == "start")
    end = next((at for kind, at, _ in marks if kind == "end"), None)
    during = sorted(float(load) for kind, _, load in marks if kind == "sample")
    before = next(float(load) for kind, _, load in marks if kind == "start")
    window = f"{start[:10]}, {start[11:16]} to {end[11:16]} UTC" if end else f"from {start[:16].replace('T', ' ')} UTC"
    load = (f"1 minute load average {before:.1f} before the pass; during it (pass included) median "
            f"{during[len(during) // 2]:.1f}, range {during[0]:.1f} to {during[-1]:.1f} over {len(during)} samples"
            if during else f"1 minute load average {before:.1f} before the pass")
    return f"Timed, quiet slot, {window}. {load}."


def esc(value) -> str:
    return html.escape(str(value))


def template_parts(path: Path) -> tuple[str, str]:
    """The template's stylesheet and its renderer script, untouched."""
    source = path.read_text()
    style = re.search(r"<style>.*?</style>", source, re.S).group(0)
    renderer = re.search(r"<script>\n\(\(\) => \{.*?</script>", source, re.S).group(0)
    return style, renderer


def columns(summary: dict, clips: Path) -> list[dict]:
    """One column per engine and mode with warm results or clips; baselines first."""
    engines = sorted(summary["engines"], key=lambda e: (BASELINES.index(e["engine"]["id"])
                                                         if e["engine"]["id"] in BASELINES else len(BASELINES)))
    cols = []
    for entry in engines:
        for mode in ft.MODES:
            warm = {row["textId"]: row for row in entry["byText"] if row["temperature"] == "warm" and row["mode"] == mode}
            suffix = "--full" if mode == "full" else ""
            if warm or any(clips.glob(f"*--{entry['engine']['id']}{suffix}.wav")):
                cols.append({"entry": entry, "mode": mode, "suffix": suffix, "warm": warm})
    return cols


def clip_card(col: dict, text_id: str, clips: Path) -> str:
    engine = col["entry"]["engine"]
    name = f"{text_id}--{engine['id']}{col['suffix']}.wav"
    label = f"{engine['label']} · {'full' if col['mode'] == 'full' else 'streamed'}"
    row = col["warm"].get(text_id)
    numbers = "no warm numbers"
    if row:
        parts = [f"first audio {row['firstAudioMs']['median']:.0f} ms", f"RTF {row['rtf']['median']:.2f}"]
        if row["playbackStartMs"]:
            parts.insert(1, f"playback start {row['playbackStartMs']['median']:.0f} ms")
        numbers = " · ".join(parts)
    audio = (f'<audio controls preload="none" src="clips/{esc(name)}"></audio>' if (clips / name).is_file()
             else '<span class="fine">no clip</span>')
    return f'<div class="clip"><b>{esc(label)}</b>{audio}<span class="fine">{esc(numbers)}</span></div>'


def engines_table(cols: list[dict]) -> str:
    rows = []
    for col in cols:
        entry, engine = col["entry"], col["entry"]["engine"]
        pooled = {row["length"]: row for row in entry["byLength"] if row["temperature"] == "warm" and row["mode"] == col["mode"]}
        first = " / ".join(f"{pooled[l]['firstAudioMs']['median']:.0f}" if l in pooled else "n/a" for l, _, _ in LENGTHS)
        rtf = " / ".join(f"{pooled[l]['rtf']['median']:.2f}" if l in pooled else "n/a" for l, _, _ in LENGTHS)
        under = max((row["underruns"]["max"] for row in pooled.values()), default=None)
        cold = entry["loadMs"]["cold"]
        rows.append(
            f'<tr><th scope="row">{esc(engine["label"])}<span class="d"> · {esc(col["mode"])}</span></th>'
            f'<td>{esc(engine["runtime"])}, {esc(engine["provider"])}<br><span class="d">{esc(engine["modelRevision"])}</span></td>'
            f'<td class="num">{first}</td><td class="num">{rtf}</td>'
            f'<td class="num">{"n/a" if under is None else f"{under:.0f}"}</td>'
            f'<td class="num">{"n/a" if cold is None else f"{cold['median']:.0f}"}</td></tr>')
    return ('<table><thead><tr><th scope="col">Engine</th><th scope="col">Runtime</th>'
            '<th scope="col" class="num">First audio ms<br>short / med / long</th><th scope="col" class="num">RTF<br>short / med / long</th>'
            '<th scope="col" class="num">Underruns max</th><th scope="col" class="num">Cold load ms</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def policies_table(cols: list[dict]) -> str:
    """The standard optional engine fields (CONTRACT.md), one row per engine."""
    engines = list({col["entry"]["engine"]["id"]: col["entry"]["engine"] for col in cols}.values())
    head = "".join(f'<th scope="col">{esc(POLICY_TITLES[key])}</th>' for key in ft.POLICY_KEYS)
    rows = "".join(f'<tr><th scope="row">{esc(engine["label"])}</th>'
                   + "".join(f'<td>{esc(engine.get(key) or "not stated")}</td>' for key in ft.POLICY_KEYS) + "</tr>"
                   for engine in engines)
    return f'<table><thead><tr><th scope="col">Engine</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="assets/fast-tts directory; listen.html is written here")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--slot-log", type=Path, help="timed pass slot log: window and load for the status line")
    args = parser.parse_args()
    clips = args.root / "clips"
    summary = json.loads(args.summary.read_text())
    style, renderer = template_parts(args.template)
    cols = columns(summary, clips)
    env = summary["engines"][0]["environment"] if summary["engines"] else {}
    if not summary["timed"]:
        status = "Untimed smoke run: numbers are indicative only, not cited results."
    elif args.slot_log:
        status = slot_status(args.slot_log)
    else:
        status = "Timed, quiet slot."

    sections = []
    for length, title, note in LENGTHS:
        groups = []
        for item in (t for t in ft.load_cohort() if t["length"] == length):
            cards = "".join(clip_card(col, item["id"], clips) for col in cols)
            groups.append(f'<div class="group"><h3>{esc(item["kind"])} <span class="unit">{esc(item["id"])}</span></h3>'
                          f'<p class="say">{esc(item["text"])}</p><div class="clips">{cards}</div></div>')
        sections.append(f'<section class="row"><p class="index">{esc(title)}</p><div class="body">'
                        f'<h2>{esc(note)}</h2>{"".join(groups)}</div></section>')

    machine = f"{env.get('machine', 'unknown')}, {env.get('os', '')}, {env.get('memoryGiB', '?')} GiB"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>fast-tts listening page</title>
{style}
<style>{STYLE}</style>
</head>
<body>
<main>
  <header class="masthead">
    <p class="index">fast-tts listening page</p>
    <h1>Same texts, every engine</h1>
    <div class="lead">
      <p class="key"><i>Status</i><span>{esc(status)}</span></p>
      <p><i>Numbers</i><span>Warm medians. First audio: text submit to first decoded samples available to the caller.</span></p>
      <p><i>Order</i><span>Piper Kristin and Kokoro baselines first, then candidates.</span></p>
    </div>
  </header>

  <section id="summary">
    {"".join(sections)}
    <section class="row">
      <p class="index">Engines</p>
      <div class="body">
        <h2>Warm medians per length, pooled over texts</h2>
        {engines_table(cols)}
      </div>
    </section>
    <section class="row">
      <p class="index">Policies</p>
      <div class="body">
        <h2>How each engine streams, chunks and stops</h2>
        {policies_table(cols)}
      </div>
    </section>
    <section class="row">
      <p class="index">Method</p>
      <div class="body">
        <h2>How the numbers were taken</h2>
        <dl class="facts">
          <dt>First audio</dt><dd>Monotonic clock from text submit to the first chunk of decoded samples reaching the caller.</dd>
          <dt>Playback start</dt><dd>Text submit to the first buffer the audio device accepted; shown only where a run played audio.</dd>
          <dt>RTF</dt><dd>Text submit to last chunk ready, over total audio duration. Below 1 is faster than real time.</dd>
          <dt>Underruns</dt><dd>Chunks not ready when gapless playback reaches them.</dd>
          <dt>Warm</dt><dd>One fresh process per engine, one discarded warmup, then repeated passes; cold load is in the engines table.</dd>
          <dt>Machine</dt><dd>{esc(machine)}</dd>
          <dt>Contract</dt><dd>speak-easy experiments/fast-tts/CONTRACT.md at {esc(env.get('harnessCommit') or 'unknown')}</dd>
        </dl>
      </div>
    </section>
  </section>

  <section id="decisions" class="row" aria-label="Decisions">
    <p class="index">Decide</p>
  </section>

  <footer>
    <button type="button" id="copy">Copy answers</button>
    <span id="status" role="status"></span>
    <p class="fine">Built by experiments/fast-tts/build_listen.py. No quality scores; judge by ear.</p>
    <textarea id="answer-text" readonly tabindex="-1" aria-label="Answers to paste to the lead"></textarea>
  </footer>
</main>

<script type="application/json" id="tm-decisions">
{{"schema": "tm.decisions.v1", "page": "fast-tts-listen", "revision": "1", "decisions": []}}
</script>
{renderer}
</body>
</html>
"""
    (args.root / "listen.html").write_text(page)
    print(f"wrote {args.root / 'listen.html'}")


if __name__ == "__main__":
    main()
