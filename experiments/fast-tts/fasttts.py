"""Shared measurement helper for fast-tts engines (see CONTRACT.md).

Import from an engine script run with `python -I` (isolated mode drops the script
directory from sys.path, so add this directory explicitly):

    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # the experiments/fast-tts dir
    import fasttts as ft

Stdlib only. `DevicePlayer` (playback start endpoint) needs numpy and sounddevice.
This module is also the one implementation of the derived metrics: `derive` turns
a raw run into first audio, playback start, RTF and underruns, and summarize.py
uses it for every engine, TypeScript or Python.

A Python engine script defines an `Engine` and calls `run_engine`, which owns the
CLI (the flags of render-baselines.ts), load timing, warmup, the cold and warm
loops and clip writing. `Result` and `Run` stay public for scripts that need
their own loop.

CLI: `python3 -I experiments/fast-tts/fasttts.py selftest --out <dir> [--play]`
writes a synthetic engine's result file to check a setup end to end.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

SCHEMA = "fast-tts.result.v1"
HERE = Path(__file__).resolve().parent
TEMPERATURES = ("cold", "warm")
MODES = ("stream", "full")
WARMUP_TEXT = "Warm up."


def now_ms() -> float:
    """Monotonic clock in milliseconds (perf_counter_ns)."""
    return time.perf_counter_ns() / 1e6


def load_cohort(ids: list[str] | None = None) -> list[dict]:
    texts = json.loads((HERE / "cohort.json").read_text())["texts"]
    if not ids:
        return texts
    unknown = set(ids) - {t["id"] for t in texts}
    if unknown:
        raise SystemExit(f"unknown cohort ids: {sorted(unknown)}")
    return [t for t in texts if t["id"] in ids]


def environment(versions: dict[str, str]) -> dict:
    """Machine, OS and runtime versions. `versions` names the engine's own stack."""

    def sysctl(key: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    mac = platform.mac_ver()[0]
    memory = sysctl("hw.memsize")
    return {
        "machine": sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
        "arch": platform.machine(),
        "cpuCount": os.cpu_count(),
        "memoryGiB": round(int(memory) / 2**30) if memory else None,
        "os": f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}",
        "versions": {"python": platform.python_version(), **versions},
        "harnessCommit": git_commit(),
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Run:
    """One synthesis of one text. Created by `Result.start`, which marks text submit."""

    def __init__(self, text_id: str, mode: str, temperature: str, repeat: int) -> None:
        self.row = {"textId": text_id, "mode": mode, "temperature": temperature, "repeat": repeat,
                    "playbackStartMs": None, "chunks": []}
        self.submit_ms = now_ms()

    def chunk(self, audio_ms: float, text: str | None = None) -> float:
        """Record a chunk of decoded audio now available to the caller; returns its ready time."""
        ready = now_ms() - self.submit_ms
        self.row["chunks"].append({"index": len(self.row["chunks"]), "readyMs": round(ready, 3),
                                   "audioMs": round(audio_ms, 3), "text": text})
        return ready

    def playback_started(self, at_ms: float) -> None:
        """Record the absolute `now_ms()` time the audio device accepted the first buffer."""
        self.row["playbackStartMs"] = round(at_ms - self.submit_ms, 3)


class Result:
    """One process, one engine: collects runs and writes the shared result JSON."""

    def __init__(self, engine: dict, temperature: str, timed: bool, repeats: int,
                 versions: dict[str, str], playback: str | None = None) -> None:
        if temperature not in TEMPERATURES:
            raise ValueError(f"temperature must be one of {TEMPERATURES}")
        self.engine = engine
        self.temperature = temperature
        self.process = {"temperature": temperature, "timed": timed, "repeats": repeats,
                        "startedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "clock": "time.perf_counter_ns (monotonic)", "playback": playback,
                        "loadMs": None, "warmupMs": None, "loadDetail": {}}
        self.versions = versions
        self.runs: list[Run] = []

    def start(self, text_id: str, mode: str, repeat: int) -> Run:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        run = Run(text_id, mode, self.temperature, repeat)
        self.runs.append(run)
        return run

    def document(self) -> dict:
        doc = {"schema": SCHEMA, "engine": self.engine, "environment": environment(self.versions),
               "process": self.process, "runs": [run.row for run in self.runs]}
        validate(doc)
        return doc

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.document(), indent=1) + "\n")


ENGINE_KEYS = ("id", "label", "runtime", "provider", "model", "modelRevision", "voice", "sampleRate")
POLICY_KEYS = ("streaming", "chunkPolicy", "cancellation")  # standard optional engine fields, free text


def validate(doc: dict) -> None:
    """Raise ValueError when a result document breaks the contract."""
    problems = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    engine = doc.get("engine", {})
    problems += [f"engine.{key} missing" for key in ENGINE_KEYS if engine.get(key) in (None, "")]
    if not str(engine.get("id", "")).replace("-", "").replace("_", "").replace(".", "").isalnum():
        problems.append("engine.id must be letters, digits, '-', '_' or '.' (it names clip files)")
    problems += [f"engine.{key} must be non-empty text" for key in POLICY_KEYS
                 if key in engine and not (isinstance(engine[key], str) and engine[key])]
    env = doc.get("environment", {})
    problems += [f"environment.{key} missing" for key in ("machine", "os", "versions") if not env.get(key)]
    process = doc.get("process", {})
    if process.get("temperature") not in TEMPERATURES:
        problems.append("process.temperature must be cold or warm")
    if not isinstance(process.get("loadMs"), (int, float)):
        problems.append("process.loadMs missing")
    if process.get("temperature") == "warm" and not isinstance(process.get("warmupMs"), (int, float)):
        problems.append("process.warmupMs missing on a warm process")
    runs = doc.get("runs") or []
    if not runs:
        problems.append("no runs")
    for run in runs:
        where = f"run {run.get('textId')}/{run.get('mode')}/{run.get('repeat')}"
        if run.get("mode") not in MODES or run.get("temperature") != process.get("temperature"):
            problems.append(f"{where}: bad mode or temperature")
        chunks = run.get("chunks") or []
        if not chunks:
            problems.append(f"{where}: no chunks")
        ready = [c.get("readyMs") for c in chunks]
        if any(not isinstance(r, (int, float)) for r in ready) or ready != sorted(ready):
            problems.append(f"{where}: chunk readyMs missing or not ascending")
        if any(not isinstance(c.get("audioMs"), (int, float)) or c["audioMs"] <= 0 for c in chunks):
            problems.append(f"{where}: chunk audioMs missing or not positive")
    if problems:
        raise ValueError("; ".join(problems))


def derive(run: dict) -> dict:
    """Derived metrics for one raw run; the single definition every engine shares.

    Playback is simulated from the chunk list: it starts at the measured playback
    start, else at the first chunk's ready time, and plays chunks back to back.
    A chunk not ready when the previous one ends is an underrun; playback stalls
    until it is ready.
    """
    chunks = run["chunks"]
    first_audio = chunks[0]["readyMs"]
    synth_wall = chunks[-1]["readyMs"]
    audio = sum(c["audioMs"] for c in chunks)
    cursor = run["playbackStartMs"] if run["playbackStartMs"] is not None else first_audio
    underruns, stall = 0, 0.0
    for chunk in chunks:
        if chunk["readyMs"] > cursor:
            underruns += 1
            stall += chunk["readyMs"] - cursor
            cursor = chunk["readyMs"]
        cursor += chunk["audioMs"]
    return {"firstAudioMs": first_audio, "playbackStartMs": run["playbackStartMs"], "synthWallMs": synth_wall,
            "audioMs": audio, "rtf": synth_wall / audio, "chunks": len(chunks),
            "underruns": underruns, "stallMs": stall}


def write_wav(path: Path, samples, sample_rate: int) -> None:
    """Mono 16 bit PCM WAV from float samples in [-1, 1] (any iterable or numpy array)."""
    pcm = array.array("h", (int(max(-1.0, min(1.0, float(s))) * 32767) for s in samples))
    if sys.byteorder == "big":
        pcm.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_digests(expected: Iterable[tuple[Path, str]]) -> None:
    """Exit unless every file's sha256 matches; call off the clock, before load."""
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise SystemExit(f"{path}: sha256 mismatch, expected {digest}")


class Engine(Protocol):
    """One engine as `run_engine` drives it. Construct off the clock (digest checks)."""

    sample_rate: int

    def load(self) -> dict:
        """Load the model until it accepts text; returns `process.loadDetail`. Timed as `loadMs`."""

    def describe(self) -> tuple[dict, dict[str, str]]:
        """After load: the engine fields except `id` (CONTRACT.md, with POLICY_KEYS), and the stack versions."""

    def stream(self, text: str) -> Iterable[tuple[Any, str | None]]:
        """Each chunk as soon as the caller has it: (float samples, chunk text or None)."""

    def full(self, text: str) -> Any:
        """All float samples from one call."""


def run_engine(description: str, engines: Iterable[str], build: Callable[[str, Path], Engine]) -> None:
    """The engine script CLI: one engine, one temperature and mode per process (CONTRACT.md).

    Cold loads and runs one --text with no warmup; warm loads, discards one
    WARMUP_TEXT synthesis, then runs --repeats passes over the texts. The first
    repeat of each text is written to --clips. `build(engine_id, root)` returns
    the engine, not yet loaded.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=list(engines), required=True)
    parser.add_argument("--temperature", choices=TEMPERATURES, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd() / "models" / "tts")
    parser.add_argument("--mode", choices=MODES, default="stream")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--text", action="append")
    parser.add_argument("--clips", type=Path)
    parser.add_argument("--timed", action="store_true")
    args = parser.parse_args()
    texts = load_cohort(args.text or (["status"] if args.temperature == "cold" else None))
    if args.temperature == "cold" and len(texts) != 1:
        raise SystemExit("cold takes exactly one --text")
    repeats = 1 if args.temperature == "cold" else args.repeats

    engine = build(args.engine, args.root)
    start = now_ms()
    load_detail = engine.load()
    load_ms = now_ms() - start
    fields, versions = engine.describe()
    result = Result({"id": args.engine, **fields}, args.temperature, args.timed, repeats, versions)
    result.process["loadMs"] = round(load_ms, 3)
    result.process["loadDetail"] = load_detail
    if args.temperature == "warm":
        warm = now_ms()
        engine.full(WARMUP_TEXT)
        result.process["warmupMs"] = round(now_ms() - warm, 3)

    suffix = "--full" if args.mode == "full" else ""
    for item in texts:
        for repeat in range(repeats):
            run = result.start(item["id"], args.mode, repeat)
            clip = measure(engine, run, item["text"])
            if repeat == 0 and args.clips:
                write_wav(args.clips / f"{item['id']}--{args.engine}{suffix}.wav", clip, engine.sample_rate)
    result.write(args.json)
    print(f"{args.engine} {args.temperature} {args.mode}: {len(result.runs)} runs, load {load_ms:.0f} ms -> {args.json}")


def measure(engine: Engine, run: Run, text: str) -> Iterable:
    """Synthesize one text in the run's mode, recording each chunk; returns the clip samples."""
    if run.row["mode"] == "full":
        clip = engine.full(text)
        run.chunk(len(clip) / engine.sample_rate * 1000, text)
        return clip
    pieces = []
    for samples, chunk_text in engine.stream(text):
        run.chunk(len(samples) / engine.sample_rate * 1000, chunk_text)
        pieces.append(samples)
    return itertools.chain.from_iterable(pieces)


class DevicePlayer:
    """Gapless mono player whose callback reports the playback start endpoint.

    `open()` starts the device on silence, off the request path. `begin()` before
    each run resets the endpoint; `write()` queues float32 samples; the first
    callback that hands real samples to the device sets `started_ms` (a `now_ms()`
    value) for `Run.playback_started`. `drain()` waits for the queue to play out.
    """

    def __init__(self, sample_rate: int) -> None:
        import numpy as np
        import sounddevice as sd

        self._np, self._sd, self.sample_rate = np, sd, sample_rate
        self._lock = threading.Lock()
        self._queue: list = []
        self.started_ms: float | None = None
        self._stream = None

    def open(self) -> None:
        self._stream = self._sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                             callback=self._callback)
        self._stream.start()

    def begin(self) -> None:
        with self._lock:
            self.started_ms = None

    def write(self, samples) -> None:
        with self._lock:
            self._queue.append(self._np.asarray(samples, dtype=self._np.float32).reshape(-1))

    def drain(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._queue:
                    break
            time.sleep(0.01)
        time.sleep(0.1)  # let the device play out its last buffer

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _callback(self, out, frames, _time_info, _status) -> None:
        out.fill(0)
        filled = 0
        with self._lock:
            while filled < frames and self._queue:
                head = self._queue[0]
                take = min(frames - filled, head.size)
                out[filled:filled + take, 0] = head[:take]
                filled += take
                if take == head.size:
                    self._queue.pop(0)
                else:
                    self._queue[0] = head[take:]
            if filled and self.started_ms is None:
                self.started_ms = now_ms()


def selftest(out: Path, play: bool) -> None:
    """A synthetic engine: sine chunks per sentence, synthesis paced at RTF 0.2."""
    import math

    rate = 24_000
    engine = {"id": "selftest-sine", "label": "Selftest sine (synthetic)", "runtime": f"python {platform.python_version()}",
              "provider": "cpu", "model": "synthetic", "modelRevision": "none", "voice": "none", "sampleRate": rate}
    player = DevicePlayer(rate) if play else None
    result = Result(engine, "warm", timed=False, repeats=1, versions={},
                    playback="sounddevice callback, first real samples" if play else None)
    start = now_ms()
    if player:
        player.open()
    result.process["loadMs"] = now_ms() - start
    result.process["warmupMs"] = 0.0
    for item in load_cohort(["status", "summary"]):
        sentences = [s for s in item["text"].replace("?", ".").split(". ") if s]
        if player:
            player.begin()
        run = result.start(item["id"], "stream", 0)
        clip = []
        for sentence in sentences:
            seconds = len(sentence.split()) * 0.3
            time.sleep(seconds * 0.2)
            samples = [0.05 * math.sin(2 * math.pi * 440 * i / rate) for i in range(int(seconds * rate))]
            clip += samples
            run.chunk(len(samples) / rate * 1000, sentence)
            if player:
                player.write(samples)
        if player:
            player.drain()
            if player.started_ms is not None:
                run.playback_started(player.started_ms)
        write_wav(out / f"{item['id']}--{engine['id']}.wav", clip, rate)
    if player:
        player.close()
    result.write(out / f"{engine['id']}--warm.json")
    for run in result.runs:
        print(run.row["textId"], {k: round(v, 1) if isinstance(v, float) else v for k, v in derive(run.row).items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["selftest"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--play", action="store_true", help="play through DevicePlayer (numpy, sounddevice)")
    args = parser.parse_args()
    selftest(args.out, args.play)
