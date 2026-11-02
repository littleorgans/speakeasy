"""Register Audio8 voices and render or time the cohort through upstream's ONNX runtime.

Run with the runtime's own venv and -I; the runtime package is imported from
--runtime-dir (onnx_runtime for 0.6B INT4, onnx_runtime_0_1b_int8 for 0.1B INT8).
Both runtimes share one API: ArkTtsRuntime.synthesize() and .stream(chunk_frames).

  register: encode a reference WAV/MP3 plus its exact transcript into a voice profile.
  bundled:  register the 0.1B model's bundled reference codes (a Chinese speaker).
  render:   cold load, one warmup, then `runs` passes per text and mode.
            full:   TTFA is the whole synthesis (no audio before it ends).
            stream: TTFA is the first stream chunk; RTF is time to the last chunk over audio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def load_runtime(runtime_dir: Path):
    sys.path.insert(0, str(runtime_dir))
    from arktts_runtime.registration import VoiceRegistration
    from arktts_runtime.runtime import ArkTtsRuntime

    return ArkTtsRuntime, VoiceRegistration


def register(args) -> None:
    _, VoiceRegistration = load_runtime(args.runtime_dir)
    manifest = json.loads((args.model_dir / "runtime_manifest.json").read_text())
    registration = VoiceRegistration(
        args.model_dir / "registration", args.voices_dir, manifest["model_fingerprint"]
    )
    meta = registration.register(
        args.audio.read_bytes(), args.audio.name, args.text, args.name, overwrite=True
    )
    print(f"registered {args.name}: codes {meta['shape']}")


def bundled(args) -> None:
    import numpy as np

    manifest = json.loads((args.model_dir / "runtime_manifest.json").read_text())
    codes = np.load(args.model_dir / manifest["reference_codes"], allow_pickle=False)
    target = args.voices_dir / args.name
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "codes.npy", codes.astype(np.uint16))
    meta = {
        "name": args.name,
        "reference_text": manifest["reference_text"],
        "shape": list(codes.shape),
        "model_fingerprint": manifest["model_fingerprint"],
        "source_kind": "model_reference_codes",
    }
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"registered {args.name}: codes {list(codes.shape)}")


def run_once(runtime, mode: str, text: str, voice: str, seed: int, chunk_frames: int):
    start = time.perf_counter()
    if mode == "full":
        audio, codes = runtime.synthesize(text=text, voice=voice, seed=seed)
        end = time.perf_counter() - start
        return audio, {"ttfaMs": end * 1000, "lastReadyMs": end * 1000, "frames": int(codes.shape[1])}
    import numpy as np

    chunks, first, frames = [], None, 0
    for event in runtime.stream(chunk_frames=chunk_frames, text=text, voice=voice, seed=seed):
        if event["type"] == "audio_chunk":
            first = first if first is not None else time.perf_counter() - start
            chunks.append(event["audio"])
        else:
            frames = int(event["codes"].shape[1])
    end = time.perf_counter() - start
    return np.concatenate(chunks), {
        "ttfaMs": first * 1000, "lastReadyMs": end * 1000, "frames": frames, "chunks": len(chunks)
    }


def render(args) -> None:
    import soundfile as sf

    ArkTtsRuntime, _ = load_runtime(args.runtime_dir)
    cohort = json.loads((Path(__file__).parent / "cohort.json").read_text())
    texts = [item for item in cohort["texts"] if not args.text or item["id"] in args.text]

    start = time.perf_counter()
    runtime = ArkTtsRuntime(args.model_dir, args.voices_dir, threads=args.threads)
    cold_load_ms = (time.perf_counter() - start) * 1000
    rate = int(runtime.manifest["sample_rate"])
    start = time.perf_counter()
    runtime.synthesize(text="Warm up.", voice=args.voice[0], seed=args.seed, max_new_tokens=32)
    warmup_ms = (time.perf_counter() - start) * 1000

    results = []
    for voice in args.voice:
        for mode in args.mode:
            for item in texts:
                passes = []
                for run in range(args.runs):
                    audio, timing = run_once(runtime, mode, item["text"], voice, args.seed, args.chunk_frames)
                    audio_ms = audio.size / rate * 1000
                    passes.append({**timing, "audioMs": audio_ms, "rtf": timing["lastReadyMs"] / audio_ms})
                    if run == 0:
                        name = f"{item['id']}--{args.label}-{voice}{'-stream' if mode == 'stream' else ''}.wav"
                        sf.write(str(args.out / name), audio, rate)
                results.append({
                    "voice": voice, "mode": mode, "id": item["id"], "runs": passes,
                    "medianTtfaMs": statistics.median(p["ttfaMs"] for p in passes),
                    "medianRtf": statistics.median(p["rtf"] for p in passes),
                })
                print(f"{args.label} {voice} {mode} {item['id']}: "
                      f"ttfa {results[-1]['medianTtfaMs']:.0f} ms, rtf {results[-1]['medianRtf']:.2f}")
    args.json.write_text(json.dumps({
        "schema": 1, "engine": args.label, "modelDir": args.model_dir.name, "threads": args.threads,
        "sampleRate": rate, "seed": args.seed, "chunkFrames": args.chunk_frames, "runsPerText": args.runs,
        "coldLoadMs": cold_load_ms, "warmupMs": warmup_ms, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["register", "bundled", "render"])
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--name", help="voice name (register, bundled)")
    parser.add_argument("--audio", type=Path, help="reference audio (register)")
    parser.add_argument("--text", nargs="*", help="transcript (register) or cohort ids (render)")
    parser.add_argument("--label", help="engine label used in clip names (render)")
    parser.add_argument("--voice", nargs="+", help="voice names (render)")
    parser.add_argument("--mode", nargs="+", choices=["full", "stream"], default=["full"])
    parser.add_argument("--out", type=Path, help="clip directory (render)")
    parser.add_argument("--json", type=Path, help="timings output (render)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=5, help="upstream CLI default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-frames", type=int, default=12, help="upstream stream default")
    parsed = parser.parse_args()
    parsed.runtime_dir = parsed.runtime_dir.resolve()
    if parsed.command == "register":
        parsed.text = " ".join(parsed.text or [])
    {"register": register, "bundled": bundled, "render": render}[parsed.command](parsed)
