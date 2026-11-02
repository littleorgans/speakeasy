"""fast-tts candidate: Kyutai Pocket TTS 18M (October 2026), PyTorch CPU.

Emits the shared result JSON (CONTRACT.md) for one engine per process, like
render-baselines.ts: cold loads and runs one --text with no warmup; warm loads,
discards one "Warm up." synthesis, then runs --repeats passes over the texts.

Stream mode is the model's own incremental audio: the whole text goes to one
TTSModel.generate_audio_stream call, which splits it into chunks of at most 50
tokens, generates 12.5 Hz latent frames on one thread and decodes them with Mimi
on another; each yield is one decode call of one or more queued 80 ms frames.
No host segmentation and no silence trim. The bill_boerst voice state loads
once at startup and every call deep copies it (copy_state=True).
Cancellation checkpoint: upstream's `stop` event, checked before every latent
frame and every text chunk (these runs never set it).

Run from the speak-easy repo root with the env from README.md:
    $TMPDIR/venv-pocket/bin/python -I experiments/fast-tts/candidates/pocket-18m/render.py \
        --engine pocket-18m-bill --temperature warm --json <file> [--clips <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # local snapshot only

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/fast-tts
import fasttts as ft  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = "kyutai/pocket-tts-without-voice-cloning"
REVISION = "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"
SOURCE = "kyutai-labs/pocket-tts@91a9e96433bfe23375eea719b03f0081466f8e20"
LANGUAGE_DIR = "pocket-tts-english-18m-2026-10-a7056f50/languages/english_18m_2026-10"
CONFIG = HERE / "english_18m_2026-10.local.yaml"
FILES = {"model.safetensors": "599329dcea2f2a8d17b575e1c6e11b78822453c9073040d55c26faadf4883188",
         "tokenizer.model": "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"}
VOICES = {"bill_boerst": "927a31d7627daf996efce71ee7898a799a9386738077c3bd401bd1445f6f6657"}
ENGINES = {"pocket-18m-bill": ("Pocket TTS 18M · bill_boerst", "bill_boerst")}
WARMUP_TEXT = "Warm up."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Pocket:
    def __init__(self, root: Path, voice: str) -> None:
        self.root = root
        self.voice_path = root / LANGUAGE_DIR / "embeddings" / f"{voice}.safetensors"
        checks = [(root / LANGUAGE_DIR / name, digest) for name, digest in FILES.items()]
        for path, digest in checks + [(self.voice_path, VOICES[voice])]:
            if sha256(path) != digest:
                raise SystemExit(f"{path}: sha256 mismatch, expected {digest}")

    def load(self) -> dict:
        start = ft.now_ms()
        import torch
        from pocket_tts import TTSModel

        imported = ft.now_ms()
        with tempfile.TemporaryDirectory() as scratch:  # load_model takes a path: the template with --root filled in
            config = Path(scratch) / "english_18m_2026-10.yaml"
            config.write_text(CONFIG.read_text().replace("MODELS_ROOT", str(self.root.resolve())))
            self.model = TTSModel.load_model(config=str(config))
        loaded = ft.now_ms()
        self.state = self.model.get_state_for_audio_prompt(self.voice_path)
        self.sample_rate = self.model.sample_rate
        self.threads = torch.get_num_threads()
        return {"importMs": round(imported - start, 3), "modelLoadMs": round(loaded - imported, 3),
                "voiceStateMs": round(ft.now_ms() - loaded, 3), "torchThreads": self.threads,
                "note": "loadMs is torch and pocket_tts import, load_model on the local config, "
                        "and the bill_boerst voice state from its safetensors"}

    def stream(self, text: str):
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield chunk.numpy(), None

    def full(self, text: str) -> np.ndarray:
        return self.model.generate_audio(self.state, text).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--temperature", choices=ft.TEMPERATURES, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd() / "models" / "tts")
    parser.add_argument("--mode", choices=ft.MODES, default="stream")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--text", action="append")
    parser.add_argument("--clips", type=Path)
    parser.add_argument("--timed", action="store_true")
    args = parser.parse_args()
    label, voice = ENGINES[args.engine]
    texts = ft.load_cohort(args.text or (["status"] if args.temperature == "cold" else None))
    if args.temperature == "cold" and len(texts) != 1:
        raise SystemExit("cold takes exactly one --text")
    repeats = 1 if args.temperature == "cold" else args.repeats

    engine = Pocket(args.root, voice)  # digest checks off the clock
    start = ft.now_ms()
    load_detail = engine.load()
    load_ms = ft.now_ms() - start
    versions = {name: version(name) for name in ("pocket-tts", "torch", "sentencepiece", "numpy")}
    result = ft.Result({
        "id": args.engine, "label": label, "runtime": f"pocket-tts {versions['pocket-tts']} {SOURCE} "
        f"(torch {versions['torch']})", "provider": "cpu", "model": f"{REPO}/languages/english_18m_2026-10",
        "modelRevision": REVISION, "voice": voice, "sampleRate": engine.sample_rate, "threads": engine.threads,
        "streaming": "native incremental: generate_audio_stream, 12.5 Hz latents decoded by Mimi as queued",
        "chunkPolicy": "upstream: text split at sentences into chunks of at most 50 tokens; each yield is one "
                       "Mimi decode of every queued 80 ms frame (1920 samples each), so the first chunk is one frame",
        "cancellation": "stop event before every latent frame (80 ms of audio) and every text chunk; "
                        "queued frames still decode",
    }, args.temperature, args.timed, repeats, versions)
    result.process["loadMs"] = round(load_ms, 3)
    result.process["loadDetail"] = load_detail
    if args.temperature == "warm":
        warm = ft.now_ms()
        engine.full(WARMUP_TEXT)
        result.process["warmupMs"] = round(ft.now_ms() - warm, 3)

    for item in texts:
        for repeat in range(repeats):
            run = result.start(item["id"], args.mode, repeat)
            if args.mode == "stream":
                pieces = []
                for samples, line in engine.stream(item["text"]):
                    run.chunk(samples.size / engine.sample_rate * 1000, line)
                    pieces.append(samples)
                clip = np.concatenate(pieces)
            else:
                clip = engine.full(item["text"])
                run.chunk(clip.size / engine.sample_rate * 1000, item["text"])
            if repeat == 0 and args.clips:
                suffix = "--full" if args.mode == "full" else ""
                ft.write_wav(args.clips / f"{item['id']}--{args.engine}{suffix}.wav", clip, engine.sample_rate)
    result.write(args.json)
    print(f"{args.engine} {args.temperature} {args.mode}: {len(result.runs)} runs, load {load_ms:.0f} ms -> {args.json}")


if __name__ == "__main__":
    main()
