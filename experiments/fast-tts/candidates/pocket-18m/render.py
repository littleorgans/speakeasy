"""fast-tts candidate: Kyutai Pocket TTS 18M (October 2026), PyTorch CPU.

Emits the shared result JSON (CONTRACT.md) for one engine per process through
fasttts.run_engine, with the flags of render-baselines.ts.

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


class Pocket:
    def __init__(self, engine_id: str, root: Path) -> None:
        self.label, self.voice = ENGINES[engine_id]
        self.root = root
        self.voice_path = root / LANGUAGE_DIR / "embeddings" / f"{self.voice}.safetensors"
        checks = [(root / LANGUAGE_DIR / name, digest) for name, digest in FILES.items()]
        ft.verify_digests(checks + [(self.voice_path, VOICES[self.voice])])

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

    def describe(self) -> tuple[dict, dict[str, str]]:
        versions = {name: version(name) for name in ("pocket-tts", "torch", "sentencepiece", "numpy")}
        return {
            "label": self.label, "runtime": f"pocket-tts {versions['pocket-tts']} {SOURCE} (torch {versions['torch']})",
            "provider": "cpu", "model": f"{REPO}/languages/english_18m_2026-10", "modelRevision": REVISION,
            "voice": self.voice, "sampleRate": self.sample_rate, "threads": self.threads,
            "streaming": "native incremental: generate_audio_stream, 12.5 Hz latents decoded by Mimi as queued",
            "chunkPolicy": "upstream: text split at sentences into chunks of at most 50 tokens; each yield is one "
                           "Mimi decode of every queued 80 ms frame (1920 samples each), so the first chunk is one frame",
            "cancellation": "stop event before every latent frame (80 ms of audio) and every text chunk; "
                            "queued frames still decode",
        }, versions

    def stream(self, text: str):
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield chunk.numpy(), None

    def full(self, text: str) -> np.ndarray:
        return self.model.generate_audio(self.state, text).numpy()


if __name__ == "__main__":
    ft.run_engine(__doc__, ENGINES, Pocket)
