"""Kokoro v1.0 on MLX, from the fast-tts kokoro-mlx candidate, as a speech out voice.

Backend options are exactly setupId and an absolute local modelPath (the pinned
mlx-community/Kokoro-82M-bf16 snapshot). Construction is lazy; load attests the
snapshot and runtime, loads the model and builds the misaki G2P pipeline, which
mlx-audio would otherwise build inside the first generate. speak renders one
segment; the host's sentence pipeline plans and trims segments, as for sherpa.
"""

import copy
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import time


VOICE_ID = "kokoro-mlx-bf16"
PYTHON_VERSION = "3.12.5"
RUNTIME_VERSIONS = {"mlx-audio": "0.6.3", "mlx": "0.32.4", "misaki": "0.9.4", "numpy": "2.5.4"}
LANG_CODE = "a"  # misaki en-US G2P with espeak-ng fallback
SAMPLE_RATE = 24_000
# mlx-community/Kokoro-82M-bf16@a71e4d38b236d968966a2002c4c895dbd12b1c3c
FILES = {
    "config.json": (2351, "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"),
    "kokoro-v1_0.safetensors": (327115152, "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8"),
    "voices/af_heart.safetensors": (522320, "2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094"),
    "voices/am_michael.safetensors": (522320, "3940147ded35deba0bb52e8132f89b719298e0520258c34584358aa5a24da2ea"),
}
VOICES = ("af_heart", "am_michael")


def _validate_snapshot(model_path):
    if not model_path.is_dir():
        raise ValueError("Kokoro MLX modelPath must be an existing local snapshot directory")
    for name, (size, expected) in FILES.items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Kokoro MLX snapshot size mismatch: {name}")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Kokoro MLX snapshot SHA256 mismatch: {name}")


def _runtime_identity():
    if platform.python_version() != PYTHON_VERSION:
        raise ValueError(f"Kokoro MLX requires Python {PYTHON_VERSION}")
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_VERSIONS}
    if versions != RUNTIME_VERSIONS:
        raise ValueError("Kokoro MLX runtime version mismatch; use requirements-kokoro-mlx.txt")
    return {"python": PYTHON_VERSION, **versions}


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath"}:
            raise ValueError("Kokoro MLX options require exactly setupId and modelPath")
        if options["setupId"] != VOICE_ID:
            raise ValueError("Kokoro MLX backend requires kokoro-mlx-bf16")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or "\0" in model_path or not Path(model_path).is_absolute():
            raise ValueError("Kokoro MLX modelPath must be absolute")
        self._model_path = Path(model_path)
        self._model = None
        self._metadata = None

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        _validate_snapshot(self._model_path)
        runtime = _runtime_identity()
        # Local snapshot only: mlx-audio otherwise asks the hub for voice packs.
        os.environ["HF_HUB_OFFLINE"] = "1"
        start = time.perf_counter()
        import numpy as np
        from mlx_audio.tts.utils import load_model

        self._np = np
        self._model = load_model(str(self._model_path))
        if self._model.config.sample_rate != SAMPLE_RATE:
            raise ValueError("Kokoro MLX sample rate differs from the pinned 24000 Hz")
        self._model._get_pipeline(LANG_CODE)
        self._metadata = {"setupId": VOICE_ID, "runtime": runtime, "sampleRate": SAMPLE_RATE,
                          "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def speak(self, text, voice, speed):
        if self._model is None:
            raise ValueError("Kokoro MLX backend is not loaded")
        if voice not in VOICES:
            raise ValueError("Unsupported Kokoro MLX voice")
        start = time.perf_counter()
        # No split pattern: the segment is already one planned sentence part.
        parts = [self._np.asarray(result.audio, dtype=self._np.float32).reshape(-1)
                 for result in self._model.generate(text, voice=voice, speed=speed,
                                                    lang_code=LANG_CODE, split_pattern=None)]
        samples = self._np.concatenate(parts) if parts else self._np.zeros(0, dtype=self._np.float32)
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("Kokoro MLX timing must be finite")
        return {"samples": samples.astype("<f4").tobytes(), "sampleRate": SAMPLE_RATE, "processingMs": elapsed}

    def close(self):
        self._model = None
