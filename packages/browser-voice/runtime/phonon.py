"""Phonon on ptts, Gradium's official Rust runtime, as a speech out voice.

Backend options are exactly setupId, an absolute local modelPath (a pinned
gradium/phonon checkpoint directory) and files, its pins from the host
registry. Construction is lazy; load attests the payload and runtime, then
loads the checkpoint from that directory only (ptts downloads nothing for a
local path) on the CPU in f32, with its own default voice and English text
normalizer, which the voice-out research's ONNX adapter lacked. speak renders
one segment whole: ptts streams 80 ms frames, but a worker replies once per
request, so the host's sentence pipeline streams by sentence.
"""

import copy
import math
from pathlib import Path
import time

from pins import attest_snapshot, pinned_files, runtime_identity


VOICES = {"phonon-31m": "default"}
PYTHON_VERSION = "3.12.5"
RUNTIME_VERSIONS = {"ptts": "0.6.0", "numpy": "2.5.4"}
SAMPLE_RATE = 24_000
DEVICE = "cpu"


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files"}:
            raise ValueError("Phonon options require exactly setupId, modelPath and files")
        if options["setupId"] not in VOICES:
            raise ValueError("Phonon backend requires a Phonon voice")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or "\0" in model_path or not Path(model_path).is_absolute():
            raise ValueError("Phonon modelPath must be absolute")
        self._voice_id = options["setupId"]
        self._model_path = Path(model_path)
        self._files = pinned_files(options["files"], "Phonon")
        self._model = None
        self._metadata = None

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, self._files, "Phonon")
        runtime = runtime_identity(PYTHON_VERSION, RUNTIME_VERSIONS, "requirements-phonon.txt", "Phonon")
        start = time.perf_counter()
        import numpy as np
        import ptts

        self._np = np
        self._model = ptts.TTS(str(self._model_path), device=DEVICE, lang="en")
        if self._model.sample_rate != SAMPLE_RATE or VOICES[self._voice_id] not in self._model.voices:
            raise ValueError("Phonon checkpoint differs from its pinned rate or voice")
        self._metadata = {"setupId": self._voice_id, "runtime": runtime, "sampleRate": SAMPLE_RATE,
                          "device": self._model.device, "quant": self._model.quant,
                          "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def speak(self, text, voice, speed):
        if self._model is None:
            raise ValueError("Phonon backend is not loaded")
        if voice != VOICES[self._voice_id]:
            raise ValueError("Unsupported Phonon voice")
        if speed != 1:
            raise ValueError("Phonon has no speed control; speak at 1")
        start = time.perf_counter()
        samples = self._np.asarray(self._model.synth(text, voice=voice), dtype=self._np.float32).reshape(-1)
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("Phonon timing must be finite")
        return {"samples": samples.astype("<f4").tobytes(), "sampleRate": SAMPLE_RATE, "processingMs": elapsed}

    def close(self):
        self._model = None
