"""Kyutai Pocket TTS 18M, from the fast-tts pocket-18m candidate, as a speech out voice.

Backend options are exactly setupId, an absolute local modelPath (the
kyutai/pocket-tts-without-voice-cloning snapshot) and files, its pins from the
host registry. Construction is lazy; load attests the snapshot and runtime, loads the model from pocket-english-18m.yaml
with this snapshot's paths and the bill_boerst voice state. speak renders one
segment on PyTorch CPU; the host's sentence pipeline plans the segments.
"""

import copy
import math
import os
from pathlib import Path
import tempfile
import time

from pins import attest_snapshot, pinned_files, runtime_identity


VOICE_ID = "pocket-18m"
PYTHON_VERSION = "3.12.5"
RUNTIME_VERSIONS = {"pocket-tts": "4.0.1", "torch": "2.15.0", "sentencepiece": "0.2.2", "numpy": "2.5.4"}
CONFIG = Path(__file__).with_name("pocket-english-18m.yaml")
SAMPLE_RATE = 24_000
LANGUAGE = "languages/english_18m_2026-10"
VOICES = ("bill_boerst",)


class Backend:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) != {"setupId", "modelPath", "files"}:
            raise ValueError("Pocket options require exactly setupId, modelPath and files")
        if options["setupId"] != VOICE_ID:
            raise ValueError("Pocket backend requires pocket-18m")
        model_path = options["modelPath"]
        if not isinstance(model_path, str) or "\0" in model_path or not Path(model_path).is_absolute():
            raise ValueError("Pocket modelPath must be absolute")
        self._model_path = Path(model_path)
        self._files = pinned_files(options["files"], "Pocket")
        self._model = None
        self._state = None
        self._metadata = None

    def load(self):
        if self._metadata is not None:
            return copy.deepcopy(self._metadata)
        attest_snapshot(self._model_path, self._files, "Pocket")
        runtime = runtime_identity(PYTHON_VERSION, RUNTIME_VERSIONS, "requirements-pocket-18m.txt", "Pocket")
        os.environ["HF_HUB_OFFLINE"] = "1"  # local snapshot only
        start = time.perf_counter()
        import numpy as np
        from pocket_tts import TTSModel

        self._np = np
        # load_model takes a config path: the pinned template with this snapshot's directory filled in.
        with tempfile.TemporaryDirectory() as scratch:
            config = Path(scratch) / CONFIG.name
            config.write_text(CONFIG.read_text().replace("MODEL_DIR/", f"{self._model_path}/"))
            self._model = TTSModel.load_model(config=str(config))
        if self._model.sample_rate != SAMPLE_RATE:
            raise ValueError("Pocket sample rate differs from the pinned 24000 Hz")
        self._state = self._model.get_state_for_audio_prompt(
            self._model_path / LANGUAGE / "embeddings" / "bill_boerst.safetensors")
        self._metadata = {"setupId": VOICE_ID, "runtime": runtime, "sampleRate": SAMPLE_RATE,
                          "loadMs": (time.perf_counter() - start) * 1000}
        return copy.deepcopy(self._metadata)

    def speak(self, text, voice, speed):
        if self._model is None:
            raise ValueError("Pocket backend is not loaded")
        if voice not in VOICES:
            raise ValueError("Unsupported Pocket voice")
        if speed != 1:
            raise ValueError("Pocket has no speed control; speak at 1")
        start = time.perf_counter()
        # copy_state keeps the loaded voice state unchanged between segments.
        audio = self._model.generate_audio(self._state, text, copy_state=True)
        samples = self._np.asarray(audio.numpy(), dtype=self._np.float32).reshape(-1)
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("Pocket timing must be finite")
        return {"samples": samples.astype("<f4").tobytes(), "sampleRate": SAMPLE_RATE, "processingMs": elapsed}

    def close(self):
        self._model = None
        self._state = None
