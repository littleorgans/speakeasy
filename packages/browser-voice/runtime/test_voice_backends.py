"""Voice backend contracts without their runtimes: options, snapshot attestation and speak guards."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

# (module, voice id, a pinned voice, the first pinned file, the runtime module load must not import)
BACKENDS = [
    ("kokoro_mlx", "kokoro-mlx-bf16", "am_michael", "config.json", "mlx_audio"),
    ("pocket_18m", "pocket-18m", "bill_boerst", "languages/english_18m_2026-10/model.safetensors", "pocket_tts"),
]


class VoiceBackendTests(unittest.TestCase):
    def test_options_are_exactly_the_voice_and_an_absolute_snapshot(self):
        for module, voice_id, _, _, _ in BACKENDS:
            backend = importlib.import_module(module).Backend
            for options in ({"setupId": voice_id}, {"setupId": "kokoro-v1.0", "modelPath": "/m"},
                            {"setupId": voice_id, "modelPath": "relative"},
                            {"setupId": voice_id, "modelPath": "/m", "voice": "x"}):
                with self.subTest(module=module, options=options), self.assertRaises(ValueError):
                    backend(options)

    def test_load_refuses_a_wrong_snapshot_before_importing_the_runtime(self):
        for module, voice_id, _, first_file, runtime in BACKENDS:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as root:
                backend = importlib.import_module(module).Backend({"setupId": voice_id, "modelPath": root})
                with self.assertRaisesRegex(ValueError, f"size mismatch: {first_file}"):
                    backend.load()
                self.assertNotIn(runtime, sys.modules)

    def test_speak_needs_a_loaded_model_and_a_pinned_voice(self):
        for module, voice_id, voice, _, _ in BACKENDS:
            with self.subTest(module=module):
                backend = importlib.import_module(module).Backend({"setupId": voice_id, "modelPath": "/m"})
                with self.assertRaisesRegex(ValueError, "not loaded"):
                    backend.speak("Hi.", voice, 1.0)
                backend._model = object()
                with self.assertRaisesRegex(ValueError, "voice"):
                    backend.speak("Hi.", "af_bella", 1.0)

    def test_pocket_refuses_speed_it_cannot_honour(self):
        backend = importlib.import_module("pocket_18m").Backend({"setupId": "pocket-18m", "modelPath": "/m"})
        backend._model = object()
        with self.assertRaisesRegex(ValueError, "speed"):
            backend.speak("Hi.", "bill_boerst", 1.5)


if __name__ == "__main__":
    unittest.main()
