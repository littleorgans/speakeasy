"""Kokoro MLX backend contract without MLX: options and snapshot attestation."""

import sys
import tempfile
import unittest
from pathlib import Path

import kokoro_mlx


class KokoroMlxTests(unittest.TestCase):
    def test_options_are_exactly_the_voice_and_an_absolute_snapshot(self):
        for options in ({"setupId": "kokoro-mlx-bf16"}, {"setupId": "kokoro-v1.0", "modelPath": "/m"},
                        {"setupId": "kokoro-mlx-bf16", "modelPath": "relative"},
                        {"setupId": "kokoro-mlx-bf16", "modelPath": "/m", "voice": "af_heart"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                kokoro_mlx.Backend(options)

    def test_load_refuses_a_wrong_snapshot_before_importing_mlx(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "config.json").write_text("{}")
            backend = kokoro_mlx.Backend({"setupId": "kokoro-mlx-bf16", "modelPath": root})
            with self.assertRaisesRegex(ValueError, "size mismatch: config.json"):
                backend.load()
        self.assertNotIn("mlx_audio", sys.modules)

    def test_speak_needs_a_loaded_model_and_a_pinned_voice(self):
        backend = kokoro_mlx.Backend({"setupId": "kokoro-mlx-bf16", "modelPath": "/m"})
        with self.assertRaisesRegex(ValueError, "not loaded"):
            backend.speak("Hi.", "af_heart", 1.0)
        backend._model = object()
        with self.assertRaisesRegex(ValueError, "voice"):
            backend.speak("Hi.", "af_bella", 1.0)


if __name__ == "__main__":
    unittest.main()
