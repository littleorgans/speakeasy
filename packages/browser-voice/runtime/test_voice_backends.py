"""Voice backend contracts without their runtimes: options, pins, snapshot attestation and speak guards."""

import hashlib
import importlib
import struct
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import native_voice

PIN = [{"name": "weights.bin", "bytes": 4, "sha256": hashlib.sha256(b"pins").hexdigest()}]
# (module, voice id, a pinned voice, extra options, the runtime module load must not import)
BACKENDS = [
    ("kokoro_mlx", "kokoro-mlx-bf16", "am_michael", {}, "mlx_audio"),
    ("pocket_18m", "pocket-18m", "bill_boerst", {}, "pocket_tts"),
    ("native_voice", "kokoro-coreml", "af_heart", {"runnerPath": "/r"}, None),
    ("phonon", "phonon-31m", "default", {}, "ptts"),
    ("qwen_tts", "qwen3-tts-0.6b", "ryan", {"runnerPath": "/r"}, None),
    ("native_voice", "kokoro-ane-folded-heart", "af_heart", {"runnerPath": "/r"}, None),
]

# A runner speaking the native frame protocol: READY, then per request one AUDIO frame per part and END.
FAKE_RUNNER = textwrap.dedent("""\
    #!{python}
    import json, os, struct, sys
    out = os.fdopen(os.dup(1), "wb")
    def frame(kind, count, payload):
        out.write(struct.pack("<ii", kind, count) + payload); out.flush()
    info = json.dumps({{"args": sys.argv[1:]}}).encode()
    frame(1, len(info), info)
    for line in sys.stdin:
        line = line.rstrip("\\n")
        if line == "Q": break
        parts = line[2:].split("\\x1f")
        if "fail" in parts:
            error = b'{{"error":"synthesis failed"}}'
            frame(4, len(error), error); continue
        for part in parts:
            frame(2, 2, struct.pack("<2f", len(part), 0.5))
        frame(3, 2, b"{{}}")
""")


class VoiceBackendTests(unittest.TestCase):
    def backend(self, module, voice_id, extra, root="/m", files=PIN):
        return importlib.import_module(module).Backend({"setupId": voice_id, "modelPath": root, "files": files, **extra})

    def test_options_are_exactly_the_voice_its_pins_and_absolute_paths(self):
        for module, voice_id, _, extra, _ in BACKENDS:
            options = {"setupId": voice_id, "modelPath": "/m", "files": PIN, **extra}
            for bad in ({**options, "setupId": "kokoro-v1.0"}, {**options, "modelPath": "relative"},
                        {**options, "voice": "x"}, {k: v for k, v in options.items() if k != "files"},
                        {**options, "files": []}, {**options, "files": [{**PIN[0], "name": "../escape"}]},
                        {**options, "files": [{**PIN[0], "bytes": 0}]}):
                with self.subTest(module=module, options=bad), self.assertRaises(ValueError):
                    importlib.import_module(module).Backend(bad)

    def test_load_refuses_a_wrong_snapshot_before_importing_the_runtime(self):
        for module, voice_id, _, extra, runtime in BACKENDS:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as root:
                (Path(root) / "weights.bin").write_bytes(b"nope")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch: weights.bin"):
                    self.backend(module, voice_id, extra, root).load()
                if runtime:
                    self.assertNotIn(runtime, sys.modules)

    def test_speak_needs_a_loaded_model_and_a_pinned_voice(self):
        for module, voice_id, voice, extra, _ in BACKENDS:
            with self.subTest(module=module):
                backend = self.backend(module, voice_id, extra)
                with self.assertRaisesRegex(ValueError, "not loaded"):
                    backend.speak("Hi.", voice, 1.0)
                backend._model = backend._metadata = object()
                with self.assertRaisesRegex(ValueError, "voice"):
                    backend.speak("Hi.", "af_bella", 1.0)

    def test_runtimes_without_speed_control_refuse_other_speeds(self):
        for module, voice_id, voice, extra in (("pocket_18m", "pocket-18m", "bill_boerst", {}),
                                               ("native_voice", "kokoro-coreml", "af_heart", {"runnerPath": "/r"}),
                                               ("phonon", "phonon-31m", "default", {}),
                                               ("qwen_tts", "qwen3-tts-0.6b", "ryan", {"runnerPath": "/r"})):
            backend = self.backend(module, voice_id, extra)
            backend._model = backend._metadata = object()
            with self.subTest(module=module), self.assertRaisesRegex(ValueError, "speed"):
                backend.speak("Hi.", voice, 1.5)

    def test_native_runner_parts_audio_errors_and_one_process_per_launch_voice(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Path(root) / "runner"
            runner.write_text(FAKE_RUNNER.format(python=sys.executable))
            runner.chmod(0o755)
            (Path(root) / "weights.bin").write_bytes(b"pins")
            backend = self.backend("native_voice", "kokoro-coreml", {"runnerPath": str(runner)}, root)
            try:
                metadata = backend.load()
                self.assertEqual(metadata["runtime"]["args"], [root, "af_heart", "legacy"])
                result = backend.speak("ab\x1fcde", "af_heart", 1)
                self.assertEqual(struct.unpack("<4f", result["samples"]), (2, 0.5, 3, 0.5))
                self.assertEqual(result["sampleRate"], 24_000)
                with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
                    backend.speak("fail", "af_heart", 1)
                with self.assertRaisesRegex(ValueError, "separator"):
                    backend.speak("a\nb", "af_heart", 1)
                backend.speak("x", "am_michael", 1)
                self.assertEqual(len(backend._processes), 2, "Core ML takes its voice at launch")
                processes = [process for process, _ in backend._processes.values()]
            finally:
                backend.close()
            self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertFalse(native_voice.SPECS["piper-kristin-split"]["perVoice"])

    def test_a_runner_that_loads_its_models_on_first_synthesis_is_warmed_inside_load(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Path(root) / "runner"
            # The frame runner, also logging each request's parts to requests.log beside it.
            runner.write_text(FAKE_RUNNER.format(python=sys.executable).replace(
                '    parts = line[2:].split("\\x1f")\n',
                '    parts = line[2:].split("\\x1f")\n'
                '    open(os.path.join(os.path.dirname(sys.argv[0]), "requests.log"), "a").write(line[2:] + "\\n")\n'))
            runner.chmod(0o755)
            (Path(root) / "weights.bin").write_bytes(b"pins")
            for voice_id, generator in (("kokoro-ane-heart", "hybrid"), ("kokoro-ane-folded-heart", "folded")):
                (Path(root) / "requests.log").write_text("")
                backend = self.backend("native_voice", voice_id, {"runnerPath": str(runner)}, root)
                try:
                    self.assertEqual(backend.load()["runtime"]["args"], [root, "af_heart", generator])
                    backend.speak("ab", "af_heart", 1)
                finally:
                    backend.close()
                self.assertEqual((Path(root) / "requests.log").read_text().split(), ["Hello.", "ab"],
                                 "load warms the runner once, before the first visible request")


if __name__ == "__main__":
    unittest.main()
