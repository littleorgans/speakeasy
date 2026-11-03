"""Nemotron streaming contract checks with a scripted mlx-audio session; no model runtime is imported."""

import array
import contextlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

import nemotron
import worker


class FakeVector:
    """Just enough of a NumPy vector: a size and slices."""

    def __init__(self, size):
        self.size = size

    def __getitem__(self, window):
        return FakeVector(len(range(*window.indices(self.size))))


class FakeSession:
    """mlx-audio's NemotronStreamingSession: queued audio decodes one 2,560 sample chunk per step."""

    def __init__(self, runtime, delay_ms):
        runtime.sessions.append(delay_ms)
        self._lock = threading.Lock()
        self._queued, self._encoded, self.closed, self.done, self.words = 0, [], False, False, 0

    def feed(self, data):
        self._queued += data.size

    def step(self, *, max_decode_tokens):
        if self._queued >= 2_560:
            self._queued -= 2_560
            self.words += 1
            return [" w%d" % self.words]
        if self.closed:
            self._queued, self.done = 0, True
            return ["."]
        self._queued = 0
        return []

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self):
        self.sessions = []
        model = types.SimpleNamespace(model_type="nemotron_asr_streaming",
                                      preprocessor_config=types.SimpleNamespace(sample_rate=16000))
        model.create_streaming_session = lambda temperature, language, transcription_delay_ms: FakeSession(
            self, transcription_delay_ms)
        self.np = types.SimpleNamespace(frombuffer=lambda view, dtype: FakeVector(view.nbytes // 4),
                                        isfinite=lambda data: types.SimpleNamespace(all=lambda: True))
        self.stt = types.SimpleNamespace(load=lambda path, lazy, strict: model)

    @contextlib.contextmanager
    def installed(self):
        modules = {"numpy": self.np, "mlx": types.ModuleType("mlx"), "mlx.core": types.SimpleNamespace(synchronize=lambda: None),
                   "mlx_audio": types.ModuleType("mlx_audio"), "mlx_audio.stt": self.stt}
        with patch.dict(sys.modules, modules), patch.object(nemotron, "attest_snapshot"), \
                patch.object(nemotron, "runtime_identity", return_value={"python": "3.12.5"}):
            yield


def samples(count):
    return memoryview(array.array("f", [0.25] * count).tobytes()).cast("f")


class NemotronTest(unittest.TestCase):
    def backend(self):
        return nemotron.Backend({"setupId": "nemotron/streaming", "modelPath": "/local/snapshot"})

    def test_worker_routes_the_setup_and_options_are_exact(self):
        self.assertEqual(worker.BACKENDS["nemotron/streaming"], "nemotron")
        for bad in ({"setupId": "nemotron/streaming"}, {"setupId": "hex/unified", "modelPath": "/m"},
                    {"setupId": "nemotron/streaming", "modelPath": "relative"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nemotron.Backend(bad)

    def test_live_words_grow_at_the_chosen_delay_and_the_final_is_a_fresh_pass_at_560(self):
        runtime = FakeRuntime()
        with runtime.installed():
            backend = self.backend()
            self.assertEqual(backend.load()["delayMs"], 560)
            first = backend.feed(samples(2_560 * 2), True, {"delayMs": 160})
            self.assertEqual((first["stable"], first["tentative"]), (" w1 w2", ""), "emitted words are final")
            self.assertEqual(backend.feed(samples(2_560), False)["stable"], " w1 w2 w3")
            worker.validate_partial(first)
            result = backend.transcribe(samples(2_560 * 3))
            self.assertEqual(result["text"], " w1 w2 w3.", "the final drains and flushes its own session")
            self.assertEqual(runtime.sessions, [160, 560])
            self.assertEqual(backend.feed(samples(1), True)["stable"], "", "start opens a fresh stream")
            self.assertEqual(runtime.sessions[-1], 560, "no settings mean the research delay")
            for bad in ({"delayMs": 320}, {"delayMs": True}, {"lookaheadMs": 480}, []):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    backend.feed(samples(1), True, bad)
            with self.assertRaises(ValueError):
                backend.feed(samples(1), False, {"delayMs": 560})

    def test_feed_needs_a_started_stream_and_use_needs_load(self):
        runtime = FakeRuntime()
        with runtime.installed():
            backend = self.backend()
            with self.assertRaises(RuntimeError):
                backend.transcribe(samples(1))
            backend.load()
            with self.assertRaises(ValueError):
                backend.feed(samples(1), False)


if __name__ == "__main__":
    unittest.main()
