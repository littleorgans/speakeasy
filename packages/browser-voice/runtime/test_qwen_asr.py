"""Standard library bridge tests. No speech.cpp library or inference is loaded."""

import array
import ctypes as C
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import qwen_asr as qwen


# Independent ABI expectation from the measured native.py bridge.
P = C.c_void_p
EXPECTED_SIGNATURES = {
    "speech_version": (C.c_char_p, []),
    "speech_last_error": (C.c_char_p, []),
    "speech_option_from_name": (C.c_int, [C.c_char_p, C.POINTER(C.c_int)]),
    "speech_load_params_new": (C.c_int, [C.POINTER(P)]),
    "speech_load_params_free": (None, [P]),
    "speech_load_params_set_device": (C.c_int, [P, C.c_char_p]),
    "speech_load_params_set_threads": (C.c_int, [P, C.c_int]),
    "speech_load_params_set_warmup": (C.c_int, [P, C.c_int]),
    "speech_model_load": (C.c_int, [C.c_char_p, P, C.POINTER(P)]),
    "speech_model_free": (None, [P]),
    "speech_model_get_info": (C.c_int, [P, C.POINTER(P)]),
    "speech_model_info_free": (None, [P]),
    "speech_model_info_json": (C.c_char_p, [P]),
    "speech_model_info_takes": (C.c_int, [P, C.c_int]),
    "speech_model_info_option_steers": (C.c_int, [P, C.c_int]),
    "speech_request_new": (C.c_int, [P, C.POINTER(P)]),
    "speech_request_free": (None, [P]),
    "speech_request_set_audio": (
        C.c_int, [P, C.POINTER(C.c_float), C.c_size_t, C.c_int]
    ),
    "speech_request_set_string": (C.c_int, [P, C.c_int, C.c_char_p]),
    "speech_transcribe": (C.c_int, [P]),
    "speech_request_result": (P, [P]),
    "speech_result_text": (C.c_char_p, [P]),
    "speech_result_stop": (C.c_int, [P]),
}
OPTIONS = {"libraryPath": "/fake/qwen/libspeech.4.dylib",
           "modelPath": "/fake/qwen/Qwen3-ASR-0.6B-Q8_0.gguf"}


class FakeFunction:
    def __init__(self, library, name):
        self.library = library
        self.name = name
        self.restype = "unset"
        self.argtypes = "unset"

    def __call__(self, *args):
        return self.library.call(self.name, args)


class FakeLibrary:
    """Use real ctypes out pointers and enforce native handle ownership."""

    def __init__(self):
        self.calls = []
        self.live = {}
        self.freed = []
        self.next_handle = 100
        self.fail = None
        self.null = None
        self.clock = 0.0
        self.audio = []
        self.responses = {
            "speech_version": b"0.10.0", "speech_last_error": b"fake failure",
            "speech_model_info_takes": 1, "speech_model_info_option_steers": 1,
            "speech_model_info_json": b'{"device":"MTL0"}',
            "speech_request_result": 999, "speech_result_text": b" raw text\n",
            "speech_result_stop": 3,
        }
        for name in EXPECTED_SIGNATURES:
            setattr(self, name, FakeFunction(self, name))

    def call(self, name, args):
        self.calls.append((name, args))
        self.clock += {
            "speech_model_load": 0.007,
            "speech_request_new": 0.001, "speech_request_set_string": 0.002,
            "speech_request_set_audio": 0.003, "speech_transcribe": 0.005,
            "speech_request_result": 0.2, "speech_result_text": 0.3,
            "speech_result_stop": 0.4, "speech_request_free": 0.5,
        }.get(name, 0.001)
        creates = {"speech_load_params_new": "params", "speech_model_load": "model",
                   "speech_model_get_info": "info", "speech_request_new": "request"}
        frees = {"speech_load_params_free": "params", "speech_model_free": "model",
                 "speech_model_info_free": "info", "speech_request_free": "request"}
        if name in creates and name != self.null:
            self.next_handle += 1
            self.live[self.next_handle] = creates[name]
            C.cast(args[-1], C.POINTER(P))[0] = self.next_handle
        if name in frees:
            handle = args[0].value
            assert self.live.pop(handle) == frees[name], "invalid or double free"
            self.freed.append((handle, frees[name]))
            return None
        if name == "speech_option_from_name":
            C.cast(args[1], C.POINTER(C.c_int))[0] = {b"language": 71, b"prompt": 19}[args[0]]
        if name == "speech_request_set_audio":
            assert self.live[args[0].value] == "request"
            self.audio.append((C.string_at(args[1], args[2] * 4), args[2], args[3]))
        if name in self.responses:
            return self.responses[name]
        occurrence = sum(called == name for called, _ in self.calls)
        return 17 if self.fail == (name, occurrence) else 0

    def arguments(self, name):
        return [args for called, args in self.calls if called == name]


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeLibrary()
        self.loader = self.enterContext(patch.object(qwen.C, "CDLL", return_value=self.fake))
        self.digests = self.enterContext(patch.object(qwen, "_digest", side_effect=lambda path: {
            OPTIONS["libraryPath"]: qwen.LIBRARY_SHA256,
            OPTIONS["modelPath"]: qwen.MODEL_SHA256,
        }[str(path)]))
        self.enterContext(patch.object(qwen.platform, "python_version", return_value="3.13.2"))
        self.enterContext(patch.object(qwen.time, "perf_counter", side_effect=lambda: self.fake.clock))
        self.backend = qwen.Backend(OPTIONS)
        self.addCleanup(self.backend.close)

    def test_lazy_load_pinned_identity_options_signatures_and_idempotency(self):
        self.loader.assert_not_called()
        self.digests.assert_not_called()
        info = self.backend.load()
        self.loader.assert_called_once_with(OPTIONS["libraryPath"])
        self.assertEqual(set(qwen.SIGNATURES), set(EXPECTED_SIGNATURES))
        for name, signature in EXPECTED_SIGNATURES.items():
            fn = getattr(self.fake, name)
            self.assertEqual((fn.restype, fn.argtypes), signature, name)
        self.assertEqual(info["setupId"], "qwen/context")
        self.assertEqual(info["runtimeVersion"], "0.10.0")
        self.assertEqual(info["runtimeCommit"], "06590811a90041f935a470ba0e2b85346d2fc041")
        self.assertEqual(info["modelRevision"], "ee78f8709547f39bea3b924a16c72403826cac92")
        self.assertEqual(info["modelSha256"], qwen.MODEL_SHA256)
        self.assertEqual(info["librarySha256"], qwen.LIBRARY_SHA256)
        self.assertEqual(info["pythonVersion"], "3.13.2")
        self.assertEqual(info["modelInfo"], {"device": "MTL0"})
        self.assertEqual(info["deviceRequested"], "gpu")
        self.assertEqual((info["threads"], info["warmup"], info["language"], info["prompt"]),
                         (2, False, "auto", "launch, claude, grok"))
        for name, expected in (("device", b"gpu"), ("threads", 2), ("warmup", 0)):
            self.assertEqual(self.fake.arguments("speech_load_params_set_" + name)[0][1], expected)
        self.assertEqual(self.fake.arguments("speech_model_load")[0][0], OPTIONS["modelPath"].encode())
        self.assertEqual(self.fake.arguments("speech_option_from_name")[0][0], b"language")
        self.assertEqual(self.fake.arguments("speech_model_info_takes")[0][1], 19)
        self.assertAlmostEqual(info["loadMs"], 7)
        self.assertEqual(list(self.fake.live.values()), ["model"])
        self.assertEqual(self.fake.arguments("speech_transcribe"), [])
        info["modelInfo"]["device"] = "changed"
        self.assertEqual(self.backend.load()["modelInfo"], {"device": "MTL0"})
        self.assertEqual(len(self.fake.arguments("speech_model_load")), 1)
        self.backend.close()
        self.backend.close()
        self.assertEqual(self.fake.live, {})
        self.assertEqual(len(self.fake.arguments("speech_model_free")), 1)

    def test_exact_float32_view_raw_text_and_original_timing_boundaries(self):
        self.backend.load()
        self.fake.responses["speech_result_text"] = " \nClauDe, GROK café\t ".encode()
        values = array.array("f", [99, -0.0, 2.5, -3.25, 1e-45, 0.25, 88])
        original = values.tobytes()
        view = memoryview(values)[1:-1]
        payload = view.tobytes()
        result = self.backend.transcribe(view)
        self.assertEqual(result["text"], " \nClauDe, GROK café\t ")
        self.assertEqual(self.fake.audio, [(payload, 5, 16000)])
        self.assertEqual(result["diagnostics"]["sampleCount"], 5)
        self.assertEqual(result["diagnostics"]["pcmSha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(payload[:4], struct.pack("<f", -0.0))
        self.assertEqual(values.tobytes(), original)
        self.assertAlmostEqual(result["processingMs"], 13)
        self.assertAlmostEqual(result["decodeMs"], 5)
        self.assertEqual(result["stop"], 3)
        self.assertEqual([(args[1], args[2]) for args in self.fake.arguments("speech_request_set_string")],
                         [(71, b"auto"), (19, b"launch, claude, grok")])
        self.assertEqual(list(self.fake.live.values()), ["model"])

    def test_readonly_buffers_empty_text_and_fresh_requests(self):
        self.backend.load()
        self.fake.responses["speech_result_text"] = b""
        payload = struct.pack("<4f", -0.0, 1e-45, 2.0, -1.5)
        result = self.backend.transcribe(memoryview(payload).cast("f"))
        self.assertEqual(result["text"], "")
        self.assertEqual(self.fake.audio[0], (payload, 4, 16000))
        empty = self.backend.transcribe(array.array("f"))
        self.assertEqual(empty["text"], "")
        self.assertEqual(self.fake.audio[1], (b"", 0, 16000))
        self.assertEqual(empty["diagnostics"]["pcmSha256"], hashlib.sha256(b"").hexdigest())
        requests = [handle for handle, kind in self.fake.freed if kind == "request"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(set(requests)), 2)
        self.assertEqual(len(self.fake.arguments("speech_model_load")), 1)
        self.assertEqual(len(self.fake.arguments("speech_transcribe")), 2)

    def test_options_and_unloaded_calls_fail_before_native_load(self):
        for options in ({}, {**OPTIONS, "threads": 3}, {**OPTIONS, "modelPath": "relative"},
                        {**OPTIONS, "libraryPath": "/bad\0path"}, {**OPTIONS, "modelPath": None},
                        {**OPTIONS, "setupId": "mlx-parakeet/v3-greedy"}):
            with self.subTest(options=tuple(options)):
                with self.assertRaises(ValueError):
                    qwen.Backend(options)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            self.backend.transcribe(array.array("f", [0]))
        self.loader.assert_not_called()

    def test_worker_factory_pcm_hash_and_native_identity_agree(self):
        import worker

        backend, ready = worker.handle_request(
            {"operation": "load", "options": OPTIONS}, b"", None,
            "qwen/context", worker.backend_factory)
        self.addCleanup(backend.close)
        self.assertEqual(ready["metadata"]["setupId"], "qwen/context")
        payload = struct.pack("<5f", -0.0, 2.5, -3.25, 1e-45, 0.25)
        pcm_hash = hashlib.sha256(payload).hexdigest()
        header = {"operation": "transcribe", "sampleRate": 16000, "channels": 1,
                  "sampleCount": 5, "pcmBytes": len(payload), "sha256": pcm_hash}
        _, reply = worker.handle_request(header, payload, backend, "qwen/context", worker.backend_factory)
        self.assertEqual(reply["type"], "result")
        self.assertEqual(reply["result"]["diagnostics"]["pcmSha256"], pcm_hash)
        self.assertEqual(self.fake.audio, [(payload, 5, 16000)])
        with self.assertRaisesRegex(worker.ProtocolError, "PCM hash mismatch"):
            worker.handle_request({**header, "sha256": "0" * 64}, payload,
                                  backend, "qwen/context", worker.backend_factory)
        self.assertEqual(len(self.fake.arguments("speech_request_new")), 1)

    def test_runtime_or_artifact_drift_is_rejected_before_cdll(self):
        with patch.object(qwen.platform, "python_version", return_value="3.13.3"):
            with self.assertRaisesRegex(RuntimeError, "Python 3.13.2"):
                self.backend.load()
        for bad_path in OPTIONS.values():
            with self.subTest(path=Path(bad_path).name):
                self.digests.side_effect = lambda path: "0" * 64 if str(path) == bad_path else {
                    OPTIONS["libraryPath"]: qwen.LIBRARY_SHA256,
                    OPTIONS["modelPath"]: qwen.MODEL_SHA256,
                }[str(path)]
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    self.backend.load()
        self.loader.assert_not_called()

    def test_all_load_status_failures_release_even_partial_handles(self):
        failures = [("speech_option_from_name", 1), ("speech_option_from_name", 2),
                    ("speech_load_params_new", 1), ("speech_load_params_set_device", 1),
                    ("speech_load_params_set_threads", 1), ("speech_load_params_set_warmup", 1),
                    ("speech_model_load", 1), ("speech_model_get_info", 1)]
        for failure in failures:
            with self.subTest(failure=failure):
                fake = FakeLibrary()
                fake.fail = failure
                self.loader.return_value = fake
                backend = qwen.Backend(OPTIONS)
                with self.assertRaisesRegex(RuntimeError, "speech status 17: fake failure"):
                    backend.load()
                backend.close()
                self.assertEqual(fake.live, {})
                self.assertEqual(fake.arguments("speech_transcribe"), [])

    def test_version_capabilities_and_model_info_errors_release_model(self):
        cases = [("speech_version", b"0.9.0"), ("speech_version", None),
                 ("speech_model_info_takes", 0), ("speech_model_info_option_steers", 0),
                 ("speech_model_info_json", b"{"), ("speech_model_info_json", b"[]"),
                 ("speech_model_info_json", None)]
        for name, response in cases:
            with self.subTest(name=name, response=response):
                fake = FakeLibrary()
                fake.responses[name] = response
                self.loader.return_value = fake
                backend = qwen.Backend(OPTIONS)
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    backend.load()
                backend.close()
                self.assertEqual(fake.live, {})

    def test_missing_symbol_and_null_load_handles_fail_without_leaks(self):
        del self.fake.speech_request_set_audio
        with self.assertRaises(AttributeError):
            self.backend.load()
        self.assertEqual(self.fake.live, {})
        for name in ("speech_load_params_new", "speech_model_load", "speech_model_get_info"):
            with self.subTest(name=name):
                fake = FakeLibrary()
                fake.null = name
                self.loader.return_value = fake
                backend = qwen.Backend(OPTIONS)
                with self.assertRaisesRegex(RuntimeError, "null"):
                    backend.load()
                self.assertEqual(fake.live, {})

    def test_request_status_failures_release_request_and_keep_model(self):
        self.backend.load()
        failures = [("speech_request_new", 1), ("speech_request_set_string", 1),
                    ("speech_request_set_string", 2), ("speech_request_set_audio", 1),
                    ("speech_transcribe", 1)]
        for name, occurrence in failures:
            with self.subTest(name=name, occurrence=occurrence):
                count = len(self.fake.arguments(name))
                self.fake.fail = (name, count + occurrence)
                with self.assertRaisesRegex(RuntimeError, "speech status 17: fake failure"):
                    self.backend.transcribe(array.array("f", [0.25]))
                self.assertEqual(list(self.fake.live.values()), ["model"])
                self.fake.fail = None
        self.assertEqual(self.fake.arguments("speech_request_result"), [])
        self.assertEqual(self.backend.transcribe(array.array("f", [0.25]))["text"], " raw text\n")

    def test_null_request_result_text_and_invalid_utf8_release_request(self):
        self.backend.load()
        self.fake.null = "speech_request_new"
        with self.assertRaisesRegex(RuntimeError, "null handle"):
            self.backend.transcribe(array.array("f", [0]))
        self.fake.null = None
        for name, value in (("speech_request_result", None), ("speech_result_text", None),
                            ("speech_result_text", b"\xff")):
            with self.subTest(name=name, value=value):
                previous = self.fake.responses[name]
                self.fake.responses[name] = value
                with self.assertRaises((RuntimeError, UnicodeDecodeError)):
                    self.backend.transcribe(array.array("f", [0]))
                self.assertEqual(list(self.fake.live.values()), ["model"])
                self.fake.responses[name] = previous

    def test_invalid_samples_fail_before_request_creation(self):
        self.backend.load()
        invalid = [[], b"\0\0\0\0", array.array("d", [0]), array.array("i", [0]),
                   memoryview(array.array("f", [0, 1, 2]))[::2],
                   memoryview(bytes(16)).cast("f", shape=[2, 2])]
        invalid.extend(array.array("f", [value]) for value in (float("nan"), float("inf"), -float("inf")))
        for samples in invalid:
            with self.subTest(sample_type=type(samples).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    self.backend.transcribe(samples)
        with patch.object(qwen.sys, "byteorder", "big"):
            with self.assertRaisesRegex(ValueError, "little endian"):
                self.backend.transcribe(array.array("f", [0]))
        self.assertEqual(self.fake.arguments("speech_request_new"), [])

    def test_representation_change_is_detected_before_native_request(self):
        self.backend.load()
        samples = array.array("f", [-0.0])

        class ConvertingArray(array.array):
            def frombytes(self, payload):
                super().frombytes(payload)
                self[0] = 0.0

        with patch.object(qwen.array, "array", ConvertingArray):
            with self.assertRaisesRegex(ValueError, "PCM identity mismatch"):
                self.backend.transcribe(samples)
        self.assertEqual(self.fake.arguments("speech_request_new"), [])

    def test_missing_error_detail_still_releases_request(self):
        self.backend.load()
        self.fake.responses["speech_last_error"] = None
        self.fake.fail = ("speech_transcribe", 1)
        with self.assertRaisesRegex(RuntimeError, "speech status 17: no error detail"):
            self.backend.transcribe(array.array("f", [0]))
        self.assertEqual(list(self.fake.live.values()), ["model"])


class DigestTests(unittest.TestCase):
    def test_digest_reads_exact_file_bytes(self):
        with tempfile.TemporaryDirectory(prefix="speakeasy-qwen-test-") as directory:
            path = Path(directory) / "fake-artifact"
            payload = b"\0pinned artifact bytes\xff"
            path.write_bytes(payload)
            self.assertEqual(qwen._digest(path), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
