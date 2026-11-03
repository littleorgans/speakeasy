"""Transport tests use the standard library and fake backends only."""

import contextlib
import hashlib
import io
import json
import math
import struct
import unittest

import worker


SETUP = "mlx-parakeet/v3-greedy"
GENERATION = "test-generation"


def frame(operation, request_id=1, body=b"", setup_id=SETUP, **fields):
    header = {"version": 1, "generation": GENERATION, "id": request_id,
              "setupId": setup_id, "operation": operation, "pcmBytes": len(body), **fields}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(encoded)) + encoded + body


def identity(body):
    return {"sampleRate": 16_000, "channels": 1, "sampleCount": len(body) // 4,
            "pcmBytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


class Fragmented(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(3, count) if count >= 0 else 3)


class Backend:
    def __init__(self):
        self.calls = []

    def load(self):
        print("backend diagnostics")
        return {"revision": "pinned", "loadMs": 1}

    def transcribe(self, samples):
        self.calls.append(samples.tobytes())
        return {"text": "", "rawText": "", "processingMs": 0}

    def cleanup(self, text):
        self.calls.append(text)
        return {"text": text, "processingMs": 2, "diagnostics": {"scope": "fake"}}


class WorkerTests(unittest.TestCase):
    def run_worker(self, data, setup_id=SETUP, backend=None, fragmented=False):
        backend = backend or Backend()
        loaded = []

        def factory(selected, options):
            loaded.append((selected, options))
            return backend

        target = io.BytesIO()
        source = Fragmented(data) if fragmented else io.BytesIO(data)
        with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
            code = worker.serve(source, target, setup_id, GENERATION, factory)
        target.seek(0)
        replies = []
        while (reply := worker.read_frame(target)) is not None:
            replies.append(reply[0])
        return code, replies, backend, loaded, diagnostics.getvalue()

    def test_fragmented_coalesced_persistent_and_exact_pcm(self):
        body = struct.pack("<5f", 0.0, -0.0, 1.5, -2.0, 0.125)
        fields = identity(body)
        fields.pop("pcmBytes")
        data = (frame("load", options={"modelPath": "fake"})
                + frame("transcribe", 2, body, **fields)
                + frame("transcribe", 3, b"", **{k: v for k, v in identity(b"").items() if k != "pcmBytes"})
                + frame("close", 4))
        for fragmented in (False, True):
            with self.subTest(fragmented=fragmented):
                code, replies, backend, loaded, diagnostic = self.run_worker(data, fragmented=fragmented)
                self.assertEqual(code, 0)
                self.assertEqual([r["type"] for r in replies], ["ready", "result", "result", "closed"])
                self.assertEqual(backend.calls, [body, b""])
                self.assertEqual(len(loaded), 1)
                self.assertEqual(replies[1]["result"]["text"], "")
                self.assertIn("backend diagnostics", diagnostic)

    def test_feed_answers_live_words_only_where_the_backend_streams(self):
        class Streaming(Backend):
            def feed(self, samples, start, settings):
                self.calls.append((samples.tobytes(), start, settings))
                return {"stable": " heard", "tentative": " maybe", "processingMs": 4}

        body = struct.pack("<2f", 0.5, -0.25)
        fields = {k: v for k, v in identity(body).items() if k != "pcmBytes"}
        data = (frame("load", options={}) + frame("feed", 2, body, start=True, settings={"updateMs": 800}, **fields)
                + frame("feed", 3, b"", start=False, **{k: v for k, v in identity(b"").items() if k != "pcmBytes"}))
        code, replies, backend, _, _ = self.run_worker(data, backend=Streaming())
        self.assertEqual(code, 0)
        self.assertEqual([r["type"] for r in replies], ["ready", "partial", "partial"])
        self.assertEqual(replies[1]["partial"], {"stable": " heard", "tentative": " maybe", "processingMs": 4})
        self.assertEqual(backend.calls, [(body, True, {"updateMs": 800}), (b"", False, None)])
        refused = {
            "final text only": (frame("load", options={}) + frame("feed", 2, body, start=True, **fields), Backend()),
            "start must be a flag": (frame("load", options={}) + frame("feed", 2, body, start=1, **fields), Streaming()),
            "settings go with start": (frame("load", options={}) + frame("feed", 2, body, start=False, settings={}, **fields),
                                       Streaming()),
            "settings are an object": (frame("load", options={}) + frame("feed", 2, body, start=True, settings=[1], **fields),
                                       Streaming()),
            "hash checked": (frame("load", options={}) + frame("feed", 2, body, start=True, **{**fields, "sha256": "0" * 64}),
                             Streaming()),
        }
        for label, (request, owner) in refused.items():
            with self.subTest(label):
                code, replies, owner, _, _ = self.run_worker(request, backend=owner)
                self.assertEqual(code, 1)
                self.assertEqual(replies[-1]["type"], "error")
                self.assertFalse(any(isinstance(call, tuple) for call in owner.calls))
        for bad in ({"stable": 1, "tentative": "", "processingMs": 0}, {"stable": "", "tentative": "", "processingMs": -1},
                    {"stable": "", "tentative": "", "processingMs": 0, "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(worker.ProtocolError):
                worker.validate_partial(bad)

    def test_transcribe_passes_final_settings_only_when_given(self):
        class Final(Backend):
            def transcribe(self, samples, settings=None):
                self.calls.append((samples.tobytes(), settings))
                return {"text": "", "processingMs": 0}

        body = struct.pack("<2f", 0.5, -0.25)
        fields = {k: v for k, v in identity(body).items() if k != "pcmBytes"}
        data = (frame("load", options={}) + frame("transcribe", 2, body, settings={"namesBias": 2.5}, **fields)
                + frame("transcribe", 3, body, **fields))
        code, replies, backend, _, _ = self.run_worker(data, backend=Final())
        self.assertEqual(code, 0)
        self.assertEqual([r["type"] for r in replies], ["ready", "result", "result"])
        self.assertEqual(backend.calls, [(body, {"namesBias": 2.5}), (body, None)])
        code, replies, backend, _, _ = self.run_worker(
            frame("load", options={}) + frame("transcribe", 2, body, settings=[2.5], **fields), backend=Final())
        self.assertEqual((code, replies[-1]["type"], backend.calls), (1, "error", []))

    def test_cleanup_exact_text_empty_and_base_identity(self):
        setup = "mlx-cleanup/qwen-targeted"
        for text in ("", "  café\n{\"A\":\"launch\"}  "):
            with self.subTest(text=text):
                data = frame("load", setup_id=setup, options={}) + frame(
                    "cleanup", 2, setup_id=setup, text=text, B=None, basePcm=identity(b""),
                    transcriptSha256=hashlib.sha256(text.encode()).hexdigest())
                code, replies, backend, _, _ = self.run_worker(data, setup_id=setup)
                self.assertEqual(code, 0)
                self.assertEqual(backend.calls, [text])
                self.assertEqual(replies[-1]["result"]["text"], text)

    def test_backend_closes_once_before_normal_exit_or_error_reply(self):
        for ending in ("close", "eof", "protocol-error", "backend-error", "load-error"):
            with self.subTest(ending=ending):
                events = []
                terminal_snapshots = []

                class OwnedBackend(Backend):
                    def load(self):
                        events.append("load")
                        if ending == "load-error":
                            raise RuntimeError("load failed after allocation")
                        return super().load()

                    def transcribe(self, samples):
                        raise RuntimeError("decode failed")

                    def close(self):
                        events.append("close")
                        print("close diagnostic")

                class Target(io.BytesIO):
                    def write(self, data):
                        header = json.loads(data[4:])
                        if header["type"] in ("closed", "error"):
                            terminal_snapshots.append(list(events))
                        return super().write(data)

                backend = OwnedBackend()
                tail = {"close": frame("close", 2), "eof": b"",
                        "protocol-error": b"\x01", "load-error": b"",
                        "backend-error": frame("transcribe", 2, **{
                            key: value for key, value in identity(b"").items() if key != "pcmBytes"
                        })}[ending]
                with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    code = worker.serve(io.BytesIO(frame("load", options={}) + tail), Target(),
                                        SETUP, GENERATION, lambda *_: backend)
                self.assertEqual(code, 0 if ending in ("close", "eof") else 1)
                self.assertEqual(events, ["load", "close"])
                self.assertEqual(terminal_snapshots, [] if ending in ("eof", "protocol-error")
                                 else [["load", "close"]])
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("close diagnostic", stderr.getvalue())

    def test_malformed_frames_never_load(self):
        malformed = [b"\x01", struct.pack("<I", 0), struct.pack("<I", worker.MAX_HEADER_BYTES + 1),
                     struct.pack("<I", 1) + b"\xff", struct.pack("<I", 1) + b"{",
                     struct.pack("<I", 2) + b"[]",
                     struct.pack("<I", 13) + b'{"pcmBytes":0}',
                     frame("load", options={}, pcmBytes=worker.MAX_PCM_BYTES + 4),
                     frame("load", options={}, pcmBytes=True),
                     frame("load", options={}, pcmBytes=3),
                     frame("load", options={}, pcmBytes=4)]
        # Duplicate keys and nonfinite JSON numbers have no valid envelope.
        for raw in (b'{"pcmBytes":0,"pcmBytes":0}', b'{"pcmBytes":NaN}'):
            malformed.append(struct.pack("<I", len(raw)) + raw)
        for data in malformed:
            with self.subTest(data=data[:40]):
                code, _, backend, loaded, _ = self.run_worker(data)
                self.assertEqual(code, 1)
                self.assertEqual(loaded, [])
                self.assertEqual(backend.calls, [])

    def test_envelope_operation_and_state_failures_exit(self):
        bad = [frame("load", options={}, generation="stale"),
               frame("load", options={}, id=True),
               frame("load", options={}, version=True),
               frame("load", options={}, setupId="other"),
               frame("transcribe"), frame("unknown"),
               frame("load", options={}) + frame("load", 2, options={}),
               frame("load", options={}) + frame("close", 1)]
        for data in bad:
            with self.subTest(data=data):
                code, _, backend, _, _ = self.run_worker(data)
                self.assertEqual(code, 1)
                self.assertEqual(backend.calls, [])

    def test_pcm_corruption_and_nonfinite_rejected_before_backend(self):
        cases = []
        for value in (math.nan, math.inf, -math.inf):
            body = struct.pack("<f", value)
            cases.append((body, identity(body)))
        body = struct.pack("<f", 0.5)
        for key, value in (("sha256", "0" * 64), ("sampleCount", 2), ("sampleCount", True),
                           ("sampleRate", 48_000), ("channels", 2), ("sha256", "no")):
            cases.append((body, {**identity(body), key: value}))
        for body, fields in cases:
            with self.subTest(fields=fields):
                fields.pop("pcmBytes")
                code, replies, backend, _, _ = self.run_worker(
                    frame("load", options={}) + frame("transcribe", 2, body, **fields))
                self.assertEqual(code, 1)
                self.assertEqual(replies[-1]["type"], "error")
                self.assertEqual(backend.calls, [])

    def test_cleanup_hash_bounds_and_operation_rejected(self):
        setup = "mlx-cleanup/qwen-targeted"
        valid = {"text": "", "B": None, "basePcm": identity(b""),
                 "transcriptSha256": hashlib.sha256(b"").hexdigest()}
        for patch in ({"B": "other"}, {"text": "x"}, {"basePcm": {}},
                      {"basePcm": {**identity(b""), "samples": []}},
                      {"text": "x" * (worker.MAX_TEXT_BYTES + 1)}):
            with self.subTest(patch=tuple(patch)):
                code, _, backend, _, _ = self.run_worker(
                    frame("load", setup_id=setup, options={})
                    + frame("cleanup", 2, setup_id=setup, **{**valid, **patch}), setup_id=setup)
                self.assertEqual(code, 1)
                self.assertEqual(backend.calls, [])
        code, _, _, _, _ = self.run_worker(frame("load", setup_id=setup, options={})
                                          + frame("transcribe", 2, setup_id=setup), setup_id=setup)
        self.assertEqual(code, 1)

    def test_backend_exception_preserves_partial_text_and_exits(self):
        class PartialError(Exception):
            rawText = "partial\n"

        class Failing(Backend):
            def cleanup(self, text):
                raise PartialError("generation failed")

        setup = "mlx-cleanup/qwen-targeted"
        data = frame("load", setup_id=setup, options={}) + frame(
            "cleanup", 2, setup_id=setup, text="", B=None, basePcm=identity(b""),
            transcriptSha256=hashlib.sha256(b"").hexdigest()) + frame("close", 3, setup_id=setup)
        code, replies, _, _, _ = self.run_worker(data, setup, Failing())
        self.assertEqual(code, 1)
        self.assertEqual([r["type"] for r in replies], ["ready", "error"])
        self.assertEqual(replies[-1]["rawText"], "partial\n")

    def test_invalid_backend_result_or_oversized_reply_is_fatal(self):
        class Invalid(Backend):
            def transcribe(self, samples):
                return self.result

        for result in ({"text": "", "processingMs": math.nan}, {"text": "", "processingMs": True},
                       {"text": "", "processingMs": 1, "decodeMs": -1},
                       {"text": None, "processingMs": 1}, {"text": "", "processingMs": 1, "rawText": 7},
                       {"text": "x" * worker.MAX_HEADER_BYTES, "processingMs": 1}):
            with self.subTest(keys=tuple(result)):
                backend = Invalid()
                backend.result = result
                fields = {k: v for k, v in identity(b"").items() if k != "pcmBytes"}
                code, replies, _, _, _ = self.run_worker(
                    frame("load", options={}) + frame("transcribe", 2, **fields), backend=backend)
                self.assertEqual(code, 1)
                self.assertEqual(replies[-1]["type"], "error")

    def test_voice_speak_replies_with_exact_pcm_and_never_transcribes(self):
        voice_id = "kokoro-mlx-bf16"
        pcm = struct.pack("<3f", 0.25, -0.0, -1.0)

        class Voice(Backend):
            def speak(self, text, voice, speed):
                self.calls.append((text, voice, speed))
                return {"samples": self.samples, "sampleRate": 24_000, "processingMs": 4}

        backend = Voice()
        backend.samples = pcm
        data = (frame("load", setup_id=voice_id, options={})
                + frame("speak", 2, setup_id=voice_id, text="Clear the screen.", voice="af_heart", speed=1)
                + frame("close", 3, setup_id=voice_id))
        target = io.BytesIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = worker.serve(io.BytesIO(data), target, voice_id, GENERATION, lambda selected, options: backend)
        target.seek(0)
        replies = []
        while (reply := worker.read_frame(target)) is not None:
            replies.append(reply)
        self.assertEqual(code, 0)
        self.assertEqual([header["type"] for header, _ in replies], ["ready", "audio", "closed"])
        audio, body = replies[1]
        self.assertEqual(body, pcm)
        self.assertEqual(audio["audio"], {"sampleRate": 24_000, "sampleCount": 3,
                                          "sha256": hashlib.sha256(pcm).hexdigest()})
        self.assertEqual(backend.calls, [("Clear the screen.", "af_heart", 1.0)])
        cases = [(voice_id, frame("transcribe", 2, setup_id=voice_id,
                                  **{k: v for k, v in identity(b"").items() if k != "pcmBytes"}), pcm),
                 (SETUP, frame("speak", 2, text="Hi.", voice="af_heart", speed=1), pcm),
                 (voice_id, frame("speak", 2, setup_id=voice_id, text=" ", voice="af_heart", speed=1), pcm),
                 (voice_id, frame("speak", 2, setup_id=voice_id, text="Hi.", voice="af_heart", speed=3), pcm),
                 (voice_id, frame("speak", 2, setup_id=voice_id, text="Hi.", voice="af_heart", speed=1),
                  struct.pack("<f", math.nan)),
                 (voice_id, frame("speak", 2, setup_id=voice_id, text="Hi.", voice="af_heart", speed=1), b"\0" * 3)]
        for setup_id, request, samples in cases:
            with self.subTest(setup=setup_id, request=request[-60:]):
                failing = Voice()
                failing.samples = samples
                code, replies, _, _, _ = self.run_worker(frame("load", setup_id=setup_id, options={}) + request,
                                                         setup_id=setup_id, backend=failing)
                self.assertEqual(code, 1)
                self.assertEqual(replies[-1]["type"], "error")

    def test_imports_are_lazy_and_backend_options_have_fixed_setup(self):
        from unittest.mock import patch
        module = type("Module", (), {"Backend": lambda options: options})
        with patch.object(worker.importlib, "import_module", return_value=module) as imported:
            options = worker.backend_factory(SETUP, {"setupId": "wrong", "modelPath": "fake"})
        imported.assert_called_once_with("parakeet")
        self.assertEqual(options, {"setupId": SETUP, "modelPath": "fake"})


if __name__ == "__main__":
    unittest.main()
