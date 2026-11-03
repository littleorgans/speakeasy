"""Standard library bridge tests with a scripted worker. No model or Metal runtime is loaded."""

import array
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import native_asr as hexu
import pins
import worker


def setUpModule():
    """Model hashes cache in a temporary directory, never the owner's cache."""
    directory = tempfile.TemporaryDirectory(prefix="speakeasy-digest-test-")
    unittest.addModuleCleanup(directory.cleanup)
    cache = patch.object(pins, "DIGEST_CACHE", Path(directory.name))
    cache.start()
    unittest.addModuleCleanup(cache.stop)


FAKE_WORKER = """#!{python}
import json, struct, sys
mode = sys.argv[2:]
print(json.dumps({{"kind": "ready", "backend": "{backend}", "device": "MTL0", "variant": "{variant}", "loadMs": 5}}), flush=True)
fed = 0
while True:
    head = sys.stdin.buffer.read(5)
    if len(head) < 5: break
    op, count = head[:1], struct.unpack("<I", head[1:])[0]
    lookahead = struct.unpack("<ii", sys.stdin.buffer.read(8)) if op == b"S" else None
    bias = struct.unpack("<f", sys.stdin.buffer.read(4))[0] if op == b"B" else 0
    samples = struct.unpack("<%df" % count, sys.stdin.buffer.read(count * 4))
    if op in (b"T", b"B"):
        print(json.dumps({{"kind": "result", "text": " ".join(["heard %d" % len(samples)] + mode + (["names %g" % bias] if op == b"B" else [])),
                          "rawText": " heard", "processingMs": 2.5, "pauseDroppedSamples": 0, "namesBias": bias}}), flush=True)
    else:
        if lookahead: opened = "%d+%d" % lookahead
        fed = (0 if op == b"S" else fed) + len(samples)
        print(json.dumps({{"kind": "partial", "stable": "fed %d" % fed, "tentative": " at " + opened, "processingMs": 1.5}}), flush=True)
"""


class HexUnifiedTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.model = self.root / hexu.HEX_MODEL["file"]
        self.model.write_bytes(b"fake gguf")
        self.addCleanup(shutil.rmtree, self.root)

    def runner(self, variant="unified-en-0.6b", backend="metal"):
        path = self.root / "worker"
        path.write_text(FAKE_WORKER.format(python=sys.executable, variant=variant, backend=backend))
        os.chmod(path, 0o755)
        return path

    def backend(self, runner, setup_id="hex/unified"):
        return hexu.Backend({"runnerPath": str(runner), "modelPath": str(self.model), "setupId": setup_id})

    def pinned(self, setup_id="hex/unified"):
        return patch.dict(hexu.SETUPS[setup_id], {"sha256": hashlib.sha256(b"fake gguf").hexdigest()})

    def test_worker_routes_the_setup_to_this_backend(self):
        self.assertEqual(worker.BACKENDS["hex/unified"], "native_asr")

    def test_options_require_absolute_paths_and_this_setup(self):
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "relative", "modelPath": str(self.model)})
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "/w", "modelPath": "/m", "setupId": "qwen/context"})
        with self.assertRaises(ValueError):
            hexu.Backend({"runnerPath": "/w", "modelPath": "/m", "libraryPath": "/l"})

    def test_load_checks_the_model_hash_before_starting_the_worker(self):
        backend = self.backend(self.runner())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            backend.load()
        self.assertIsNone(backend.process)

    def test_load_transcribe_and_close_through_the_worker(self):
        backend = self.backend(self.runner())
        with self.pinned():
            metadata = backend.load()
        self.assertEqual(metadata["setupId"], "hex/unified")
        self.assertEqual(metadata["device"], "MTL0")
        samples = memoryview(array.array("f", [0.1, -0.2, 0.3]).tobytes()).cast("f")
        result = backend.transcribe(samples)
        self.assertEqual(result["text"], "heard 3")
        self.assertEqual(result["rawText"], " heard")
        self.assertEqual(result["processingMs"], 2.5)
        worker.validate_result(result)
        process = backend.process
        backend.close()
        self.assertEqual(process.returncode, 0)
        self.assertIsNone(backend.process)

    def test_feed_streams_live_words_and_start_opens_a_fresh_stream(self):
        backend = self.backend(self.runner())
        with self.pinned():
            backend.load()
        samples = lambda count: memoryview(array.array("f", [0.1] * count).tobytes()).cast("f")
        first = backend.feed(samples(3), True)
        self.assertEqual(first, {"stable": "fed 3", "tentative": " at 1040+1040", "processingMs": 1.5},
                         "without settings the stream opens at Hex's own 2.08 s lookahead")
        worker.validate_partial(first)
        self.assertEqual(backend.feed(samples(2), False)["stable"], "fed 5")
        self.assertEqual(backend.transcribe(samples(4))["text"], "heard 4", "a take follows a live stream")
        fresh = backend.feed(samples(1), True, {"lookaheadMs": 480})
        self.assertEqual((fresh["stable"], fresh["tentative"]), ("fed 1", " at 160+320"), "the chosen lookahead reaches the worker")
        for bad in ({"lookaheadMs": 500}, {"lookaheadMs": True}, {"chunkMs": 1040}, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backend.feed(samples(1), True, bad)
        with self.assertRaises(ValueError):
            backend.feed(samples(1), False, {"lookaheadMs": 480})
        process = backend.process
        backend.close()
        self.assertEqual(process.returncode, 0)

    def test_whole_take_setups_start_the_worker_in_their_mode_and_give_final_text_only(self):
        backend = self.backend(self.runner(), "hex/vocabulary")
        with self.pinned("hex/vocabulary"):
            metadata = backend.load()
        self.assertEqual(metadata["setupId"], "hex/vocabulary")
        self.assertIn("Claude GPT Grok Codex TM whoami", metadata["pipeline"])
        samples = memoryview(array.array("f", [0.1, -0.2]).tobytes()).cast("f")
        self.assertEqual(backend.transcribe(samples)["text"], "heard 2 --whole=vocabulary")
        with self.assertRaisesRegex(ValueError, "final text only"):
            backend.feed(samples, True)
        backend.close()
        cohere = self.backend(self.runner("cohere-transcribe-03-2026"), "cohere/transcribe")
        with self.pinned("cohere/transcribe"):
            cohere.load()
        self.assertEqual(cohere.transcribe(samples)["text"], "heard 2 --whole=cohere")
        cohere.close()
        with self.assertRaisesRegex(RuntimeError, "identity"), self.pinned("cohere/transcribe"):
            self.backend(self.runner(), "cohere/transcribe").load()
        self.assertEqual(worker.BACKENDS["hex/vocabulary"], "native_asr")
        self.assertEqual(worker.BACKENDS["cohere/transcribe"], "native_asr")

    def test_cohere_names_bias_travels_with_each_take_and_off_sends_todays_take(self):
        cohere = self.backend(self.runner("cohere-transcribe-03-2026"), "cohere/transcribe")
        with self.pinned("cohere/transcribe"):
            cohere.load()
        samples = memoryview(array.array("f", [0.1, -0.2]).tobytes()).cast("f")
        self.assertEqual(cohere.transcribe(samples, {"namesBias": 2.5})["text"], "heard 2 --whole=cohere names 2.5")
        self.assertEqual(cohere.transcribe(samples, {"namesBias": 1.25})["text"], "heard 2 --whole=cohere names 1.25",
                         "no reload between takes")
        for off in (None, {}, {"namesBias": 0}):
            with self.subTest(off=off):
                self.assertEqual(cohere.transcribe(samples, off)["text"], "heard 2 --whole=cohere", "Off is today's T take")
        for bad in ({"namesBias": 1}, {"namesBias": True}, {"names": 2.5}, [2.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cohere.transcribe(samples, bad)
        cohere.close()
        hex_backend = self.backend(self.runner())
        with self.pinned():
            hex_backend.load()
        with self.assertRaisesRegex(ValueError, "no final settings"):
            hex_backend.transcribe(samples, {"namesBias": 2.5})
        hex_backend.close()

    def test_apple_needs_no_model_file_and_streams_with_fast_results_on_request(self):
        runner = self.runner("speech-transcriber", "apple")
        with self.assertRaises(ValueError):
            hexu.Backend({"setupId": "apple/speech", "runnerPath": str(runner), "modelPath": str(self.model)})
        backend = hexu.Backend({"setupId": "apple/speech", "runnerPath": str(runner)})
        metadata = backend.load()
        self.assertEqual((metadata["modelSha256"], metadata["runtime"]), (None, "Apple Speech, system managed weights"))
        samples = memoryview(array.array("f", [0.1] * 3).tobytes()).cast("f")
        self.assertEqual(backend.feed(samples, True)["tentative"], " at 0+0", "volatile results by default")
        self.assertEqual(backend.feed(samples, True, {"fastResults": 1})["tentative"], " at 1+0")
        with self.assertRaises(ValueError):
            backend.feed(samples, True, {"lookaheadMs": 480})
        self.assertEqual(backend.transcribe(samples)["text"], "heard 3")
        backend.close()
        self.assertEqual(worker.BACKENDS["apple/speech"], "native_asr")

    def test_identity_mismatch_retires_the_worker(self):
        backend = self.backend(self.runner(variant="other"))
        with self.pinned():
            with self.assertRaisesRegex(RuntimeError, "identity"):
                backend.load()
        self.assertIsNone(backend.process)


if __name__ == "__main__":
    unittest.main()
