"""Standard library bridge tests with a scripted worker. No model or Metal runtime is loaded."""

import array
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import names
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
import hashlib, json, struct, sys
with open(sys.argv[0] + ".argv", "w") as log: json.dump(sys.argv[1:], log)
mode = [arg for arg in sys.argv[1:] if arg.startswith("--") and not arg.startswith("--names-json=")]
names = [json.loads(arg[len("--names-json="):]) for arg in sys.argv[1:] if arg.startswith("--names-json=")]
ready = {{"kind": "ready", "backend": "{backend}", "device": "MTL0", "variant": "{variant}", "loadMs": 5}}
if names and {names_fields}:
    applied = names[0][{applied}]
    ready.update(names=len(applied), namesDigest=hashlib.sha256(
        json.dumps(applied, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
print(json.dumps(ready), flush=True)
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

    def runner(self, variant="unified-en-0.6b", backend="metal", names_fields=True, applied=":"):
        path = self.root / "worker"
        path.write_text(FAKE_WORKER.format(python=sys.executable, variant=variant, backend=backend,
                                           names_fields=names_fields, applied=applied))
        os.chmod(path, 0o755)
        return path

    def argv(self, runner):
        return json.loads(Path(str(runner) + ".argv").read_text())

    def backend(self, runner, setup_id="hex/unified", **options):
        return hexu.Backend({"runnerPath": str(runner), "modelPath": str(self.model), "setupId": setup_id, **options})

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
        self.assertIn("the Names list as vocabulary at boost 2.5", metadata["pipeline"])
        self.assertEqual((metadata["names"], metadata["namesDigest"]), (0, names.names_digest([])))
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

    def test_names_workers_get_the_names_list_as_one_argv_flag_and_must_echo_it(self):
        listed = ["Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen"]
        for setup_id, variant, kind, mode in (("hex/vocabulary", "unified-en-0.6b", "metal", "--whole=vocabulary"),
                                              ("cohere/transcribe", "cohere-transcribe-03-2026", "metal", "--whole=cohere"),
                                              ("apple/dictation", "dictation-transcriber", "apple", "--module=dictation")):
            with self.subTest(setup_id=setup_id):
                runner = self.runner(variant, kind)
                model = [str(self.model)] if hexu.SETUPS[setup_id]["file"] else []
                backend = hexu.Backend({"setupId": setup_id, "runnerPath": str(runner), "names": listed,
                                        **({"modelPath": model[0]} if model else {})})
                with self.pinned(setup_id) if model else contextlib.nullcontext():
                    metadata = backend.load()
                self.assertEqual(self.argv(runner), [*model, mode,
                                                     '--names-json=["Grok","Claude","Codex","GPT","GPT-SOL","TM","Hibiki","Qwen"]'])
                self.assertEqual((metadata["names"], metadata["namesDigest"]),
                                 (8, "59d6626926c652e50c7e90fea5f95afdea75f6c27e11aa07c2996b89d51fc068"))
                backend.close()
        unicode = self.runner("cohere-transcribe-03-2026")
        siobhan = self.backend(unicode, "cohere/transcribe", names=["Siobhán"])
        with self.pinned("cohere/transcribe"):
            siobhan.load()
        siobhan.close()
        self.assertEqual(self.argv(unicode)[-1], '--names-json=["Siobhán"]', "non ASCII stays unescaped, as the digest")
        empty = self.runner("cohere-transcribe-03-2026")
        cohere = self.backend(empty, "cohere/transcribe")
        with self.pinned("cohere/transcribe"):
            self.assertEqual(cohere.load()["names"], 0)
        self.assertEqual(self.argv(empty)[-1], "--names-json=[]", "an absent list is sent as an explicit empty one")
        samples = memoryview(array.array("f", [0.1, -0.2]).tobytes()).cast("f")
        self.assertEqual(cohere.transcribe(samples, {"namesBias": 2.5})["text"], "heard 2 --whole=cohere names 2.5",
                         "an empty list still sends and echoes the bias")
        cohere.close()

    def test_a_runner_that_predates_names_or_applies_another_list_fails_load(self):
        for fake, message in (({"names_fields": False}, "predates names"), ({"applied": "1:"}, "different names list")):
            with self.subTest(message=message):
                backend = self.backend(self.runner(**fake), "hex/vocabulary", names=["Grok", "Qwen"])
                with self.pinned("hex/vocabulary"), self.assertRaisesRegex(RuntimeError, message):
                    backend.load()
                self.assertIsNone(backend.process, "a failed load retires the worker")

    def test_names_are_refused_where_the_worker_takes_none_and_validated_where_it_does(self):
        runner = self.runner()
        with self.assertRaisesRegex(ValueError, "optional setupId"):
            self.backend(runner, "hex/unified", names=["Grok"])
        with self.assertRaisesRegex(ValueError, "optional setupId"):
            hexu.Backend({"setupId": "apple/speech", "runnerPath": str(runner), "names": []})
        for bad in ("Grok", ["Grok", "Grok"], [" Grok"], ["Gr\x00ok"], ["a,b"], ["x" * 129], [str(n) for n in range(65)], [1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.backend(runner, "cohere/transcribe", names=bad)

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

    def test_apple_dictation_runs_the_same_worker_as_a_module_with_frequent_finalization_on_request(self):
        runner = self.runner("dictation-transcriber", "apple")
        backend = hexu.Backend({"setupId": "apple/dictation", "runnerPath": str(runner)})
        self.assertEqual((backend.load()["modelSha256"], self.argv(runner)), (None, ["--module=dictation", "--names-json=[]"]),
                         "an absent list is sent as an explicit empty one")
        samples = memoryview(array.array("f", [0.1] * 3).tobytes()).cast("f")
        self.assertEqual(backend.transcribe(samples)["text"], "heard 3 --module=dictation")
        self.assertEqual(backend.feed(samples, True)["tentative"], " at 0+0", "Off by default")
        self.assertEqual(backend.feed(samples, True, {"frequentFinalization": 0})["tentative"], " at 0+0")
        self.assertEqual(backend.feed(samples, True, {"frequentFinalization": 1})["tentative"], " at 1+0")
        for bad in ({"fastResults": 1}, {"frequentFinalization": 2}, {"frequentFinalization": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backend.feed(samples, True, bad)
        backend.close()
        self.assertEqual(worker.BACKENDS["apple/dictation"], "native_asr")

    def test_apple_dictation_refuses_a_speech_only_worker(self):
        backend = hexu.Backend({"setupId": "apple/dictation", "runnerPath": str(self.runner("speech-transcriber", "apple"))})
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            backend.load()
        self.assertIsNone(backend.process)

    def test_identity_mismatch_retires_the_worker(self):
        backend = self.backend(self.runner(variant="other"))
        with self.pinned():
            with self.assertRaisesRegex(RuntimeError, "identity"):
                backend.load()
        self.assertIsNone(backend.process)


if __name__ == "__main__":
    unittest.main()
