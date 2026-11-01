"""Standard library fakes only; no native runtime or model inference."""

from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cleanup


def response(text, token=7, finish="stop"):
    return SimpleNamespace(text=text, token=token, finish_reason=finish,
                           generation_tokens=1, prompt_tps=12.5, generation_tps=30.0)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="speakeasy-cleanup-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / cleanup.REVISION
        self.path.mkdir()
        for name in ("cleanup-prompt.txt", "requirements-cleanup.txt"):
            (self.root / name).write_bytes((cleanup.ROOT / name).read_bytes())
        self.template = "fake native template: é"
        files = {
            "config.json": json.dumps({"model_type": "qwen2", "quantization": {
                "bits": 4, "group_size": 64}}).encode(),
            "tokenizer_config.json": json.dumps({"chat_template": self.template}).encode(),
            "tokenizer.json": b"{}", "model.safetensors": b"fake weights",
            "model.safetensors.index.json": b"{}",
            "added_tokens.json": b"{}", "special_tokens_map.json": b"{}",
            "vocab.json": b"{}", "merges.txt": b"fake merges",
        }
        self.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        for name, data in files.items():
            (self.path / name).write_bytes(data)
        self.versions = dict(line.split("==") for line in
                             (self.root / "requirements-cleanup.txt").read_text().splitlines())
        self.distributions = [SimpleNamespace(metadata={"Name": name}, version=version)
                              for name, version in self.versions.items()]

        self.mx = ModuleType("mlx.core")
        self.mx.gpu = "fake gpu"
        self.mx.metal = SimpleNamespace(is_available=Mock(return_value=True))
        self.mx.set_default_device = Mock()
        self.mx.default_device = Mock(return_value=self.mx.gpu)
        self.mx.default_stream = Mock(return_value="fake gpu stream")
        self.mx.stream = Mock(side_effect=lambda stream: nullcontext())
        self.mx.eval = Mock()
        self.mx.synchronize = Mock()
        self.mx.random = SimpleNamespace(seed=Mock())
        self.mx.device_info = Mock(return_value={"device_name": "fake Metal"})
        self.mx.get_active_memory = Mock(return_value=123)
        self.model = SimpleNamespace(parameters=Mock(return_value=["fake parameters"]))
        self.tokenizer = SimpleNamespace(
            chat_template=self.template,
            apply_chat_template=Mock(side_effect=lambda *args, **kwargs: [1, 2, 3]),
        )
        self.lm = ModuleType("mlx_lm")
        self.lm.load = Mock(return_value=(self.model, self.tokenizer))
        self.lm.stream_generate = Mock(side_effect=lambda *args, **kwargs: iter([response("ok")]))
        self.sample = ModuleType("mlx_lm.sample_utils")
        self.sample.make_sampler = Mock(return_value=object())
        mlx = ModuleType("mlx")
        mlx.core = self.mx
        for item in (
            patch.object(cleanup, "ROOT", self.root),
            patch.object(cleanup, "MODEL_FILES", self.hashes),
            patch.object(cleanup, "TEMPLATE_SHA256", hashlib.sha256(
                json.dumps(self.template, ensure_ascii=False, sort_keys=True).encode()).hexdigest()),
            patch.object(cleanup.metadata, "distributions", return_value=self.distributions),
            patch.dict(sys.modules, {"mlx": mlx, "mlx.core": self.mx, "mlx_lm": self.lm,
                                     "mlx_lm.sample_utils": self.sample}),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.backend = cleanup.Backend({"modelPath": str(self.path)})

    def test_module_and_constructor_do_not_import_native_runtime_or_read_assets(self):
        import builtins
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "mlx" or name.startswith(("mlx.", "mlx_lm")):
                raise AssertionError("Eager native import")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("cleanup_lazy_probe", cleanup.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", side_effect=guarded_import):
            spec.loader.exec_module(module)
            with patch.object(module.Path, "read_bytes", side_effect=AssertionError("Eager file read")):
                backend = module.Backend({"modelPath": "/missing"})
                self.assertIsNone(backend.ready)
        self.lm.load.assert_not_called()

    def test_load_reports_identity_and_loads_once_without_generation(self):
        ready = self.backend.load()
        self.assertIs(self.backend.load(), ready)
        self.lm.load.assert_called_once_with(str(self.path), trust_remote_code=False)
        self.lm.stream_generate.assert_not_called()
        self.mx.eval.assert_called_once_with(["fake parameters"])
        self.mx.set_default_device.assert_called_once_with(self.mx.gpu)
        self.sample.make_sampler.assert_called_once_with(temp=0)
        self.assertEqual(ready["model"], cleanup.MODEL)
        self.assertEqual(ready["revision"], cleanup.REVISION)
        self.assertEqual(ready["quantization"], {"bits": 4, "group_size": 64})
        self.assertEqual(ready["promptSha256"], cleanup.PROMPT_SHA256)
        self.assertEqual(ready["chatTemplateSha256"], cleanup.TEMPLATE_SHA256)
        self.assertEqual(ready["fileSha256"], self.hashes)
        self.assertEqual(ready["packages"], cleanup.runtime_versions())
        self.assertEqual(ready["python"], "3.13.2")
        self.assertGreaterEqual(ready["loadMs"], 0)

    def test_exact_unicode_payload_native_template_and_fresh_request_controls(self):
        texts = ['  Écoute "claude"\n七 🗣️  ', " \tlaunch\n "]
        rows = [self.backend.cleanup(text) for text in texts]
        calls = self.tokenizer.apply_chat_template.call_args_list
        self.assertIsNot(calls[0].args[0], calls[1].args[0])
        for index, text in enumerate(texts):
            messages = calls[index].args[0]
            expected = json.dumps({"A": text, "B": None}, ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(messages, [
                {"role": "system", "content": (self.root / "cleanup-prompt.txt").read_text()},
                {"role": "user", "content": expected},
            ])
            self.assertEqual(calls[index].kwargs, {"tokenize": True, "add_generation_prompt": True})
            self.assertEqual(rows[index]["diagnostics"]["userPayloadSha256"],
                             hashlib.sha256(expected.encode("utf-8")).hexdigest())
            generated = self.lm.stream_generate.call_args_list[index]
            self.assertEqual(generated.args, (self.model, self.tokenizer))
            self.assertEqual(generated.kwargs, {
                "prompt": [1, 2, 3], "max_tokens": 512,
                "sampler": self.sample.make_sampler.return_value,
                "stream": "fake gpu stream", "prompt_cache": None,
                "logits_processors": None,
            })
        self.assertEqual([call.args for call in self.mx.random.seed.call_args_list], [(0,), (0,)])
        self.assertIsNot(self.lm.stream_generate.call_args_list[0].kwargs["prompt"],
                         self.lm.stream_generate.call_args_list[1].kwargs["prompt"])
        self.assertEqual(self.lm.load.call_count, 1)

    def test_blank_source_returns_empty_without_prompting_or_generation(self):
        ready = self.backend.load()
        self.lm.stream_generate.side_effect = lambda *args, **kwargs: iter(
            [response("Return only corrected text.")])
        for text in ("", " \t\r\n ", "\u00a0\u3000"):
            with self.subTest(text=text):
                row = self.backend.cleanup(text)
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["text"], "")
                self.assertEqual(row["rawText"], "")
                self.assertIsNone(row["error"])
                self.assertEqual(row["diagnostics"]["tokenIds"], [])
                payload = json.dumps({"A": text, "B": None}, ensure_ascii=False,
                                     separators=(",", ":"))
                self.assertEqual(row["diagnostics"]["userPayloadSha256"],
                                 hashlib.sha256(payload.encode("utf-8")).hexdigest())
                self.assertGreaterEqual(row["processingMs"], 0)
                self.assertGreaterEqual(row["wallMs"], row["processingMs"])
        self.assertIs(self.backend.load(), ready)
        self.lm.load.assert_called_once()
        self.tokenizer.apply_chat_template.assert_not_called()
        self.lm.stream_generate.assert_not_called()
        self.mx.random.seed.assert_not_called()

    def test_completed_output_is_preserved_including_empty_and_token_limit(self):
        for chunks, finish in (([""], "stop"), ([" \n", "```json\n", '{"é":1}', "\n```  "], "stop"),
                               (["\"quoted\""], "length")):
            with self.subTest(chunks=chunks, finish=finish):
                self.lm.stream_generate.side_effect = lambda *args, **kwargs: iter(
                    [response(chunk, index, finish) for index, chunk in enumerate(chunks)])
                row = self.backend.cleanup("original text must not become a fallback")
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["text"], "".join(chunks))
                self.assertEqual(row["rawText"], "".join(chunks))
                self.assertIsNone(row["error"])
                self.assertEqual(row["diagnostics"]["finishReason"], finish)
                self.assertEqual(row["diagnostics"]["hitTokenLimit"], finish == "length")
                self.assertGreaterEqual(row["processingMs"], 0)
                self.assertGreaterEqual(row["wallMs"], row["processingMs"])

    def test_partial_generation_error_preserves_raw_text_without_retry_or_fallback(self):
        def fail(*args, **kwargs):
            yield response("  partial é\n")
            raise RuntimeError("fake failure")

        self.lm.stream_generate.side_effect = fail
        with self.assertRaises(cleanup.CleanupError) as caught:
            self.backend.cleanup("base")
        row = caught.exception.result
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["text"])
        self.assertEqual(row["rawText"], "  partial é\n")
        self.assertEqual(caught.exception.rawText, row["rawText"])
        self.assertEqual(row["error"], {"type": "RuntimeError"})
        self.assertEqual(row["diagnostics"]["tokenIds"], [7])
        self.lm.stream_generate.assert_called_once()
        self.lm.stream_generate.side_effect = lambda *args, **kwargs: iter([response("next")])
        self.assertEqual(self.backend.cleanup("new take")["rawText"], "next")

    def test_no_response_for_nonempty_source_is_error(self):
        self.lm.stream_generate.side_effect = lambda *args, **kwargs: iter([])
        with self.assertRaises(cleanup.CleanupError) as caught:
            self.backend.cleanup("base")
        row = caught.exception.result
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["text"])
        self.assertEqual(row["rawText"], "")
        self.assertEqual(row["error"], {"type": "RuntimeError"})

    def test_gpu_fence_error_keeps_completed_raw_text_as_failed(self):
        self.backend.load()
        self.mx.synchronize.side_effect = [None, RuntimeError("fake fence failure"), None]
        with self.assertRaises(cleanup.CleanupError) as caught:
            self.backend.cleanup("base")
        row = caught.exception.result
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["text"])
        self.assertEqual(row["rawText"], "ok")

    def test_processing_timer_excludes_load_and_initial_fence(self):
        self.backend.load()
        with patch.object(cleanup.time, "perf_counter", side_effect=[10, 11, 13, 14]):
            row = self.backend.cleanup("base")
        self.assertEqual(row["processingMs"], 2000)
        self.assertEqual(row["wallMs"], 4000)

    def test_invalid_input_never_loads(self):
        for value in (None, 3, {"A": "text", "B": "second hypothesis"}, b"text"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.backend.cleanup(value)
        self.lm.load.assert_not_called()

    def test_bad_snapshot_path_is_rejected_before_native_load(self):
        for path in ("relative", str(self.root), str(self.root / "absent" / cleanup.REVISION)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cleanup.Backend({"modelPath": path}).load()
        self.lm.load.assert_not_called()

    def test_options_reject_other_setups_remote_paths_and_control_overrides(self):
        for options in ({}, {"modelPath": "mlx-community/model"},
                        {"modelPath": str(self.path), "setupId": "qwen/context"},
                        {"modelPath": str(self.path), "temperature": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                cleanup.Backend(options)
        backend = cleanup.Backend({"modelPath": str(self.path), "setupId": cleanup.SETUP_ID})
        self.assertEqual(backend.load()["setupId"], cleanup.SETUP_ID)

    def test_worker_frames_preserve_empty_success_and_partial_raw_error(self):
        import worker

        def fail(*args, **kwargs):
            yield response("  partial é\n")
            raise RuntimeError("fake failure")

        for output, expected_type in ((lambda *args, **kwargs: iter([response("")]), "result"),
                                      (fail, "error")):
            with self.subTest(expected_type=expected_type):
                self.lm.stream_generate.side_effect = output
                source, target = io.BytesIO(), io.BytesIO()
                base = {"sampleRate": 16000, "channels": 1, "sampleCount": 0,
                        "pcmBytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
                envelope = {"version": 1, "generation": "fake", "setupId": cleanup.SETUP_ID,
                            "pcmBytes": 0}
                worker.write_frame(source, {**envelope, "id": 1, "operation": "load", "options": {}})
                worker.write_frame(source, {**envelope, "id": 2, "operation": "cleanup",
                                           "text": "base", "B": None, "basePcm": base,
                                           "transcriptSha256": hashlib.sha256(b"base").hexdigest()})
                source.seek(0)
                with patch.object(worker.sys, "stderr", io.StringIO()):
                    code = worker.serve(source, target, cleanup.SETUP_ID, "fake",
                                        factory=lambda setup_id, options: self.backend)
                self.assertEqual(code, 0 if expected_type == "result" else 1)
                target.seek(0)
                self.assertEqual(worker.read_frame(target)[0]["type"], "ready")
                reply = worker.read_frame(target)[0]
                self.assertEqual(reply["type"], expected_type)
                if expected_type == "result":
                    self.assertEqual(reply["result"]["text"], "")
                    self.assertEqual(reply["result"]["rawText"], "")
                else:
                    self.assertEqual(reply["rawText"], "  partial é\n")
                self.assertIsNone(worker.read_frame(target))

    def test_worker_dispatches_blank_then_nonempty_with_exact_identity(self):
        import worker

        source, target = io.BytesIO(), io.BytesIO()
        base = {"sampleRate": 16000, "channels": 1, "sampleCount": 4,
                "pcmBytes": 16, "sha256": hashlib.sha256(bytes(16)).hexdigest()}
        envelope = {"version": 1, "generation": "fake", "setupId": cleanup.SETUP_ID,
                    "pcmBytes": 0}
        worker.write_frame(source, {**envelope, "id": 1, "operation": "load",
                                   "options": {"modelPath": str(self.path)}})
        cases = (("", ""), (" \t\r\n ", ""), ("\u00a0\u3000", ""), (" \tlaunch\n ", "ok"))
        for request_id, (text, _) in enumerate(cases, 2):
            worker.write_frame(source, {**envelope, "id": request_id, "operation": "cleanup",
                                       "text": text, "B": None, "basePcm": base,
                                       "transcriptSha256": hashlib.sha256(text.encode()).hexdigest()})
        worker.write_frame(source, {**envelope, "id": 6, "operation": "close"})
        source.seek(0)
        with patch.object(worker.sys, "stderr", io.StringIO()):
            code = worker.serve(source, target, cleanup.SETUP_ID, "fake")
        self.assertEqual(code, 0)
        target.seek(0)
        ready = worker.read_frame(target)[0]
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["metadata"]["model"], cleanup.MODEL)
        self.assertEqual(ready["metadata"]["revision"], cleanup.REVISION)
        self.assertEqual(ready["metadata"]["promptSha256"], cleanup.PROMPT_SHA256)
        for request_id, (text, expected) in enumerate(cases, 2):
            reply = worker.read_frame(target)[0]
            self.assertEqual({key: reply[key] for key in envelope}, envelope)
            self.assertEqual(reply["id"], request_id)
            self.assertEqual(reply["type"], "result")
            self.assertEqual(reply["result"]["text"], expected)
            self.assertEqual(reply["result"]["rawText"], expected)
            payload = json.dumps({"A": text, "B": None}, ensure_ascii=False,
                                 separators=(",", ":"))
            self.assertEqual(reply["result"]["diagnostics"]["userPayloadSha256"],
                             hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(worker.read_frame(target)[0]["type"], "closed")
        self.assertIsNone(worker.read_frame(target))
        self.lm.load.assert_called_once()
        self.lm.stream_generate.assert_called_once()
        messages = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(json.loads(messages[1]["content"]), {"A": cases[-1][0], "B": None})

    def test_prompt_drift_is_rejected_before_native_load(self):
        with (self.root / "cleanup-prompt.txt").open("ab") as prompt:
            prompt.write(b"\n")
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            self.backend.load()
        self.lm.load.assert_not_called()

    def test_each_model_file_hash_is_enforced_before_native_load(self):
        for name in self.hashes:
            path = self.path / name
            data = path.read_bytes()
            try:
                path.write_bytes(data + b"drift")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "file hash"):
                    self.backend.load()
                self.lm.load.assert_not_called()
            finally:
                path.write_bytes(data)

    def test_unpinned_extra_weight_shards_are_rejected_before_native_load(self):
        (self.path / "extra.safetensors").write_bytes(b"unapproved weights")
        with self.assertRaisesRegex(ValueError, "only the pinned"):
            self.backend.load()
        self.lm.load.assert_not_called()

    def test_native_and_loaded_template_drift_are_rejected(self):
        with patch.object(cleanup, "TEMPLATE_SHA256", "bad template hash"):
            with self.assertRaisesRegex(ValueError, "native chat template"):
                self.backend.load()
        self.lm.load.assert_not_called()
        self.tokenizer.chat_template = "runtime changed the native template"
        with self.assertRaisesRegex(ValueError, "loaded chat template"):
            self.backend.load()
        self.assertIsNone(self.backend.ready)

    def test_metal_unavailable_is_rejected_without_loading_weights(self):
        self.mx.metal.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Metal"):
            self.backend.load()
        self.lm.load.assert_not_called()

    def test_runtime_python_requirements_and_package_drift_are_rejected(self):
        with patch.object(cleanup.sys, "version_info", (3, 13, 3)):
            with self.assertRaisesRegex(ValueError, "Python"):
                self.backend.load()
        extra = SimpleNamespace(metadata={"Name": "unapproved"}, version="1")
        for distributions in (self.distributions[:-1], self.distributions + [extra]):
            with patch.object(cleanup.metadata, "distributions", return_value=distributions):
                with self.assertRaisesRegex(ValueError, "package drift"):
                    self.backend.load()
        with (self.root / "requirements-cleanup.txt").open("ab") as requirements:
            requirements.write(b"\n")
        with self.assertRaisesRegex(ValueError, "requirements hash"):
            self.backend.load()
        self.lm.load.assert_not_called()


class SourcePinsTests(unittest.TestCase):
    def test_prompt_requirements_and_model_pins_match_the_historical_contract(self):
        root = Path(cleanup.__file__).parent
        self.assertEqual(hashlib.sha256((root / "cleanup-prompt.txt").read_bytes()).hexdigest(),
                         "6feb2afb47f1bfcb5ea926ba72588f8776a12eed77e06e4fc9fdf0bc555502fd")
        self.assertEqual(hashlib.sha256((root / "requirements-cleanup.txt").read_bytes()).hexdigest(),
                         "cec49f5b582ac04d8e5963d6bc2010237bb4ece084721eb61d562b1da6c4d3fa")
        self.assertEqual(cleanup.TEMPLATE_SHA256,
                         "71672240b1a3b068067b6a332bc38d3fc05aaade9d41b4b27d7510ff4335504b")
        self.assertEqual(cleanup.MODEL, "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        self.assertEqual(cleanup.REVISION, "8b403126fc14f14cfc99bb4cfa72ecbc129ea677")


if __name__ == "__main__":
    unittest.main()
