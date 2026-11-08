"""Standard library fakes only; no MLX runtime or model inference."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cleanup
import s1_mini
import worker


def response(text, finish="stop"):
    return SimpleNamespace(text=text, finish_reason=finish, generation_tokens=3)


class S1MiniTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="speakeasy-s1-mini-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / s1_mini.REVISION
        self.path.mkdir()
        self.template = "fake qwen3 template"
        files = {name: b"fake " + name.encode() for name in s1_mini.MODEL_FILES}
        files["chat_template.jinja"] = self.template.encode()
        for name, data in files.items():
            (self.path / name).write_bytes(data)
        self.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

        self.mx = ModuleType("mlx.core")
        self.mx.gpu = "fake gpu"
        self.mx.metal = SimpleNamespace(is_available=Mock(return_value=True))
        self.mx.set_default_device = Mock()
        self.mx.default_stream = Mock(return_value="fake gpu stream")
        self.mx.stream = Mock(side_effect=lambda stream: nullcontext())
        self.mx.eval = Mock()
        self.mx.synchronize = Mock()
        self.mx.get_active_memory = Mock(return_value=123)
        self.model = SimpleNamespace(parameters=Mock(return_value=["fake parameters"]))
        self.tokenizer = SimpleNamespace(chat_template=self.template,
                                         encode=Mock(side_effect=lambda text: text.split()),
                                         apply_chat_template=Mock(return_value=[1, 2, 3, 4]))
        self.lm = ModuleType("mlx_lm")
        self.lm.load = Mock(return_value=(self.model, self.tokenizer))
        self.lm.stream_generate = Mock(side_effect=lambda *args, **kwargs: iter([response(" So"), response(" ship it. ")]))
        self.sample = ModuleType("mlx_lm.sample_utils")
        self.sample.make_sampler = Mock(return_value="greedy sampler")
        mlx = ModuleType("mlx")
        mlx.core = self.mx
        for item in (
            patch.object(s1_mini, "MODEL_FILES", self.hashes),
            patch.object(s1_mini, "runtime_versions", return_value={"mlx-lm": "0.32.2"}),
            patch.dict(sys.modules, {"mlx": mlx, "mlx.core": self.mx, "mlx_lm": self.lm,
                                     "mlx_lm.sample_utils": self.sample}),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.backend = s1_mini.Backend({"modelPath": str(self.path), "setupId": s1_mini.SETUP_ID})

    def test_worker_routes_the_model_to_this_backend_with_the_cleanup_environment(self):
        self.assertEqual(worker.BACKENDS[s1_mini.SETUP_ID], "s1_mini")
        self.assertIn(s1_mini.SETUP_ID, worker.TEXT_CLEANUPS)
        self.assertFalse(getattr(self.backend, "takes_context", False), "S1-mini takes text only")

    def test_load_attests_the_snapshot_then_loads_once_without_generation(self):
        ready = self.backend.load()
        self.assertIs(self.backend.load(), ready)
        self.lm.load.assert_called_once_with(str(self.path), trust_remote_code=False)
        self.lm.stream_generate.assert_not_called()
        self.assertEqual(ready["setupId"], s1_mini.SETUP_ID)
        self.assertEqual(ready["name"], "S1-mini by Superwhisper")
        self.assertEqual(ready["revision"], s1_mini.REVISION)
        self.assertEqual(ready["fileSha256"], self.hashes)

    def test_exact_card_prompt_control_line_thinking_off_and_token_ceiling(self):
        result = self.backend.cleanup("um so ship it")
        self.assertEqual(result["text"], "So ship it.")
        messages = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages, [
            {"role": "system", "content": "You are a text normalizer for speech-to-text transcripts. The input begins "
             "with a control line specifying the styling, structure, and context settings; clean the transcript to "
             "match those settings and output only the cleaned text."},
            {"role": "user", "content": "[Styling: semi-formal] [Structure: prose] [Context: general]\num so ship it"},
        ])
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.kwargs,
                         {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
        generated = self.lm.stream_generate.call_args
        self.assertEqual(generated.kwargs["max_tokens"], int(1.3 * 4) + 32)
        self.assertEqual(generated.kwargs["sampler"], "greedy sampler")
        self.sample.make_sampler.assert_called_once_with(temp=0)
        self.assertIsNone(generated.kwargs["prompt_cache"])
        self.assertEqual(result["diagnostics"]["inputTokens"], 4)

    def test_blank_input_is_empty_without_generating_and_long_input_is_a_limit(self):
        self.assertEqual(self.backend.cleanup(" \n ")["text"], "")
        self.tokenizer.apply_chat_template.assert_not_called()
        self.lm.stream_generate.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, r"^limit: S1-mini takes at most 1000 tokens$"):
            self.backend.cleanup("word " * 1001)
        self.lm.stream_generate.assert_not_called()

    def test_wrong_revision_hash_or_missing_notice_is_refused_before_native_load(self):
        other = Path(self.temp.name) / "main"
        other.mkdir()
        with self.assertRaisesRegex(RuntimeError, "^load: S1-mini requires the pinned local snapshot directory$"):
            s1_mini.Backend({"modelPath": str(other)}).load()
        (self.path / "model.safetensors").write_bytes(b"other weights")
        with self.assertRaisesRegex(RuntimeError, "^load: S1-mini model file hash mismatch: model.safetensors$"):
            self.backend.load()
        (self.path / "model.safetensors").write_bytes(b"fake model.safetensors")
        (self.path / "NOTICE").unlink()
        with self.assertRaisesRegex(RuntimeError, "^load: S1-mini model file missing: NOTICE$"):
            self.backend.load()
        self.lm.load.assert_not_called()

    def test_loaded_template_must_be_the_pinned_jinja(self):
        self.tokenizer.chat_template = "runtime changed the template"
        with self.assertRaisesRegex(RuntimeError, "chat template"):
            self.backend.load()
        self.assertIsNone(self.backend.ready)

    def test_options_take_only_an_absolute_model_path_for_this_model(self):
        for options in ({"modelPath": "relative"}, {"modelPath": "/m", "setupId": "mlx-cleanup/qwen-targeted"},
                        {"modelPath": "/m", "runnerPath": "/r"}, None):
            with self.subTest(options=options), self.assertRaises(ValueError):
                s1_mini.Backend(options)



class SourcePinsTests(unittest.TestCase):
    def test_pins_match_the_card_snapshot_and_reuse_the_cleanup_environment(self):
        self.assertIs(s1_mini.runtime_versions, cleanup.runtime_versions)
        self.assertEqual(s1_mini.REVISION, "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4")
        self.assertEqual(s1_mini.MODEL_FILES["model.safetensors"],
                         "69d2057077ab4dc738aaaab75d2a8ffa141e3a09fb9d956198cfce46f381131a")
        self.assertTrue({"LICENSE", "NOTICE", "chat_template.jinja"} <= set(s1_mini.MODEL_FILES))
        self.assertEqual(len(s1_mini.MODEL_FILES), 10)


if __name__ == "__main__":
    unittest.main()
