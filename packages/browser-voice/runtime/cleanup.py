"""Pinned Qwen cleanup of exact Ultra text. Native imports occur only in load()."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
SETUP_ID = "mlx-cleanup/qwen-targeted"
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
PROMPT_SHA256 = "6feb2afb47f1bfcb5ea926ba72588f8776a12eed77e06e4fc9fdf0bc555502fd"
TEMPLATE_SHA256 = "71672240b1a3b068067b6a332bc38d3fc05aaade9d41b4b27d7510ff4335504b"
REQUIREMENTS_SHA256 = "cec49f5b582ac04d8e5963d6bc2010237bb4ece084721eb61d562b1da6c4d3fa"
PYTHON = (3, 13, 2)
MODEL_FILES = {
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "config.json": "636d3e2a15e8914b8cf82b05cc2288a811f9bd93c3bf1afc00cab701a70b47c0",
    "tokenizer_config.json": "f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7",
    "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
    "model.safetensors": "0979f33d1bc58afcf696d13f57977644e7b11a6f0eec3e631d8e9463d18c0717",
    "model.safetensors.index.json": "6b98634d5044f0e2ad45228a374f8445904e571f1082392d08a6ce54f5d517ca",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SETTINGS = {
    "temperature": 0, "sampler": "greedy argmax", "seed": 0,
    "maxTokens": 512, "promptCache": "fresh per request",
    "kvQuantization": None, "logitsProcessors": None, "draftModel": None,
    "userSchema": {"A": "string", "B": "null"},
    "userEncoding": "UTF8 JSON, ensure_ascii=False, compact separators, A then B",
    "chatTemplate": "native tokenizer, system then user, add_generation_prompt=True",
    "chatTemplateTokenize": True, "trustRemoteCode": False,
}


def template_sha256(template):
    encoded = json.dumps(template, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_versions():
    if sys.version_info[:3] != PYTHON:
        raise ValueError("Cleanup requires Python 3.13.2")
    requirements = (ROOT / "requirements-cleanup.txt").read_bytes()
    if hashlib.sha256(requirements).hexdigest() != REQUIREMENTS_SHA256:
        raise ValueError("Cleanup requirements hash mismatch")
    expected = dict(line.split("==") for line in requirements.decode().splitlines())
    normalize = lambda name: name.lower().replace("_", "-")
    expected = {normalize(name): version for name, version in expected.items()}
    actual = {normalize(dist.metadata["Name"]): dist.version
              for dist in metadata.distributions()}
    if actual != expected:
        raise ValueError("Cleanup runtime package drift")
    return actual


class CleanupError(RuntimeError):
    """Keep the failed historical result and expose rawText to the transport."""

    def __init__(self, result):
        super().__init__("Cleanup generation failed: " + result["error"]["type"])
        self.rawText = result["rawText"]
        self.result = result


class Backend:
    """options.modelPath is an existing absolute snapshot directory at REVISION.

    loadMs includes local identity checks, native imports, weight loading and GPU
    synchronization, excluding process startup. cleanup processingMs excludes
    loading and the initial GPU fence, and includes template/tokenization,
    prefill, decoding and the final GPU fences. wallMs includes the initial fence.
    The host serializes calls and owns process cancellation and idle release.
    """

    def __init__(self, options):
        if (not isinstance(options, dict) or set(options) not in
                ({"modelPath"}, {"modelPath", "setupId"})):
            raise ValueError("Cleanup options require modelPath and optional setupId only")
        if options.get("setupId", SETUP_ID) != SETUP_ID:
            raise ValueError("Unsupported cleanup setupId")
        if not isinstance(options["modelPath"], str) or not Path(options["modelPath"]).is_absolute():
            raise ValueError("Cleanup modelPath must be absolute")
        self.options = dict(options)
        self.ready = None

    def load(self):
        if self.ready is not None:
            return self.ready
        started = time.perf_counter()
        path = Path(self.options["modelPath"])
        if not path.is_absolute() or path.resolve().name != REVISION or not path.is_dir():
            raise ValueError("Cleanup requires the pinned local snapshot directory")
        if {file.name for file in path.glob("*.safetensors")} != {"model.safetensors"}:
            raise ValueError("Cleanup snapshot must contain only the pinned model.safetensors weights")
        prompt = (ROOT / "cleanup-prompt.txt").read_bytes()
        if hashlib.sha256(prompt).hexdigest() != PROMPT_SHA256:
            raise ValueError("Cleanup prompt hash mismatch")
        versions = runtime_versions()
        hashes = {}
        for name, expected in MODEL_FILES.items():
            with (path / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Cleanup model file hash mismatch: " + name)
            hashes[name] = actual
        config = json.loads((path / "config.json").read_bytes())
        template = json.loads((path / "tokenizer_config.json").read_bytes())["chat_template"]
        if template_sha256(template) != TEMPLATE_SHA256:
            raise ValueError("Cleanup native chat template hash mismatch")

        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU is unavailable")
        mx.set_default_device(mx.gpu)
        stream = mx.default_stream(mx.gpu)
        with mx.stream(stream):
            model, tokenizer = load(str(path), trust_remote_code=False)
            mx.eval(model.parameters())
        mx.synchronize()
        if template_sha256(tokenizer.chat_template) != TEMPLATE_SHA256:
            raise ValueError("Cleanup loaded chat template hash mismatch")
        sampler = make_sampler(temp=0)
        self.mx, self.stream = mx, stream
        self.model, self.tokenizer = model, tokenizer
        self.sampler, self.generate = sampler, stream_generate
        self.system = prompt.decode("utf-8")
        self.ready = {
            "setupId": SETUP_ID, "model": MODEL, "revision": REVISION,
            "quantization": config.get("quantization"),
            "modelType": config.get("model_type"),
            "promptSha256": PROMPT_SHA256, "chatTemplateSha256": TEMPLATE_SHA256,
            "fileSha256": hashes, "requirementsSha256": REQUIREMENTS_SHA256,
            "python": ".".join(map(str, PYTHON)), "packages": versions,
            "settings": SETTINGS, "defaultDevice": str(mx.default_device()),
            "generationStream": str(stream), "metalAvailable": True,
            "deviceInfo": mx.device_info(), "gpuActiveBytes": mx.get_active_memory(),
            "loadMs": (time.perf_counter() - started) * 1000,
        }
        return self.ready

    def cleanup(self, text):
        if not isinstance(text, str):
            raise TypeError("Cleanup input must be exact Ultra text, including empty")
        self.load()
        mx = self.mx
        wall_start = time.perf_counter()
        started = wall_start
        raw, tokens, final, failure = "", [], None, None
        diagnostics = {"promptSha256": PROMPT_SHA256}
        try:
            mx.synchronize()
            started = time.perf_counter()
            payload = json.dumps({"A": text, "B": None}, ensure_ascii=False, separators=(",", ":"))
            diagnostics["userPayloadSha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            mx.random.seed(0)
            messages = [{"role": "system", "content": self.system},
                        {"role": "user", "content": payload}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            diagnostics["promptTokens"] = len(prompt)
            diagnostics["promptTokenIdsSha256"] = hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
            for response in self.generate(
                self.model, self.tokenizer, prompt=prompt, max_tokens=512,
                sampler=self.sampler, stream=self.stream, prompt_cache=None,
                logits_processors=None,
            ):
                raw += response.text
                tokens.append(response.token)
                final = response
            mx.synchronize()
            if final is None:
                raise RuntimeError("Generator yielded no finish response")
            diagnostics.update({
                "finishReason": final.finish_reason,
                "hitTokenLimit": final.finish_reason == "length",
                "generationTokens": final.generation_tokens,
                "enginePromptTokensPerSecond": final.prompt_tps,
                "engineGenerationTokensPerSecond": final.generation_tps,
            })
        except Exception as error:
            failure = {"type": type(error).__name__}
        finally:
            try:
                mx.synchronize()
            except Exception as error:
                if failure is None:
                    failure = {"type": type(error).__name__}
        processing_ms = (time.perf_counter() - started) * 1000
        diagnostics["tokenIds"] = tokens
        result = {
            "status": "ok" if failure is None else "error",
            "text": raw if failure is None else None, "rawText": raw,
            "processingMs": processing_ms,
            "wallMs": (time.perf_counter() - wall_start) * 1000,
            "error": failure, "diagnostics": diagnostics,
        }
        if failure is not None:
            raise CleanupError(result)
        return result
