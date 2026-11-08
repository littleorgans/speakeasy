"""S1-mini by Superwhisper: text cleanup of a final on mlx-lm, with its card's exact prompt.

Runs in the cleanup environment (requirements-cleanup.txt, runtime_versions)
on the first party BF16 snapshot at REVISION, which keeps LICENSE and NOTICE.
Native imports occur only in load(). The card fixes the system prompt and the
control line; the model takes no instruction, names or earlier takes.
"""

import hashlib
from pathlib import Path
import time

from cleanup import load_mlx, pinned_snapshot, PYTHON, REQUIREMENTS_SHA256, runtime_versions


SETUP_ID = "superwhisper/s1-mini"
MODEL = "superwhisper/s1-mini"
REVISION = "88f6b15896c73bbb13a3b596e0afe8ea0d5150b4"
# The card's exact system prompt and its suggested control line.
SYSTEM = ("You are a text normalizer for speech-to-text transcripts. The input begins with a control line "
          "specifying the styling, structure, and context settings; clean the transcript to match those "
          "settings and output only the cleaned text.")
CONTROL = "[Styling: semi-formal] [Structure: prose] [Context: general]"
# The card's recommended input bound; longer takes are refused, not chunked.
MAX_INPUT_TOKENS = 1_000
MODEL_FILES = {
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "6ca52b0bcb818c9e52db8ea18413494110c5f2def581b769f621dda951a7863b",
    "LICENSE": "f715982df6ce767ae64864d74b95644e5d658aab54717dbceeb252eb3cbcb421",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "69d2057077ab4dc738aaaab75d2a8ffa141e3a09fb9d956198cfce46f381131a",
    "NOTICE": "4feae786f1766dc58e807bcae1e7bdd06bf3610a03a23d0621b6c6c4d05f2980",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SETTINGS = {"temperature": 0, "sampler": "greedy argmax", "enableThinking": False,
            "maxTokens": "int(1.3 * input tokens) + 32", "maxInputTokens": MAX_INPUT_TOKENS,
            "promptCache": "fresh per request", "control": CONTROL}


class Backend:
    """options.modelPath is the absolute snapshot directory at REVISION.

    loadMs includes the pin checks, native imports, weight loading and GPU
    synchronization. processingMs includes tokenization, prefill, decoding and
    the final GPU fence.
    """

    def __init__(self, options):
        if not isinstance(options, dict) or set(options) not in ({"modelPath"}, {"modelPath", "setupId"}):
            raise ValueError("S1-mini options require modelPath and optional setupId only")
        if options.get("setupId", SETUP_ID) != SETUP_ID:
            raise ValueError("Unsupported S1-mini setupId")
        if not isinstance(options["modelPath"], str) or not Path(options["modelPath"]).is_absolute():
            raise ValueError("S1-mini modelPath must be absolute")
        self.path = Path(options["modelPath"])
        self.ready = None

    def load(self):
        if self.ready is not None:
            return self.ready
        started = time.perf_counter()
        try:
            hashes = pinned_snapshot(self.path, REVISION, MODEL_FILES, "S1-mini")
        except ValueError as error:
            # Fixed text from the pin check: the page shows it as the skip reason.
            raise RuntimeError(f"load: {error}") from None
        versions = runtime_versions()
        template = (self.path / "chat_template.jinja").read_text("utf-8")
        mx, stream, model, tokenizer, sampler, generate = load_mlx(self.path)
        if tokenizer.chat_template != template:
            raise RuntimeError("load: S1-mini loaded chat template differs from chat_template.jinja")
        self.mx, self.stream, self.model, self.tokenizer = mx, stream, model, tokenizer
        self.sampler, self.generate = sampler, generate
        self.ready = {
            "setupId": SETUP_ID, "model": MODEL, "name": "S1-mini by Superwhisper", "revision": REVISION,
            "fileSha256": hashes, "requirementsSha256": REQUIREMENTS_SHA256,
            "systemSha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
            "python": ".".join(map(str, PYTHON)), "packages": versions, "settings": SETTINGS,
            "gpuActiveBytes": mx.get_active_memory(), "loadMs": (time.perf_counter() - started) * 1000,
        }
        return self.ready

    def cleanup(self, text):
        self.load()
        started = time.perf_counter()
        # Blank input has nothing to clean; filler only input may come back empty too.
        if not text.strip():
            return {"text": "", "processingMs": 0.0}
        input_tokens = len(self.tokenizer.encode(text))
        if input_tokens > MAX_INPUT_TOKENS:
            raise RuntimeError(f"limit: S1-mini takes at most {MAX_INPUT_TOKENS} tokens")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CONTROL + "\n" + text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        raw, final = "", None
        for final in self.generate(self.model, self.tokenizer, prompt=prompt,
                                   max_tokens=int(1.3 * input_tokens) + 32, sampler=self.sampler,
                                   stream=self.stream, prompt_cache=None):
            raw += final.text
        self.mx.synchronize()
        if final is None:
            raise RuntimeError("execution: S1-mini generated nothing")
        return {"text": raw.strip(), "rawText": raw, "processingMs": (time.perf_counter() - started) * 1000,
                "diagnostics": {"inputTokens": input_tokens, "promptTokens": len(prompt),
                                "generationTokens": final.generation_tokens, "finishReason": final.finish_reason}}
