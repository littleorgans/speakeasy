"""Bridge over the Apple Foundation Models cleanup worker (apple_cleanup/worker.swift).

Backend options contain an absolute runnerPath and an optional setupId; the
system manages the weights. Construction starts nothing; load starts the
worker and waits for its ready line, and each take is one JSON line with its
instruction, names and earlier takes. The worker sends only error kinds; this
bridge maps each to fixed text, so no take text reaches a reason.
"""

import copy
import json
from pathlib import Path
import time

from native_runner import read_reply, start, stop
from pins import file_sha256


SETUP_ID = "apple/foundation-models"
# Why the system model is unavailable at load (SystemLanguageModel.Availability).
UNAVAILABLE = {
    "appleIntelligenceNotEnabled": "Apple Intelligence is off",
    "modelNotReady": "model not ready, it may still be downloading",
    "deviceNotEligible": "this Mac is not eligible",
}
# Per take failures (LanguageModelError and the worker's own), as "<code>: <fixed text>".
FAILURES = {
    "contextSizeExceeded": "limit: the take is too long for the model's context",
    "unsupportedLanguageOrLocale": "limit: the model does not support this language",
    "guardrailViolation": "execution: the model's guardrails refused the take",
    "refusal": "execution: the model refused the take",
    "rateLimited": "execution: the system model is rate limited",
}
SETTINGS = {"session": "fresh per take", "sampling": "greedy", "guardrails": "permissiveContentTransformations",
            "useCase": "general", "prewarm": True}


class Backend:
    takes_context = True

    def __init__(self, options):
        if not isinstance(options, dict) or set(options) not in ({"runnerPath"}, {"runnerPath", "setupId"}):
            raise ValueError("Apple cleanup options require runnerPath and optional setupId only")
        if options.get("setupId", SETUP_ID) != SETUP_ID:
            raise ValueError("Unsupported Apple cleanup setupId")
        runner = options["runnerPath"]
        if not isinstance(runner, str) or "\0" in runner or not Path(runner).is_absolute():
            raise ValueError("Apple cleanup runnerPath must be an absolute path")
        self.runner_path = Path(runner)
        self.process = None
        self.metadata = None

    def load(self):
        if self.metadata is not None:
            return copy.deepcopy(self.metadata)
        try:
            started = time.perf_counter()
            self.process = start([self.runner_path])
            ready = read_reply(self.process, "Apple cleanup")
            if ready.get("kind") == "error" and ready.get("error") == "unavailable":
                reason = UNAVAILABLE.get(ready.get("reason"), "Apple Foundation Models is unavailable")
                raise RuntimeError(f"unavailable: {reason}")
            if (ready.get("kind") != "ready" or ready.get("backend") != "apple"
                    or ready.get("variant") != "foundation-models"):
                raise RuntimeError("load: Apple cleanup worker identity mismatch")
            self.metadata = {
                "setupId": SETUP_ID, "loadMs": (time.perf_counter() - started) * 1000,
                "runtime": "Apple Foundation Models, system managed weights",
                "runnerSha256": file_sha256(self.runner_path), "workerLoadMs": ready.get("loadMs"),
                "contextSize": ready.get("contextSize"), "settings": SETTINGS,
                "loadTiming": "worker start to ready line", "processingTiming": "worker clock around respond",
            }
            return copy.deepcopy(self.metadata)
        except BaseException:
            self.close()
            raise

    def cleanup(self, text, instruction="", names=(), earlier=()):
        if self.process is None or self.metadata is None:
            raise RuntimeError("Apple cleanup backend is not loaded")
        if not text.strip():
            return {"text": "", "processingMs": 0.0}
        request = {"text": text, "instruction": instruction, "names": list(names), "earlier": list(earlier)}
        self.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.process.stdin.flush()
        reply = read_reply(self.process, "Apple cleanup")
        if reply.get("kind") == "error":
            raise RuntimeError(FAILURES.get(reply.get("error"), "execution: the model failed"))
        if reply.get("kind") != "result" or not isinstance(reply.get("text"), str):
            raise RuntimeError("execution: Apple cleanup worker returned no result")
        return {"text": reply["text"].strip(), "rawText": reply["text"], "processingMs": reply.get("processingMs")}

    def close(self):
        process, self.process = self.process, None
        self.metadata = None
        if process is not None:
            stop(process)

