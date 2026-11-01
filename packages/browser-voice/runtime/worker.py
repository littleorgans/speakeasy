"""Persistent, bounded local backend transport. No model imports at startup."""

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import struct
import sys
from array import array

VERSION = 1
MAX_HEADER_BYTES = 65_536
MAX_PCM_BYTES = 7_680_000
MAX_TEXT_BYTES = 8_192
BACKENDS = {
    "mlx-parakeet/v3-greedy": "parakeet",
    "mlx-parakeet/ultra-greedy": "parakeet",
    "mlx-cleanup/qwen-targeted": "cleanup",
    "qwen/context": "qwen_asr",
}


class ProtocolError(ValueError):
    pass


def integer(value, minimum=0, maximum=MAX_PCM_BYTES):
    return type(value) is int and minimum <= value <= maximum


def read_exact(stream, size, allow_eof=False):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks and allow_eof:
                return None
            raise ProtocolError("truncated frame")
        chunks.extend(chunk)
    return bytes(chunks)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise ProtocolError("nonfinite JSON number")


def read_frame(stream):
    prefix = read_exact(stream, 4, allow_eof=True)
    if prefix is None:
        return None
    size = struct.unpack("<I", prefix)[0]
    if not 0 < size <= MAX_HEADER_BYTES:
        raise ProtocolError("invalid header length")
    header = json.loads(read_exact(stream, size).decode("utf-8"),
                        object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if not isinstance(header, dict):
        raise ProtocolError("header must be an object")
    length = header.get("pcmBytes")
    if not integer(length) or length % 4:
        raise ProtocolError("invalid PCM byte count")
    return header, read_exact(stream, length)


def write_frame(stream, header):
    body = json.dumps(header, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if not 0 < len(body) <= MAX_HEADER_BYTES:
        raise ProtocolError("reply exceeds header limit")
    stream.write(struct.pack("<I", len(body)) + body)
    stream.flush()


def envelope(header, setup_id, generation):
    if (type(header.get("version")) is not int or header["version"] != VERSION
            or header.get("setupId") != setup_id
            or header.get("generation") != generation
            or not integer(header.get("id"), 1, 2**53 - 1)):
        raise ProtocolError("request envelope mismatch")
    return {key: header[key] for key in ("version", "generation", "id", "setupId", "operation")}


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ProtocolError("missing PCM identity")
    count = identity.get("sampleCount")
    if (not integer(count, 0, MAX_PCM_BYTES // 4)
            or type(identity.get("sampleRate")) is not int or identity["sampleRate"] != 16_000
            or type(identity.get("channels")) is not int or identity["channels"] != 1
            or type(identity.get("pcmBytes")) is not int or identity["pcmBytes"] != count * 4):
        raise ProtocolError("invalid PCM identity")
    digest = identity.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ProtocolError("invalid PCM hash")


def pcm_samples(header, body):
    validate_identity(header)
    if hashlib.sha256(body).hexdigest() != header["sha256"]:
        raise ProtocolError("PCM hash mismatch")
    if sys.byteorder == "little":
        samples = memoryview(body).cast("f")
    else:
        swapped = array("f")
        swapped.frombytes(body)
        swapped.byteswap()
        samples = memoryview(swapped)
    if any(not math.isfinite(value) for value in samples):
        raise ProtocolError("PCM must be finite")
    return samples


def cleanup_text(header):
    base = header.get("basePcm")
    validate_identity(base)
    if set(base) != {"sampleRate", "channels", "sampleCount", "pcmBytes", "sha256"}:
        raise ProtocolError("cleanup base must contain only PCM identity")
    text = header.get("text")
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ProtocolError("invalid cleanup text")
    if header.get("B", "missing") is not None:
        raise ProtocolError("cleanup requires B null")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != header.get("transcriptSha256"):
        raise ProtocolError("cleanup transcript hash mismatch")
    return text


def backend_factory(setup_id, options):
    module = importlib.import_module(BACKENDS[setup_id])
    return module.Backend({**options, "setupId": setup_id})


def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ProtocolError("backend result requires text")
    elapsed = result.get("processingMs")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ProtocolError("backend result requires finite processingMs")
    if "rawText" in result and not isinstance(result["rawText"], str):
        raise ProtocolError("backend rawText must be text")
    decode_ms = result.get("decodeMs")
    if decode_ms is not None and (type(decode_ms) not in (int, float)
                                  or not math.isfinite(decode_ms) or decode_ms < 0):
        raise ProtocolError("backend decodeMs must be finite")


def handle_request(header, body, backend, setup_id, factory):
    operation = header.get("operation")
    cleanup = setup_id == "mlx-cleanup/qwen-targeted"
    if operation != "transcribe" and body:
        raise ProtocolError("unexpected body")
    if operation == "load":
        if backend is not None or not isinstance(header.get("options"), dict):
            raise ProtocolError("invalid load request")
        backend = factory(setup_id, header["options"])
        metadata = backend.load()
        if not isinstance(metadata, dict):
            raise ProtocolError("load metadata must be an object")
        return backend, {"type": "ready", "metadata": metadata}
    if operation == "close":
        return backend, {"type": "closed"}
    if backend is None:
        raise ProtocolError("backend is not ready")
    if operation == "transcribe" and not cleanup:
        result = backend.transcribe(pcm_samples(header, body))
    elif operation == "cleanup" and cleanup:
        result = backend.cleanup(cleanup_text(header))
    else:
        raise ProtocolError("operation does not match setup")
    validate_result(result)
    return backend, {"type": "result", "result": result}


def serve(source, target, setup_id, generation, factory=backend_factory):
    backend = None
    last_id = 0

    def create_backend(selected, options):
        nonlocal backend
        # Retain ownership even if load raises before handle_request returns.
        backend = factory(selected, options)
        return backend

    def close_backend():
        nonlocal backend
        owned, backend = backend, None
        close = getattr(owned, "close", None)
        if close is not None:
            with contextlib.redirect_stdout(sys.stderr):
                close()

    try:
        while True:
            reply = None
            try:
                frame = read_frame(source)
                if frame is None:
                    return 0
                header, body = frame
                reply = envelope(header, setup_id, generation)
                if header["id"] <= last_id:
                    raise ProtocolError("request id must increase")
                last_id = header["id"]
                # This also captures Python library prints during lazy imports.
                with contextlib.redirect_stdout(sys.stderr):
                    backend, outcome = handle_request(header, body, backend, setup_id, create_backend)
                if outcome["type"] == "closed":
                    close_backend()
                write_frame(target, {**reply, **outcome, "pcmBytes": 0})
                if outcome["type"] == "closed":
                    return 0
            except Exception as error:
                # The parent may force retirement as soon as it reads an error.
                close_backend()
                if reply is not None:
                    failure = {**reply, "type": "error", "pcmBytes": 0,
                               "error": str(error)[:2048] or type(error).__name__}
                    partial = getattr(error, "rawText", None)
                    if isinstance(partial, str) and len(partial.encode("utf-8")) <= MAX_TEXT_BYTES:
                        failure["rawText"] = partial
                    try:
                        write_frame(target, failure)
                    except Exception:
                        pass
                print("worker failed: " + type(error).__name__, file=sys.stderr)
                return 1
    finally:
        close_backend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=BACKENDS, required=True)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    # Keep a private protocol descriptor. Native writes to fd 1 go to stderr too.
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with os.fdopen(protocol_fd, "wb", buffering=0) as target:
        return serve(sys.stdin.buffer, target, args.setup, args.generation, factory=backend_factory)


if __name__ == "__main__":
    sys.exit(main())
