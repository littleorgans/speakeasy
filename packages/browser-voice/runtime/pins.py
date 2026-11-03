"""Pin attestation shared by the voice backends: snapshot bytes and runtime versions."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform

# Full file hashes by resolved path, size and modification time: one small
# file per key, so concurrent workers never share one, and a file whose size
# or time changed misses and is hashed in full again.
DIGEST_CACHE = Path.home() / "Library" / "Caches" / "speakeasy" / "sha256"


def pinned_files(value, label):
    """The host registry's pins for a voice: a list of name, bytes and sha256, as a name to (size, digest) map."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} options require the pinned files")
    files = {}
    for entry in value:
        if (not isinstance(entry, dict) or set(entry) != {"name", "bytes", "sha256"}
                or not isinstance(entry["name"], str) or not entry["name"] or ".." in entry["name"].split("/")
                or entry["name"].startswith("/") or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64):
            raise ValueError(f"{label} pinned file entries need a relative name, bytes and sha256")
        files[entry["name"]] = (entry["bytes"], entry["sha256"])
    return files


def file_sha256(path):
    """A file's SHA256, hashed in full unless this path, size and modification time already were."""
    path = Path(path).resolve()
    before = path.stat()
    key = hashlib.sha256(f"{path}\0{before.st_size}\0{before.st_mtime_ns}".encode()).hexdigest()
    entry = DIGEST_CACHE / key
    try:
        cached = entry.read_text()
    except OSError:
        cached = ""
    if len(cached) == 64 and all(char in "0123456789abcdef" for char in cached):
        return cached
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    # A file that changed while it was read is never cached.
    if (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns):
        try:
            DIGEST_CACHE.mkdir(parents=True, exist_ok=True)
            pending = DIGEST_CACHE / f"{key}.{os.getpid()}"
            pending.write_text(digest)
            os.replace(pending, entry)
        except OSError:
            pass
    return digest


def attest_snapshot(model_path, files, label):
    """Checks every pinned file's size and SHA256 before any runtime sees the path."""
    if not model_path.is_dir():
        raise ValueError(f"{label} modelPath must be an existing local snapshot directory")
    for name, (size, expected) in files.items():
        path = model_path / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"{label} snapshot size mismatch: {name}")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"{label} snapshot SHA256 mismatch: {name}")


def runtime_identity(python_version, versions, requirements, label):
    """The interpreter and package versions, refused unless they match the pinned environment."""
    if platform.python_version() != python_version:
        raise ValueError(f"{label} requires Python {python_version}")
    installed = {name: importlib.metadata.version(name) for name in versions}
    if installed != versions:
        raise ValueError(f"{label} runtime version mismatch; use {requirements}")
    return {"python": python_version, **installed}
