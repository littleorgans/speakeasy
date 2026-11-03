"""Pin attestation shared by the voice backends: snapshot bytes and runtime versions."""

import hashlib
import importlib.metadata
import platform


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
