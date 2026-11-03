#!/usr/bin/env python3
"""Build the Hex Parakeet Unified worker from pinned public source.

Fetches Hex and transcribe.cpp at fixed commits into a fresh directory under
/private/tmp, extracts Hex's own pause pipeline unchanged, builds worker.rs
against the Metal runtime with the locked dependencies, and copies the binary
to --out. Reads no audio. Child diagnostics go to log files in the build
directory, which is removed after a successful build.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
HEX = ("https://github.com/anomalyco/hex.git", "826bdbc44c8806630a45c08ed8cef35a36817a17")
RUNTIME = ("https://github.com/anomalyco/transcribe.cpp.git", "7045d1c0fe7cee37747b2792977421d435e75fca")


def run(args, root, label, env):
    with (root / (label + ".log")).open("w") as log:
        result = subprocess.run(args, stdout=log, stderr=log, env=env)
    if result.returncode:
        raise SystemExit(f"{label} failed with exit {result.returncode}; see {root / (label + '.log')}")


def fetch(root, name, url, revision, env):
    dest = root / name
    run(["git", "init", "-q", str(dest)], root, name + "-init", env)
    run(["git", "-C", str(dest), "fetch", "--depth", "1", url, revision], root, name + "-fetch", env)
    run(["git", "-C", str(dest), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", "FETCH_HEAD"], root, name + "-checkout", env)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, env=env, check=True)
    if head.stdout.strip() != revision:
        raise SystemExit(f"{name} commit mismatch")
    return dest


def build(root, out):
    env = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(root / "home"),
           "CARGO_HOME": str(root / "cargo"), "CARGO_TARGET_DIR": str(root / "target"), "CARGO_BUILD_JOBS": "4"}
    (root / "home").mkdir()
    hex_source = fetch(root, "hex", *HEX, env)
    runtime = fetch(root, "transcribe", *RUNTIME, env)
    worker = root / "worker"
    (worker / "src/transcription").mkdir(parents=True)
    shutil.copyfile(HERE / "worker.rs", worker / "src/main.rs")
    # Hex's pause logic, copied verbatim rather than reimplemented.
    streaming = (hex_source / "src/transcription/engine/streaming.rs").read_text()
    trait = streaming[streaming.index("pub(super) trait StreamFeed"):streaming.index("/// Silence a dictation stream")]
    cap = streaming[streaming.index("/// Pause audio a dictation stream"):streaming.index("#[derive(Default)]\npub(super) struct FeedStats")]
    (worker / "src/hex_pause.rs").write_text("use std::collections::VecDeque;\nuse crate::Partial;\n" + trait + cap)
    pause = (hex_source / "src/transcription/pause_segmenter.rs").read_text().split("#[cfg(test)]")[0]
    (worker / "src/transcription/pause_segmenter.rs").write_text(pause)
    wrapper = runtime / "bindings/rust/transcribe-cpp"
    # The package name keeps the research lock file valid unchanged.
    (worker / "Cargo.toml").write_text(
        '[package]\nname = "hex-grok-repro"\nversion = "0.1.0"\nedition = "2021"\n'
        '[dependencies]\nhound = "3.5"\nserde_json = "1"\n'
        f'transcribe-cpp = {{ path = "{wrapper}", features = ["metal"] }}\n')
    shutil.copyfile(HERE / "Cargo.lock", worker / "Cargo.lock")
    run(["cargo", "build", "--release", "--locked", "--manifest-path", str(worker / "Cargo.toml")], root, "build", env)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "target/release/hex-grok-repro", out)
    out.chmod(0o755)
    with out.open("rb") as stream:
        print(f"{out} sha256 {hashlib.file_digest(stream, 'sha256').hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="absolute path for the worker binary")
    args = parser.parse_args()
    if not args.out.is_absolute():
        raise SystemExit("--out must be absolute")
    root = Path(tempfile.mkdtemp(prefix="hex-unified-build-", dir="/private/tmp"))
    build(root, args.out)
    shutil.rmtree(root)
