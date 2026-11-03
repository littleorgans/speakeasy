#!/usr/bin/env python3
"""Build the Piper split decoder worker from pinned public source.

Fetches piper1-gpl at a fixed commit into a fresh temporary directory, builds
libpiper (its CMake build fetches onnxruntime 1.22.0 and builds espeak-ng
statically), installs it into --out and compiles worker.c beside it, so the
worker finds its libraries relative to itself. Child diagnostics go to log
files in the build directory, which is removed after a successful build.

    python3 runtime/piper_split/build.py --out ~/Library/Caches/speakeasy/piper-split/10880972
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PIPER = ("https://github.com/OHF-Voice/piper1-gpl.git", "10880972db9e0aca101fed350eadf8f49deefe9f")
# espeak-ng's data compiler truncates paths near 180 characters, and the build
# adds up to 92 below the root; a longer root fails with a missing phsource file.
MAX_ROOT_CHARS = 80


def run(args, root, label, env):
    with (root / (label + ".log")).open("w") as log:
        # libpiper's CMake downloads onnxruntime into ./download: keep it inside the build root.
        result = subprocess.run(args, stdout=log, stderr=log, env=env, cwd=root)
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
    env = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(root / "home")}
    (root / "home").mkdir()
    source = fetch(root, "piper1-gpl", *PIPER, env)
    staged = root / "install"
    run(["cmake", "-S", str(source / "libpiper"), "-B", str(root / "b"), "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_INSTALL_PREFIX={staged}"], root, "configure", env)
    run(["cmake", "--build", str(root / "b"), "--config", "Release", "-j", "6"], root, "build", env)
    run(["cmake", "--install", str(root / "b"), "--config", "Release"], root, "install", env)
    (staged / "bin").mkdir(exist_ok=True)
    run(["cc", "-O2", "-Wall", "-Wextra", "-std=c11", str(HERE / "worker.c"), f"-I{staged / 'include'}",
         f"-L{staged / 'lib'}", "-lpiper", "-Wl,-rpath,@executable_path/../lib",
         "-o", str(staged / "bin" / "piper-split-worker")], root, "worker", env)
    shutil.copytree(staged, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ps-"))
    if len(str(root)) > MAX_ROOT_CHARS:
        shutil.rmtree(root)
        raise SystemExit(f"build root {root} is too long for espeak-ng; set TMPDIR to a shorter directory")
    build(root, out)
    shutil.rmtree(root)
    print(out / "bin" / "piper-split-worker")


if __name__ == "__main__":
    main()
