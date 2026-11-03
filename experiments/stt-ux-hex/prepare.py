#!/usr/bin/env python3
"""Fetch public code only and build in a fresh system temp directory.

No audio or transcript is read by this step. All child diagnostics go to files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
HEX_REV = "826bdbc44c8806630a45c08ed8cef35a36817a17"
RUNTIME_REV = "7045d1c0fe7cee37747b2792977421d435e75fca"
INSTALLED_SOURCE_REV = "f333d28cbfdf9ea177bf68e6c0753271077ab3e5"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, root, label, env):
    with (root / (label + ".log")).open("w") as log:
        result = subprocess.run(args, stdout=log, stderr=log, env=env)
    print(f"{label}: exit {result.returncode}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def prepare(root):
    marker = root / ".hex-grok-repro"
    if not marker.exists():
        if any(root.iterdir()):
            raise SystemExit("refusing unmarked nonempty directory")
        marker.write_text("isolated accuracy build\n")
    for directory in ["home", "cargo"]:
        (root / directory).mkdir(exist_ok=True)
    env = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
           "HOME": str(root / "home"), "CARGO_HOME": str(root / "cargo"),
           "CARGO_TARGET_DIR": str(root / "target"), "CARGO_BUILD_JOBS": "4"}
    # Downloads are git fetches and registry dependency fetches. Their inputs
    # contain public source identifiers only. Inference has a separate step.
    for name, url, revision in [
        ("hex", "https://github.com/anomalyco/hex.git", HEX_REV),
        ("transcribe", "https://github.com/anomalyco/transcribe.cpp.git", RUNTIME_REV),
    ]:
        dest = root / name
        if not dest.exists():
            run(["git", "init", "-q", str(dest)], root, name + "-init", env)
            run(["git", "-C", str(dest), "fetch", "--depth", "1", url, revision], root, name + "-fetch", env)
            run(["git", "-C", str(dest), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", "FETCH_HEAD"], root, name + "-checkout", env)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                              capture_output=True, text=True, env=env, check=True).stdout.strip()
        if head != revision:
            raise SystemExit("source commit mismatch")
        dirty = subprocess.run(["git", "-C", str(dest), "status", "--porcelain"],
                               capture_output=True, text=True, env=env, check=True).stdout
        if dirty:
            raise SystemExit("source checkout is dirty")
    run(["git", "-C", str(root / "hex"), "fetch", "--depth", "1",
         "https://github.com/anomalyco/hex.git", INSTALLED_SOURCE_REV], root, "installed-source-fetch", env)
    for rel in ["Cargo.toml", "src/parakeet.rs", "src/transcription.rs",
                "src/transcription/gguf_session.rs", "src/transcription/engine/streaming.rs",
                "src/transcription/pause_segmenter.rs", "src/dictation.rs"]:
        target = root / "installed-version-source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out, (root / "installed-source-show.log").open("ab") as err:
            result = subprocess.run(["git", "-C", str(root / "hex"), "show", INSTALLED_SOURCE_REV + ":" + rel],
                                    stdout=out, stderr=err, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
    runner = root / "runner"
    (runner / "src/transcription").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "runner.rs", runner / "src/main.rs")
    # Reuse exact Hex pause logic rather than implementing a second detector.
    streaming = (root / "hex/src/transcription/engine/streaming.rs").read_text()
    trait = streaming[streaming.index("pub(super) trait StreamFeed"):streaming.index("/// Silence a dictation stream")]
    cap = streaming[streaming.index("/// Pause audio a dictation stream"):streaming.index("#[derive(Default)]\npub(super) struct FeedStats")]
    (runner / "src/hex_pause.rs").write_text("use std::collections::VecDeque;\nuse crate::Partial;\n" + trait + cap)
    pause = (root / "hex/src/transcription/pause_segmenter.rs").read_text().split("#[cfg(test)]")[0]
    (runner / "src/transcription/pause_segmenter.rs").write_text(pause)
    wrapper = root / "transcribe/bindings/rust/transcribe-cpp"
    (runner / "Cargo.toml").write_text(
        '[package]\nname = "hex-grok-repro"\nversion = "0.1.0"\nedition = "2021"\n'
        '[dependencies]\nhound = "3.5"\nserde_json = "1"\n'
        f'transcribe-cpp = {{ path = "{wrapper}", features = ["metal"] }}\n')
    lock = HERE / "runtime.lock"
    shutil.copyfile(lock if lock.exists() else root / "hex/Cargo.lock", runner / "Cargo.lock")
    command = ["cargo", "build", "--release", "--manifest-path", str(runner / "Cargo.toml")]
    if lock.exists():
        command.append("--locked")
    run(command, root, "build", env)
    shutil.copyfile(runner / "Cargo.lock", root / "runtime.lock")
    executable = root / "target/release/hex-grok-repro"
    identity = {"hexCommit": HEX_REV, "runtimeCommit": RUNTIME_REV,
                "executable": str(executable), "executableSha256": digest(executable),
                "runnerSourceSha256": digest(HERE / "runner.rs"),
                "lockSha256": digest(root / "runtime.lock"), "buildExit": 0,
                "sourceKind": "pinned source candidate; installed binary equality unverified"}
    for label, args in [("rustc", ["rustc", "--version"]), ("cargo", ["cargo", "--version"]),
                        ("cmake", ["cmake", "--version"]), ("clang", ["clang", "--version"])]:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        # Retain only version text, never diagnostic dumps.
        identity[label] = result.stdout.splitlines()[0] if result.returncode == 0 else "unverified"
    (root / "build-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print("executable ready; identity retained in system temp", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, help="existing marked build directory")
    args = parser.parse_args()
    root = args.work or Path(tempfile.mkdtemp(prefix="hex-grok-repro-", dir="/private/tmp"))
    if root.parent != Path("/private/tmp") or not root.name.startswith("hex-grok-repro-"):
        raise SystemExit("build directory must be isolated under /private/tmp")
    print("build directory: " + str(root), flush=True)
    prepare(root)
