"""Compile the production Rust request parser without loading a model."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

WORKER = Path(__file__).resolve().parent / "hex/worker.rs"
CHECKS = r"""
fn frame(op: u8, bias: f32, samples: &[f32]) -> Vec<u8> {
    let mut bytes = vec![op];
    bytes.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    if op == b'B' { bytes.extend_from_slice(&bias.to_le_bytes()); }
    for sample in samples { bytes.extend_from_slice(&sample.to_le_bytes()); }
    bytes
}

fn main() {
    // Queued frames preserve sample boundaries and never carry bias into T.
    let mut bytes = Vec::new();
    for (op, bias) in [(b'B', 2.5), (b'T', 0.0), (b'B', 1.25), (b'T', 0.0)] {
        bytes.extend(frame(op, bias, &[0.25, -0.0]));
    }
    let mut input = std::io::Cursor::new(bytes);
    for expected in [2.5, 0.0, 1.25, 0.0] {
        let Request::Take { samples, names_bias } = read_request(&mut input).unwrap().unwrap() else {
            panic!("expected a take");
        };
        assert_eq!(names_bias, expected);
        assert_eq!(samples[0], 0.25);
        assert_eq!(samples[1].to_bits(), (-0.0f32).to_bits());
    }
    assert!(read_request(&mut input).unwrap().is_none());
    for invalid in [1.0, -1.25, f32::MAX, f32::NAN, f32::INFINITY] {
        assert!(read_request(&mut std::io::Cursor::new(frame(b'B', invalid, &[0.25]))).is_err(),
                "unoffered names bias accepted: {invalid}");
    }
}
"""


@unittest.skipUnless(shutil.which("rustc"), "rustc is not installed")
class HexProtocolTest(unittest.TestCase):
    def test_queued_takes_keep_their_bias_and_refuse_unoffered_values(self):
        source = WORKER.read_text()
        parser = source[source.index("/// A live stream's"):source.index("\nfn emit(")]
        limit = next(line for line in source.splitlines() if line.startswith("const MAX_SAMPLES:"))
        with tempfile.TemporaryDirectory(prefix="speakeasy-hex-protocol-") as directory:
            code, binary = Path(directory) / "protocol.rs", Path(directory) / "protocol"
            code.write_text("use std::{error::Error, io::{self, Read}};\n"
                            + limit + "\n" + parser + CHECKS)
            subprocess.run(["rustc", "--edition=2021", str(code), "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
