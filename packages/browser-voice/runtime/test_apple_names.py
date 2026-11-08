"""Compile the worker's names functions without loading Apple speech assets."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from names import names_digest, names_option

WORKER = Path(__file__).resolve().parent / "apple_speech/worker.swift"


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("swiftc"), "requires macOS and swiftc")
class AppleNamesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="speakeasy-apple-names-")
        cls.addClassCleanup(cls.directory.cleanup)
        source = WORKER.read_text()
        declarations = "\n".join(line for line in source.splitlines()
                                 if line.startswith(("enum Failure:", "let maxNames =")))
        functions = source[source.index("func parseNames("):source.index("/// The transcriber this worker runs")]
        probe = Path(cls.directory.name) / "names.swift"
        cls.binary = Path(cls.directory.name) / "names"
        probe.write_text("import Foundation\nimport CryptoKit\n" + declarations + "\n" + functions + """
let names = try parseNames(CommandLine.arguments[1])
let result: [String: Any] = ["names": names, "namesDigest": namesDigest(names)]
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result))
""")
        subprocess.run(["swiftc", str(probe), "-o", str(cls.binary)], check=True,
                       stdin=subprocess.DEVNULL, capture_output=True)

    def assert_host_parity(self, names):
        names_option(names)
        reply = subprocess.run([str(self.binary), json.dumps(names, ensure_ascii=False)], check=True,
                               stdin=subprocess.DEVNULL, capture_output=True)
        self.assertEqual(json.loads(reply.stdout), {"names": names, "namesDigest": names_digest(names)})

    def test_byte_distinct_unicode_names_keep_both_spellings(self):
        self.assert_host_parity(["Caf\u00e9", "Cafe\u0301"])

    def test_zero_width_spaces_in_host_names_are_preserved(self):
        self.assert_host_parity(["\u200bGrok\u200b"])


if __name__ == "__main__":
    unittest.main()
