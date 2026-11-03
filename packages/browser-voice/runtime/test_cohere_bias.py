"""Cohere's names bias selector, compiled from hex/cohere-sequence-bias.patch with clang++.

The research selector checks (ct-pbias verify-selector.py), carried over to the
run extension: the patch's own SequenceBias struct is extracted, built against
stub ggml and transcribe.h types, and run. No model or Metal runtime is loaded.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PATCH = Path(__file__).resolve().parent / "hex/cohere-sequence-bias.patch"

STUBS = """#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
struct ggml_tensor { long ne[1]; std::vector<float> values; };
void ggml_backend_tensor_get(ggml_tensor * t, void * out, size_t offset, size_t n) {
    std::memcpy(out, reinterpret_cast<char *>(t->values.data()) + offset, n);
}
enum transcribe_status { TRANSCRIBE_OK, TRANSCRIBE_ERR_INVALID_ARG };
struct transcribe_ext { uint64_t size; uint32_t kind; };
const uint32_t TRANSCRIBE_EXT_KIND_COHERE_SEQUENCE_BIAS = 0x42534843u;
struct transcribe_cohere_sequence_bias_ext {
    transcribe_ext ext; float bias; const int32_t * tokens; const int32_t * lengths; int32_t n_sequences;
};
struct transcribe_run_params { const transcribe_ext * family; };
transcribe_status transcribe_ext_check(const transcribe_ext * ext, uint32_t kind, uint64_t size) {
    return ext->kind == kind && ext->size >= size ? TRANSCRIBE_OK : TRANSCRIBE_ERR_INVALID_ARG;
}
"""

CHECKS = """
static transcribe_status parse(float bias, std::vector<std::vector<int32_t>> names, SequenceBias & out, int vocab = 6,
                               uint32_t kind = TRANSCRIBE_EXT_KIND_COHERE_SEQUENCE_BIAS) {
    std::vector<int32_t> tokens, lengths;
    for (const auto & ids : names) { tokens.insert(tokens.end(), ids.begin(), ids.end()); lengths.push_back(ids.size()); }
    transcribe_cohere_sequence_bias_ext ext{{sizeof(ext), kind}, bias, tokens.data(), lengths.data(),
                                            static_cast<int32_t>(lengths.size())};
    transcribe_run_params params{&ext.ext};
    return SequenceBias::from_params(&params, vocab, out);
}

int main() {
    const std::vector<std::vector<int32_t>> names = {{1, 2, 3}, {1, 4, 3}, {1, 2, 3}};
    SequenceBias b;
    assert(parse(2.5f, names, b) == TRANSCRIBE_OK);
    assert(b.sequences.size() == 5);  // every prefix; shared [1] and [1,2] once
    ggml_tensor t{{6}, {0, 0, 0, 0, 0, 2}};
    assert(b.select(&t, 0, {}) == 1);  // name start applies with empty history
    t.values = {0, -10, 0, 0, -10, 2};
    assert(b.select(&t, 0, {1}) == 2);  // proper two token prefix
    assert(b.select(&t, 0, {1, 2}) == 3);  // full name completion
    assert(b.select(&t, 0, {1, 2, 0}) == 5);  // suffix must be adjacent
    assert(b.select(&t, 0, {0, 1, 2}) == 3);  // arbitrary earlier history
    t.values = {0, -10, -10, 0, -10, 3};
    assert(b.select(&t, 0, {1, 2}) == 5);  // duplicate names never double a prefix
    b.sequences = {{1, 2, 3}, {2, 3}};
    t.values = {0, -10, -10, 0, -10, 4};
    assert(b.select(&t, 0, {1, 2}) == 3);  // different matching contexts add
    assert(b.select(&t, 0, {2}) == 5);  // only one match is insufficient
    b.amount = 0;
    assert(b.select(&t, 0, {1, 2}) == 5);
    SequenceBias zero;
    assert(parse(0.0f, names, zero) == TRANSCRIBE_OK && zero.amount == 0.0f && zero.sequences.empty());  // bypass
    transcribe_run_params none{nullptr};
    assert(SequenceBias::from_params(&none, 6, zero) == TRANSCRIBE_OK && zero.amount == 0.0f);  // no extension is off
    assert(parse(2.5f, {{1, 6}}, zero) == TRANSCRIBE_ERR_INVALID_ARG);  // token outside the vocabulary
    assert(parse(2.5f, {{1}, {}}, zero) == TRANSCRIBE_ERR_INVALID_ARG);  // empty sequence
    assert(parse(2.5f, {}, zero) == TRANSCRIBE_ERR_INVALID_ARG);  // nothing to bias
    assert(parse(NAN, names, zero) == TRANSCRIBE_ERR_INVALID_ARG);
    assert(parse(2.5f, names, zero, 6, 0) == TRANSCRIBE_ERR_INVALID_ARG);  // another extension kind
}
"""


def patched_selector():
    """The SequenceBias struct exactly as the patch adds it to model.cpp."""
    section = PATCH.read_text().split("diff --git a/src/arch/cohere/model.cpp")[1].split("diff --git")[0]
    added = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    start = added.index("struct SequenceBias {")
    return added[start:added.index("\n};\n", start) + 4]


@unittest.skipUnless(shutil.which("clang++"), "clang++ is not installed")
class SequenceBiasTest(unittest.TestCase):
    def test_patched_selector_biases_every_prefix_once_and_refuses_invalid_extensions(self):
        with tempfile.TemporaryDirectory(prefix="speakeasy-cohere-bias-") as directory:
            source, binary = Path(directory) / "selector.cpp", Path(directory) / "selector"
            source.write_text(STUBS + patched_selector() + CHECKS)
            subprocess.run(["clang++", "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
