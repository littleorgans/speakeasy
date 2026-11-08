"""The backend side of the Names list: the host's canonical list passes unchanged, anything else fails."""

import unittest

from names import MAX_NAME_BYTES, MAX_NAMES, names_digest, names_option

EIGHT = ["Grok", "Claude", "Codex", "GPT", "GPT-SOL", "TM", "Hibiki", "Qwen"]


class NamesTests(unittest.TestCase):
    def test_the_digest_is_the_one_every_host_and_worker_computes(self):
        self.assertEqual(names_digest(EIGHT), "59d6626926c652e50c7e90fea5f95afdea75f6c27e11aa07c2996b89d51fc068")
        self.assertEqual(names_digest([]), "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945")

    def test_a_canonical_list_passes_unchanged_as_a_copy(self):
        widest = [f"{n:02d}" + "é" * 63 for n in range(MAX_NAMES)]
        self.assertEqual(len(widest[0].encode()), MAX_NAME_BYTES)
        for value in (EIGHT, [], widest, ["Siobhán", "Speak Easy"]):
            with self.subTest(count=len(value)):
                self.assertEqual(names_option(value), value)
                self.assertIsNot(names_option(value), value)

    def test_anything_the_host_would_have_changed_or_refused_fails_naming_it(self):
        for bad in ("Grok", None, {"Grok": 1}, [1], [""], [" Grok"], ["Grok\t"], ["Gr\x00ok"], ["Gr\nok"],
                    ["Gr\x7fok"], ["Gr\x85ok"], ["Grok, Claude"], ["x" * 129], ["é" * 65], ["Grok", "Grok"],
                    [str(n) for n in range(65)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                names_option(bad)
        with self.assertRaisesRegex(ValueError, "'Grok, Claude'"):
            names_option(["Grok, Claude"])


if __name__ == "__main__":
    unittest.main()
