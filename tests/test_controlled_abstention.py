"""Verify missing-key labels, counterbalancing and abstention denominators."""

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from data.controlled_abstention import build_examples
from evaluation.shared.benchmarks import ControlledBenchmark
from evaluation.shared.runner import rescore
from evaluation.storage import append_result


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class ControlledAbstentionTests(unittest.TestCase):
    def test_deterministic_balanced_and_counterbalanced(self):
        rows = build_examples(pairs=3, fact_count=4, filler_repetitions=2)
        self.assertEqual(
            rows, build_examples(pairs=3, fact_count=4, filler_repetitions=2)
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row["metadata"]["answerable"] for row in rows), 6)
        roles = defaultdict(list)
        prefixes = defaultdict(list)
        for row in rows:
            metadata = row["metadata"]
            prefix = row["prompt"].split("What is Record ")[0]
            self.assertEqual(metadata["query_key"] in prefix, metadata["answerable"])
            roles[(metadata["pair_id"], metadata["query_key"])].append(
                metadata["answerable"]
            )
            prefixes[row["group_id"]].append(prefix)
        for values in roles.values():
            self.assertEqual(sorted(values), [False, True])
        for values in prefixes.values():
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0], values[1])

    def test_correct_and_false_abstention_are_separate(self):
        adapter = ControlledBenchmark()
        for row in build_examples(pairs=1, filler_repetitions=0):
            example = adapter.prepare(row, Tokenizer())
            abstention = adapter.score(example, " UNKNOWN ")
            answer = adapter.score(example, "V9999999")
            if row["metadata"]["answerable"]:
                self.assertEqual(abstention["false_abstention_rate"], 1)
                self.assertNotIn("correct_abstention_rate", abstention)
            else:
                self.assertEqual(abstention["correct_abstention_rate"], 1)
                self.assertEqual(answer["unanswerable_nonabstention_rate"], 1)
                self.assertNotIn("false_abstention_rate", abstention)

    def test_always_abstaining_fails_joint_correctness(self):
        adapter = ControlledBenchmark()
        examples = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for row in build_examples(pairs=2, filler_repetitions=0):
                example = adapter.prepare(row, Tokenizer())
                examples.append(example)
                append_result(
                    output / "predictions.jsonl",
                    {"example_id": example.example_id, "hypothesis": "UNKNOWN"},
                )
            summary = rescore(adapter, examples, output)
        self.assertEqual(summary["metrics"]["exact_match"], 0.5)
        self.assertEqual(summary["metrics"]["correct_abstention_rate"], 1)
        self.assertEqual(summary["metrics"]["false_abstention_rate"], 1)
        self.assertEqual(summary["group_all_correct"], 0)
        self.assertEqual(summary["strata"]["answerable=False"]["examples"], 4)

    def test_invalid_abstention_annotations_rejected(self):
        adapter = ControlledBenchmark()
        row = build_examples(pairs=1, filler_repetitions=0)[0]
        row["metadata"]["answerable"] = False
        with self.assertRaisesRegex(ValueError, "Unanswerable"):
            adapter.prepare(row, Tokenizer())
        row["metadata"]["answerable"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            adapter.prepare(row, Tokenizer())

    def test_existing_controlled_rows_keep_original_scores(self):
        adapter = ControlledBenchmark()
        example = adapter.prepare(
            {"example_id": "old", "prompt": "Question?", "answer": "amber"}, Tokenizer()
        )
        self.assertEqual(adapter.score(example, "amber"), {"exact_match": 1})


if __name__ == "__main__":
    unittest.main()
