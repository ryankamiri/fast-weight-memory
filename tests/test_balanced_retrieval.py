"""In-memory checks for paired retrieval, split safety and exact token eviction."""

from collections import Counter, defaultdict
from dataclasses import replace
import unittest
from unittest.mock import patch
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from data.balanced_retrieval import RetrievalConfig, build_bundle, export_bundle, token_span
from data.dataloader import CausalLMCollator
from evaluation.shared.benchmarks import ControlledBenchmark


def tiny_tokenizer():
    vocabulary = {"<eos>": 0}
    for character in sorted(ByteLevel.alphabet()):
        vocabulary[character] = len(vocabulary)
    backend = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")


class BalancedRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tiny_tokenizer()
        cls.config = RetrievalConfig(
            train_families=1, val_families=1, test_families=1,
            max_sequence_tokens=2048,
        )
        cls.bundle = build_bundle(cls.tokenizer, cls.config)

    def test_queries_orders_and_remappings_are_balanced(self):
        for rows in self.bundle.values():
            self.assertEqual(len(rows), 8)
            groups = defaultdict(list)
            bindings = defaultdict(set)
            positions = Counter()
            for row in rows:
                groups[row["group_id"]].append(row)
                metadata = row["metadata"]
                positions[metadata["queried_fact_order"]] += 1
                for key, value in metadata["bindings"].items():
                    bindings[key].add(value)
            self.assertEqual(positions, {0: 4, 1: 4})
            for variants in groups.values():
                self.assertEqual(len(variants), 2)
                self.assertEqual(len({row["metadata"]["prefix_text"] for row in variants}), 1)
                self.assertEqual(len({row["answer"] for row in variants}), 2)
                prefixes = set()
                for row in variants:
                    end = row["metadata"]["query_start_token"]
                    prefixes.add(tuple(row["prompt_input_ids"][:end]))
                self.assertEqual(len(prefixes), 1)
            for values in bindings.values():
                self.assertEqual(len(values), 2)

    def test_splits_are_disjoint_and_reproducible(self):
        self.assertEqual(self.bundle, build_bundle(self.tokenizer, self.config))
        keys_seen = set()
        families_seen = set()
        ids_seen = set()
        for rows in self.bundle.values():
            keys = set()
            families = set()
            for row in rows:
                self.assertNotIn(row["example_id"], ids_seen)
                ids_seen.add(row["example_id"])
                keys.update(row["metadata"]["bindings"])
                families.add(row["metadata"]["family_id"])
            self.assertFalse(keys & keys_seen)
            self.assertFalse(families & families_seen)
            keys_seen.update(keys)
            families_seen.update(families)
        larger = build_bundle(self.tokenizer, replace(self.config, train_families=2))
        self.assertEqual(self.bundle["test"], larger["test"])
        different = build_bundle(self.tokenizer, replace(self.config, seed=43))
        self.assertNotEqual(self.bundle["train"], different["train"])

    def test_exact_gap_training_labels_and_shared_adapter(self):
        benchmark = ControlledBenchmark()
        for rows in self.bundle.values():
            for row in rows:
                metadata = row["metadata"]
                prompt_ids = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
                self.assertEqual(prompt_ids, row["prompt_input_ids"])
                self.assertEqual(row["input_ids"][:len(prompt_ids)], prompt_ids)
                self.assertEqual(row["labels"][:len(prompt_ids)], [-100] * len(prompt_ids))
                self.assertEqual(row["labels"][len(prompt_ids):], row["input_ids"][len(prompt_ids):])
                self.assertGreater(metadata["answer_token_count"], 1)
                self.assertEqual(row["input_ids"][-1], self.tokenizer.eos_token_id)
                query_start = metadata["query_start_token"]
                for start, end in metadata["fact_token_spans"]:
                    self.assertGreaterEqual(query_start - end, 192)
                    self.assertLessEqual(end, query_start - 128)
                    self.assertGreater(end, start)
                example = benchmark.prepare(row, self.tokenizer)
                self.assertEqual(list(example.prompt.token_ids), prompt_ids)
                self.assertEqual(benchmark.score(example, row["answer"])["exact_match"], 1)

    def test_capacity_variants_and_budget_failures(self):
        config = replace(self.config, fact_counts=(4,), max_sequence_tokens=4096)
        rows = build_bundle(self.tokenizer, config)["train"]
        self.assertEqual(len(rows), 32)
        counts = Counter(row["metadata"]["queried_fact_order"] for row in rows)
        self.assertEqual(counts, {0: 8, 1: 8, 2: 8, 3: 8})
        with self.assertRaisesRegex(ValueError, "max_sequence_tokens"):
            build_bundle(self.tokenizer, replace(self.config, max_sequence_tokens=200))
        with self.assertRaises(ValueError):
            build_bundle(self.tokenizer, replace(self.config, min_gap_tokens=127))
        with self.assertRaises(ValueError):
            build_bundle(self.tokenizer, replace(self.config, fact_counts=(2, 2)))

    def test_training_collator_and_multitoken_label_pool(self):
        config = replace(self.config, label_pool=("blue harbor", "silver mountain"))
        row = build_bundle(self.tokenizer, config)["train"][0]
        batch = CausalLMCollator()([row])
        self.assertEqual(batch["input_ids"].tolist(), [row["input_ids"]])
        self.assertEqual(batch["labels"].tolist(), [row["labels"]])
        self.assertGreater(row["metadata"]["answer_token_count"], 2)

    def test_eight_facts_and_existing_destination_guard(self):
        config = replace(self.config, fact_counts=(8,), max_sequence_tokens=4096)
        rows = build_bundle(self.tokenizer, config)["test"]
        self.assertEqual(len(rows), 128)
        self.assertEqual(len({row["group_id"] for row in rows}), 16)
        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "never overwrite"):
                export_bundle(self.tokenizer, self.config, Path("unused"))

    def test_boundary_overlap_is_counted_conservatively(self):
        self.assertEqual(token_span([(0, 5), (5, 12), (12, 15)], 4, 10), [0, 2])
        with self.assertRaises(ValueError):
            token_span([(0, 2)], 3, 4)


if __name__ == "__main__":
    unittest.main()
