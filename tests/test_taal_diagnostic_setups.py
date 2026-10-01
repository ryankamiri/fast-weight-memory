import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.dataloader import create_dataloader
from data.taal_conflict_control import build_examples
from evaluation.taal.read_path_diagnostic import relocate_fact
from training.config import BridgeMemoryDataConfig, BridgeMemoryLossConfig, BridgeMemoryRecordRange


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class TaalDiagnosticSetupsTests(unittest.TestCase):
    def test_conflicting_pair_differs_only_at_evicted_answer(self):
        with patch("data.taal_conflict_control.single_token_labels") as labels:
            labels.return_value = [("one", 1000), ("two", 1001), ("three", 1002), ("four", 1003)]
            rows = build_examples(CharacterTokenizer(), pairs=2, window=128, gap=512)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["final_answer_position"] - rows[0]["fact_position"], 512)
        self.assertEqual(rows[0]["final_query_position"], rows[1]["final_query_position"])
        self.assertEqual(
            sum(left != right for left, right in zip(rows[0]["input_ids"], rows[1]["input_ids"])),
            1,
        )
        self.assertEqual(rows[0]["target_token_id"], 1000)
        self.assertEqual(rows[1]["target_token_id"], 1001)

    def test_relocation_preserves_length_query_and_target(self):
        with patch("data.taal_conflict_control.single_token_labels") as labels:
            labels.return_value = [("one", 1000), ("two", 1001), ("three", 1002), ("four", 1003)]
            row = build_examples(CharacterTokenizer(), pairs=2, window=128, gap=512)[0]
        moved = relocate_fact(row, CharacterTokenizer(), 200)
        self.assertEqual(len(moved["input_ids"]), len(row["input_ids"]))
        self.assertEqual(moved["final_query_position"], row["final_query_position"])
        self.assertEqual(moved["input_ids"][row["final_query_position"]:], row["input_ids"][row["final_query_position"]:])
        self.assertEqual(len(moved["input_ids"]) - moved["fact_position"], 200)
        self.assertEqual(moved["input_ids"][moved["fact_position"]], 1000)

    def test_local_jsonl_loads_as_bridge_memory(self):
        with patch("data.taal_conflict_control.single_token_labels") as labels:
            labels.return_value = [("one", 1000), ("two", 1001), ("three", 1002), ("four", 1003)]
            rows = build_examples(CharacterTokenizer(), pairs=2, window=128, gap=192)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = BridgeMemoryDataConfig(
                dataset_id=str(path), dataset_config="local-jsonl",
                batch_size=1, seq_len=512, num_workers=0,
            )
            records = BridgeMemoryRecordRange(
                start=0, end=4, conditions=["micro_conflict"], query_variants=["exact"],
            )
            loss = BridgeMemoryLossConfig(all_tokens_weight=0, delayed_answer_weight=1)
            batches = list(create_dataloader(config, records, loss, shuffle=False, seed=42))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0]["target_token_ids"].item(), 1000)
