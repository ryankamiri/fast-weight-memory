import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from datasets import load_dataset as load_hf_dataset
import pyarrow as pa
import pyarrow.parquet as pq

from data.dataloader import BridgeMemoryCollator, CausalLMCollator, create_dataloader
from training.config import (
    BridgeMemoryDataConfig,
    BridgeMemoryLossConfig,
    BridgeMemoryRecordRange,
    CausalLMDataConfig,
    CausalLMLossConfig,
    RecordRange,
)


class DataLoaderTests(unittest.TestCase):
    def test_record_ranges_are_disjoint_with_multiple_workers(self):
        with TemporaryDirectory() as folder:
            paths = []
            for shard in range(2):
                path = Path(folder) / f"train-{shard}.parquet"
                rows = [{"source_record_index": i, "input_ids": [i] * (3 if i == 2 else 4)}
                        for i in range(shard * 6, (shard + 1) * 6)]
                pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
                paths.append(str(path))

            def load_local(dataset_id, **kwargs):
                return load_hf_dataset("parquet", data_files={"train": paths}, **kwargs)

            for workers in (0, 2):
                config = CausalLMDataConfig(
                    dataset_id="local", seq_len=4, num_workers=workers,
                )
                loss = CausalLMLossConfig()
                with patch("data.dataloader.load_dataset", side_effect=load_local):
                    val = create_dataloader(
                        config, RecordRange(0, 4), loss, shuffle=False, seed=42,
                    )
                    training = create_dataloader(
                        config, RecordRange(4, 11), loss, shuffle=True, seed=42,
                    )
                    duplicate = create_dataloader(
                        config, RecordRange(4, 11), loss, shuffle=True, seed=42,
                    )
                val_ids = [int(item["input_ids"][0, 0]) for item in val]
                train_ids = [int(item["input_ids"][0, 0]) for item in training]
                repeat_ids = [int(item["input_ids"][0, 0]) for item in duplicate]
                self.assertEqual(sorted(val_ids), [0, 1, 3])
                self.assertEqual(sorted(train_ids), list(range(4, 11)))
                self.assertEqual(train_ids, repeat_ids)
                self.assertFalse(set(val_ids) & set(train_ids))

    def test_collator_preserves_tokens_and_does_not_shift_labels(self):
        result = CausalLMCollator()([
            {"input_ids": [1, 2, 3]}, {"input_ids": [5, 6, 7]},
        ])
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [5, 6, 7]]))
        self.assertEqual(result["input_ids"].dtype, torch.long)
        torch.testing.assert_close(result["labels"], result["input_ids"])
        result["labels"][0, 0] = -100
        self.assertEqual(result["input_ids"][0, 0].item(), 1)

    def test_bridge_collator_appends_answer_and_builds_delayed_mask(self):
        example = {
            "input_ids": [1, 2, 3],
            "target_token_id": 7,
            "candidate_token_ids": [7, 8, 9],
            "condition": "bridge",
            "query_variant": "exact",
        }
        delayed_only = BridgeMemoryCollator(
            all_tokens_weight=0.0, delayed_answer_weight=1.0,
        )([example])
        torch.testing.assert_close(delayed_only["input_ids"], torch.tensor([[1, 2, 3, 7]]))
        self.assertNotIn("labels", delayed_only)
        torch.testing.assert_close(
            delayed_only["delayed_labels"], torch.tensor([[-100, -100, -100, 7]]),
        )
        self.assertEqual(delayed_only["conditions"], ["bridge"])

        combined = BridgeMemoryCollator(
            all_tokens_weight=1.0, delayed_answer_weight=1.0,
        )([example, {**example, "target_token_id": 8}])
        torch.testing.assert_close(combined["labels"], combined["input_ids"])
        self.assertEqual(combined["input_ids"].shape, (2, 4))
        torch.testing.assert_close(
            combined["delayed_labels"][:, -1], torch.tensor([7, 8]),
        )
        with self.assertRaisesRegex(ValueError, "equal sequence lengths"):
            BridgeMemoryCollator(all_tokens_weight=0.0, delayed_answer_weight=1.0)(
                [example, {**example, "input_ids": [1, 2]}],
            )
        with self.assertRaisesRegex(ValueError, "equal candidate counts"):
            BridgeMemoryCollator(all_tokens_weight=0.0, delayed_answer_weight=1.0)(
                [example, {**example, "candidate_token_ids": [7, 8]}],
            )

    def test_bridge_loader_filters_facts_conditions_and_length(self):
        source = Dataset.from_list([
            {
                "fact_id": fact,
                "input_ids": [fact, 1, 2] if fact != 3 else [fact] * 8,
                "target_token_id": 10 + fact,
                "candidate_token_ids": [10 + fact, 20 + fact],
                "condition": condition,
                "query_variant": "exact",
            }
            for fact in range(4)
            for condition in ("bridge", "no_bridge")
        ]).to_iterable_dataset()
        config = BridgeMemoryDataConfig(
            dataset_id="memory", dataset_config="t4-s2-c1",
            batch_size=1, seq_len=6, num_workers=0,
        )
        records = BridgeMemoryRecordRange(
            start=1, end=4, split="test", conditions=["bridge"],
        )
        with patch("data.dataloader.load_dataset", return_value=source):
            loader = create_dataloader(
                config, records,
                BridgeMemoryLossConfig(all_tokens_weight=0.0, delayed_answer_weight=1.0),
                shuffle=False, seed=42,
            )
        rows = list(loader)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["input_ids"][0, 0].item() for row in rows], [1, 2])
        self.assertTrue(all(row["conditions"] == ["bridge"] for row in rows))

    def test_filter_batching_and_worker_sharding(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                source = Dataset.from_dict({
                    "input_ids": [[i] * (2 if i == 0 else 5 if i == 7 else 4) for i in range(8)],
                    "domain": ["books"] * 8,
                }).to_iterable_dataset(num_shards=2)
                config = CausalLMDataConfig(batch_size=2, seq_len=4, num_workers=workers)
                with patch("data.dataloader.load_dataset", return_value=source) as load:
                    loader = create_dataloader(
                        config, RecordRange(), CausalLMLossConfig(), shuffle=True, seed=42,
                    )
                load.assert_called_once_with("ryankamiri/prolong-qwen", split="train", streaming=True)
                seen = []
                for batch in loader:
                    self.assertEqual(set(batch), {"input_ids", "labels"})
                    self.assertEqual(batch["input_ids"].shape[1], 4)
                    torch.testing.assert_close(batch["labels"], batch["input_ids"])
                    seen.extend(batch["input_ids"][:, 0].tolist())
                self.assertEqual(sorted(seen), list(range(1, 7)))

    def test_defaults_and_invalid_arguments(self):
        source = Dataset.from_dict({"input_ids": [[1]]}).to_iterable_dataset()
        config = CausalLMDataConfig()
        with patch("data.dataloader.load_dataset", return_value=source):
            loader = create_dataloader(
                config, config.train, CausalLMLossConfig(), shuffle=True, seed=42,
            )
        self.assertEqual(loader.batch_size, 1)
        self.assertEqual(loader.num_workers, 2)
        self.assertIsInstance(loader.collate_fn, CausalLMCollator)
        for name, value in (("batch_size", 0), ("seq_len", -1), ("num_workers", -1)):
            invalid = CausalLMDataConfig()
            setattr(invalid, name, value)
            with self.assertRaises(ValueError):
                create_dataloader(
                    invalid, invalid.train, CausalLMLossConfig(), shuffle=True, seed=42,
                )


if __name__ == "__main__":
    unittest.main()
