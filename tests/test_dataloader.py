import unittest
from unittest.mock import patch

import torch
from datasets import Dataset

from data.dataloader import CausalLMCollator, create_dataloader


class DataLoaderTests(unittest.TestCase):
    def test_collator_preserves_tokens_and_does_not_shift_labels(self):
        result = CausalLMCollator()([
            {"input_ids": [1, 2, 3]}, {"input_ids": [5, 6, 7]},
        ])
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [5, 6, 7]]))
        self.assertEqual(result["input_ids"].dtype, torch.long)
        torch.testing.assert_close(result["labels"], result["input_ids"])
        result["labels"][0, 0] = -100
        self.assertEqual(result["input_ids"][0, 0].item(), 1)

    def test_filter_batching_and_worker_sharding(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                source = Dataset.from_dict({
                    "input_ids": [[i] * (2 if i == 0 else 5 if i == 7 else 4) for i in range(8)],
                    "domain": ["books"] * 8,
                }).to_iterable_dataset(num_shards=2)
                with patch("data.dataloader.load_dataset", return_value=source) as load:
                    loader = create_dataloader(batch_size=2, seq_len=4, num_workers=workers)
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
        with patch("data.dataloader.load_dataset", return_value=source):
            loader = create_dataloader()
        self.assertEqual(loader.batch_size, 1)
        self.assertEqual(loader.num_workers, 2)
        self.assertIsInstance(loader.collate_fn, CausalLMCollator)
        for args in ({"batch_size": 0}, {"seq_len": -1}, {"num_workers": -1}):
            with self.assertRaises(ValueError):
                create_dataloader(**args)


if __name__ == "__main__":
    unittest.main()
