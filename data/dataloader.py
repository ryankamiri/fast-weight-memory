"""Single-GPU loading of pre-tokenized, independent training records."""

from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def has_sequence_length(example, seq_len):
    return len(example["input_ids"]) == seq_len


class CausalLMCollator:
    def __call__(self, examples) -> dict[str, torch.Tensor]:
        # [B, S], no padding or packing across record boundaries.
        input_ids = torch.stack([
            torch.tensor(example["input_ids"], dtype=torch.long)
            for example in examples
        ])
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def create_dataloader(
    dataset_id: str = "ryankamiri/prolong-qwen",
    batch_size: int = 1,
    seq_len: int = 65536,
    num_workers: int = 2,
) -> DataLoader:
    """Stream batches of records already matching seq_len; no further truncation."""
    for name, value, minimum in (
        ("batch_size", batch_size, 1), ("seq_len", seq_len, 1), ("num_workers", num_workers, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    dataset = load_dataset(dataset_id, split="train", streaming=True)
    dataset = dataset.filter(partial(has_sequence_length, seq_len=seq_len))
    # Keep only IDs before buffering; preserve each source record independently.
    dataset = dataset.select_columns(["input_ids"])
    dataset = dataset.shuffle(seed=42, buffer_size=128)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CausalLMCollator(),
        pin_memory=True,
        drop_last=False,
    )
