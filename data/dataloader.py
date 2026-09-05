"""Single-GPU loading of pre-tokenized, independent training records."""

from functools import partial
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def has_sequence_length(example, seq_len):
    return len(example["input_ids"]) == seq_len


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
    revision: str | None = None,
    start: int = 0,
    end: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    shuffle_buffer_size: int = 128,
) -> DataLoader:
    """Stream batches of records already matching seq_len; no further truncation."""
    for name, value, minimum in (
        ("batch_size", batch_size, 1), ("seq_len", seq_len, 1), ("num_workers", num_workers, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    if start < 0 or (end is not None and end <= start):
        raise ValueError("Expected a nonempty half-open record range [start, end)")
    load_options = {}
    if revision is not None:
        load_options["revision"] = revision
    # Conversion preserves one row per source index. Parquet predicates select
    # the same original records for every worker and skip unrelated row groups.
    filters = []
    if start:
        filters.append(("source_record_index", ">=", start))
    if end is not None:
        filters.append(("source_record_index", "<", end))
    if filters:
        load_options["filters"] = filters
    dataset = load_dataset(dataset_id, split="train", streaming=True, **load_options)
    dataset = dataset.filter(partial(has_sequence_length, seq_len=seq_len))
    # Keep only IDs before buffering; preserve each source record independently.
    dataset = dataset.select_columns(["input_ids"])
    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CausalLMCollator(),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
