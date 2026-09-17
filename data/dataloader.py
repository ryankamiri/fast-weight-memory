"""Single-GPU loading of pre-tokenized, independent training records."""

from functools import partial
import random
from typing import overload

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from training.config import (
    BridgeMemoryDataConfig,
    BridgeMemoryLossConfig,
    BridgeMemoryRecordRange,
    CausalLMDataConfig,
    CausalLMLossConfig,
    DataConfig,
    LossConfig,
    RecordRange,
)


def has_sequence_length(example, seq_len):
    return len(example["input_ids"]) == seq_len


def fits_bridge_sequence(example, seq_len):
    return len(example["input_ids"]) + 1 <= seq_len


def matches_bridge_selection(example, conditions, query_variants):
    return (
        (conditions is None or example["condition"] in conditions)
        and (query_variants is None or example["query_variant"] in query_variants)
    )


def in_record_range(example, index_field, start, end):
    index = example[index_field]
    return index >= start and (end is None or index < end)


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


class BridgeMemoryCollator:
    def __init__(self, all_tokens_weight: float, delayed_answer_weight: float):
        self.all_tokens_weight = all_tokens_weight
        self.delayed_answer_weight = delayed_answer_weight

    def __call__(self, examples) -> dict[str, torch.Tensor | list[str]]:
        if len(examples) != 1:
            raise ValueError("Variable-length bridge-memory training requires batch_size=1")
        example = examples[0]
        target = int(example["target_token_id"])
        input_ids = torch.tensor(
            [list(example["input_ids"]) + [target]], dtype=torch.long,
        )
        batch: dict[str, torch.Tensor | list[str]] = {
            "input_ids": input_ids,
            "target_token_ids": torch.tensor([target], dtype=torch.long),
            "candidate_token_ids": torch.tensor(
                [example["candidate_token_ids"]], dtype=torch.long,
            ),
            "conditions": [example["condition"]],
            "query_variants": [example["query_variant"]],
        }
        if self.all_tokens_weight > 0:
            batch["labels"] = input_ids.clone()
        if self.delayed_answer_weight > 0:
            delayed_labels = torch.full_like(input_ids, -100)
            delayed_labels[:, -1] = target
            batch["delayed_labels"] = delayed_labels
        return batch


@overload
def create_dataloader(
    config: BridgeMemoryDataConfig,
    records: BridgeMemoryRecordRange,
    loss: BridgeMemoryLossConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader: ...


@overload
def create_dataloader(
    config: CausalLMDataConfig,
    records: RecordRange,
    loss: CausalLMLossConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader: ...


def create_dataloader(
    config: DataConfig,
    records: RecordRange,
    loss: LossConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Stream independent causal-LM or bridge-memory training records."""
    batch_size = config.batch_size
    seq_len = config.seq_len
    num_workers = config.num_workers
    for name, value, minimum in (
        ("batch_size", batch_size, 1), ("seq_len", seq_len, 1), ("num_workers", num_workers, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    start, end = records.start, records.end
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Expected a nonempty half-open record range [start, end)")
    if isinstance(config, CausalLMDataConfig):
        if not isinstance(loss, CausalLMLossConfig) or type(records) is not RecordRange:
            raise TypeError("CausalLMDataConfig requires causal-LM loss and record-range types")
    elif isinstance(config, BridgeMemoryDataConfig):
        if not isinstance(loss, BridgeMemoryLossConfig) or not isinstance(records, BridgeMemoryRecordRange):
            raise TypeError("BridgeMemoryDataConfig requires bridge-memory loss and record-range types")
        if batch_size != 1:
            raise ValueError("Variable-length bridge_memory records require batch_size=1")
    else:
        raise TypeError(f"Unsupported data config: {type(config).__name__}")

    load_options = {}
    if config.revision is not None:
        load_options["revision"] = config.revision
    # Parquet predicates skip unrelated row groups before streaming. ProLong
    # ranges address source records; bridge ranges address fact IDs.
    index_field = "source_record_index" if isinstance(config, CausalLMDataConfig) else "fact_id"
    filters = []
    if start:
        filters.append((index_field, ">=", start))
    if end is not None:
        filters.append((index_field, "<", end))
    if filters:
        load_options["filters"] = filters
    load_args = [config.dataset_id]
    if isinstance(config, BridgeMemoryDataConfig):
        load_args.append(config.dataset_config)
    dataset = load_dataset(*load_args, split=records.split, streaming=True, **load_options)
    if filters:
        # Keep the semantic guard even when a custom dataset builder ignores
        # Parquet pushdown; normally the predicate was already applied cheaply.
        dataset = dataset.filter(partial(
            in_record_range, index_field=index_field, start=start, end=end,
        ))
    if isinstance(config, CausalLMDataConfig):
        dataset = dataset.filter(partial(has_sequence_length, seq_len=seq_len))
        dataset = dataset.select_columns(["input_ids"])
        collator = CausalLMCollator()
    else:
        dataset = dataset.filter(partial(fits_bridge_sequence, seq_len=seq_len))
        dataset = dataset.filter(partial(
            matches_bridge_selection,
            conditions=records.conditions,
            query_variants=records.query_variants,
        ))
        dataset = dataset.select_columns([
            "input_ids", "target_token_id", "candidate_token_ids",
            "condition", "query_variant",
        ])
        collator = BridgeMemoryCollator(
            all_tokens_weight=loss.all_tokens_weight,
            delayed_answer_weight=loss.delayed_answer_weight,
        )
    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=config.shuffle_buffer_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
