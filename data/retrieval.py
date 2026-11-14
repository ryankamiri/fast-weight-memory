"""Manifest-checked retrieval splits and deterministic, resumable episode order."""

import json
from pathlib import Path
import random

from torch.utils.data import IterableDataset

from evaluation.storage import file_sha256


def load_retrieval_splits(config):
    manifest_path = Path(config.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("generator_version") != "balanced-retrieval-v1":
        raise ValueError("Expected a balanced-retrieval-v1 manifest")

    if set(config.split_files) != {"train", "val"}:
        raise ValueError("Training loads only explicit train and val files; test stays locked")

    splits = {}
    seen_families, seen_keys, seen_ids = set(), set(), set()
    for split, filename in config.split_files.items():
        path = Path(filename)
        expected = manifest["files"][split]
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {split}")

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != expected["rows"] or not rows:
            raise ValueError(f"Dataset row count mismatch: {split}")

        families, keys = set(), set()
        for row in rows:
            metadata = row["metadata"]
            if metadata["split"] != split or row["example_id"] in seen_ids:
                raise ValueError("Wrong split or duplicate example identity")

            seen_ids.add(row["example_id"])
            families.add(metadata["family_id"])
            keys.update(metadata["bindings"])
            ids, labels, prompt = row["input_ids"], row["labels"], row["prompt_input_ids"]
            if not 2 <= len(ids) <= config.seq_len:
                raise ValueError("Retrieval episode exceeds sequence budget; do not silently filter")

            if len(labels) != len(ids) or ids[:len(prompt)] != prompt:
                raise ValueError("Retrieval prompt/input/label lengths disagree")

            if labels[:len(prompt)] != [-100] * len(prompt) or labels[len(prompt):] != ids[len(prompt):]:
                raise ValueError("Retrieval requires answer-only labels")

            if len(prompt) >= len(ids) or row["answer"] != metadata["bindings"][metadata["query_key"]]:
                raise ValueError("Missing answer supervision or incorrect binding")

        if families & seen_families or keys & seen_keys:
            raise ValueError("Train/val context families or record keys overlap")

        seen_families.update(families)
        seen_keys.update(keys)
        splits[split] = rows
    return splits, manifest


class RetrievalDataset(IterableDataset):
    def __init__(self, rows, seed, shuffle, identity):
        self.rows = rows
        self.seed = seed
        self.shuffle = shuffle
        self.identity = identity
        self.epoch = 0
        self.cursor = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.cursor = 0

    def __iter__(self):
        order = list(range(len(self.rows)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        while self.cursor < len(order):
            index = order[self.cursor]
            self.cursor += 1
            yield self.rows[index]

    def state_dict(self):
        return {
            "epoch": self.epoch, "cursor": self.cursor,
            "identity": self.identity, "seed": self.seed, "shuffle": self.shuffle,
        }

    def load_state_dict(self, state):
        for name in ("identity", "seed", "shuffle"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Resume changes data {name}")

        if type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("Invalid resume epoch")

        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(self.rows):
            raise ValueError("Invalid resume data cursor")

        self.epoch, self.cursor = state["epoch"], state["cursor"]
