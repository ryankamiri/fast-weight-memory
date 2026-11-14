"""Generate split-safe associative retrieval with tokenizer-verified source eviction."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import yaml
from transformers import AutoTokenizer

from evaluation.storage import append_result, atomic_json, file_sha256

VERSION = "balanced-retrieval-v1"
SPLITS = ("train", "val", "test")
FILLER = "The neutral passage continues without additional record information.\n"
LABELS = (
    "amber", "anchor", "arrow", "atlas", "birch", "bronze", "cedar", "coral",
    "delta", "elm", "fern", "flint", "grove", "harbor", "indigo", "ivory",
)


@dataclass(frozen=True)
class RetrievalConfig:
    seed: int = 42
    label_pool: tuple[str, ...] = LABELS
    train_families: int = 256
    val_families: int = 64
    test_families: int = 128
    fact_counts: tuple[int, ...] = (2,)
    kv_window: int = 128
    min_gap_tokens: int = 192
    max_sequence_tokens: int = 1024

    def validate(self):
        for value in (self.seed, self.kv_window, self.min_gap_tokens, self.max_sequence_tokens):
            if type(value) is not int:
                raise ValueError("Seed and token budgets must be integers")

        if not self.label_pool or len(set(self.label_pool)) != len(self.label_pool):
            raise ValueError("label_pool must contain unique labels")

        for label in self.label_pool:
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError("Labels must be nonempty strings without surrounding whitespace")

            if "\n" in label or "\r" in label:
                raise ValueError("Labels cannot inject additional record lines")

        for count in (self.train_families, self.val_families, self.test_families):
            if type(count) is not int or count < 1:
                raise ValueError("Each split requires a positive integer family count")

        if not self.fact_counts or len(set(self.fact_counts)) != len(self.fact_counts):
            raise ValueError("fact_counts must be nonempty and unique")

        for count in self.fact_counts:
            if type(count) is not int or count not in (2, 4, 8):
                raise ValueError("Supported fact counts are 2, 4 and 8")

        if max(self.fact_counts) > len(self.label_pool):
            raise ValueError("label_pool must cover the largest fact count")

        if self.kv_window < 1 or self.min_gap_tokens < self.kv_window:
            raise ValueError("Gap must evict every fact before the first query token")

        if self.max_sequence_tokens <= self.min_gap_tokens:
            raise ValueError("Sequence budget must exceed the requested gap")


def token_span(offsets, start: int, end: int) -> list[int]:
    indices = []
    for index, (left, right) in enumerate(offsets):
        if right > start and left < end:
            indices.append(index)
    if not indices:
        raise ValueError("Tokenizer did not represent the requested text span")

    return [indices[0], indices[-1] + 1]


def render_example(tokenizer, config, family_id, keys, values, order, assignment, query_index):
    instruction = "Return only the archive label for the requested record.\n\n"
    prefix = instruction
    fact_char_spans = []
    for index in order:
        start = len(prefix)
        prefix += f"Record {keys[index]}'s archive label is {values[index]}.\n"
        fact_char_spans.append([start, len(prefix)])
    query = f"\nWhat is Record {keys[query_index]}'s archive label?\nAnswer:"
    answer = values[query_index]
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids or tokenizer.eos_token_id is None:
        raise ValueError("Training requires a nonempty answer and an EOS token")

    # Re-tokenize the exact complete prompt; boundary merges must not fake eviction.
    for repetitions in range(config.max_sequence_tokens + 1):
        context = prefix + FILLER * repetitions
        prompt = context + query
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = list(encoded["input_ids"])
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in prompt_ids:
            raise ValueError("Unknown tokens would invalidate the record bindings")

        offsets = encoded["offset_mapping"]
        spans = []
        for start, end in fact_char_spans:
            spans.append(token_span(offsets, start, end))
        query_start = token_span(offsets, len(context), len(prompt))[0]
        gap = query_start - spans[-1][1]
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        if len(full_ids) > config.max_sequence_tokens:
            raise ValueError("Cannot meet the verified gap within max_sequence_tokens")

        if gap >= config.min_gap_tokens:
            break
    else:
        raise ValueError("Filler did not produce the required token gap")

    group_id = f"{family_id}-assignment{assignment}-order{order[0]}"
    return {
        "example_id": f"{group_id}-query{query_index}",
        "group_id": group_id,
        "prompt": prompt,
        "answer": answer,
        "prompt_input_ids": prompt_ids,
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id],
        "metadata": {
            "family_id": family_id,
            "fact_count": len(keys),
            "query_key": keys[query_index],
            "queried_fact_order": order.index(query_index),
            "bindings": dict(zip(keys, values)),
            "fact_order": order,
            "assignment": assignment,
            "fact_token_spans": spans,
            "query_start_token": query_start,
            "prefix_text": context,
            "verified_gap_tokens": gap,
            "kv_window": config.kv_window,
            "filler_repetitions": repetitions,
            "answer_token_count": len(answer_ids),
        },
    }


def build_bundle(tokenizer, config: RetrievalConfig) -> dict[str, list[dict]]:
    config.validate()
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with offsets is required")

    # Distinct labels must remain distinct to the tokenizer, including multi-token labels.
    encoded_labels = set()
    for label in config.label_pool:
        ids = tuple(tokenizer.encode(" " + label, add_special_tokens=False))
        if not ids or ids in encoded_labels:
            raise ValueError("Label pool contains empty or tokenizer-equivalent labels")

        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
            raise ValueError("Unknown label tokens cannot define a retrieval answer")

        encoded_labels.add(ids)
    bundle = {}
    used_keys = set()
    counts = (config.train_families, config.val_families, config.test_families)
    for split, families in zip(SPLITS, counts):
        rows = []
        for family in range(families):
            family_id = f"{split}-family{family:06d}"
            seed = hashlib.sha256(f"{config.seed}/{family_id}".encode()).hexdigest()
            rng = random.Random(seed)
            count = config.fact_counts[family % len(config.fact_counts)]
            keys = []
            for index in range(count):
                key = f"R{rng.getrandbits(64):016x}"
                if key in used_keys:
                    raise ValueError("Duplicate record identifier across context families")

                keys.append(key)
                used_keys.add(key)
            values = rng.sample(list(config.label_pool), count)
            order = list(range(count))
            rng.shuffle(order)
            orders = (order, list(reversed(order)))
            # Cyclic remapping balances each label across every key, within one split.
            for assignment in range(count):
                assigned_values = values[assignment:] + values[:assignment]
                for fact_order in orders:
                    for query_index in range(count):
                        row = render_example(
                            tokenizer, config, family_id, keys, assigned_values,
                            fact_order, assignment, query_index,
                        )
                        row["metadata"]["split"] = split
                        rows.append(row)
        bundle[split] = rows
    return bundle


def export_bundle(tokenizer, config: RetrievalConfig, destination: Path):
    if destination.exists():
        raise ValueError("Use a new destination; never overwrite a locked split")

    bundle = build_bundle(tokenizer, config)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "generator_version": VERSION,
        "generator_sha256": file_sha256(Path(__file__)),
        "config": asdict(config),
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "special_tokens": tokenizer.special_tokens_map,
        "split_unit": "family_id; all queries, orders and remappings stay together",
        "filler": "fixed synthetic neutral passage; no natural documents",
        "gap_definition": "query_start_token - last_fact_exclusive_end",
        "training": "prompt IDs plus separately encoded space+answer and EOS; answer-only labels",
        "files": {},
    }
    for split, rows in bundle.items():
        path = destination / f"{split}.jsonl"
        for row in rows:
            append_result(path, row)
        manifest["files"][split] = {"rows": len(rows), "sha256": file_sha256(path)}
    atomic_json(destination / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local saved fast-tokenizer directory")
    parser.add_argument("--config", type=Path, required=True, help="YAML RetrievalConfig fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = RetrievalConfig(**yaml.safe_load(args.config.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    export_bundle(tokenizer, config, args.output)


if __name__ == "__main__":
    main()
