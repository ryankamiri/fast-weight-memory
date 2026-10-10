"""Counterbalanced, family-held-out data for decoding labels from memory state."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION, build_examples


def build_probe_examples(tokenizer, *, seed: int = 73) -> list[dict]:
    base = build_examples(tokenizer, pairs=8, window=128, gap=192)
    rng = random.Random(seed)
    labels = [(row["answer"], row["target_token_id"]) for row in base]
    rows = []
    mappings = set()
    subjects = ("weather", "a journey", "the landscape", "a quiet room")
    details = ("clouds passed overhead", "the road was empty", "the day was calm", "a clock ticked")
    for family in range(12):
        mapping = list(range(len(labels)))
        rng.shuffle(mapping)
        while tuple(mapping) in mappings:
            rng.shuffle(mapping)
        mappings.add(tuple(mapping))
        split = "train" if family < 8 else "val" if family < 10 else "test"
        # Context families and label-to-record mappings never cross a split.
        sentences = [
            f" A neutral passage about {rng.choice(subjects)} says that {rng.choice(details)}."
            for _ in range(4)
        ]
        filler_seed = tokenizer.encode("".join(sentences), add_special_tokens=False)
        if not filler_seed:
            raise ValueError("Empty probe filler")
        for index, original in enumerate(base):
            row = dict(original)
            ids = list(original["input_ids"])
            word, target = labels[mapping[index]]
            ids[row["fact_position"]] = target
            fact_end = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
            filler_start = row["fact_position"] + 1 + len(fact_end)
            filler_length = row["final_query_position"] - filler_start
            ids[filler_start:row["final_query_position"]] = (
                filler_seed * (filler_length // len(filler_seed) + 1)
            )[:filler_length]
            rows.append({
                **row, "input_ids": ids, "answer": word, "target_token_id": target,
                "example_id": f"probe-family-{family:02d}-{index:02d}",
                "fact_id": len(rows), "context_family": family, "probe_split": split,
                "pair_group": f"family-{family:02d}/{row['record_name']}",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rows = build_probe_examples(tokenizer, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION, "seed": args.seed,
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "families": 12, "split_families": {"train": 8, "val": 2, "test": 2},
        "working_memory_size": 128, "fact_to_answer_gap": 192,
        "comparison": "Paired label alternatives, balanced labels in every family; disjoint family splits.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
