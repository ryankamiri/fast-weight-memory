"""Paired equal-token-length neutral and competing-record filler."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_state_probe import build_probe_examples
from evaluation.taal.lifecycle import lifecycle_boundaries


def build_interference_examples(tokenizer, *, seed=73):
    base = build_probe_examples(tokenizer, seed=seed)
    labels = sorted({(row["answer"], row["target_token_id"]) for row in base})
    rng = random.Random(seed + 1000)
    mappings = {}
    for family in sorted({row["context_family"] for row in base}):
        mapping = list(range(len(labels)))
        rng.shuffle(mapping)
        while any(i == value for i, value in enumerate(mapping)):
            rng.shuffle(mapping)
        mappings[family] = mapping
    rows = []
    for original in base:
        boundary = lifecycle_boundaries(original, tokenizer)
        target_index = labels.index((original["answer"], original["target_token_id"]))
        word, target = labels[mappings[original["context_family"]][target_index]]
        competitor = tokenizer.encode("User: Remember that Record R9999's archive label is", add_special_tokens=False)
        competitor += [target] + tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
        stop = boundary["fact"] + len(competitor)
        if stop > boundary["final"] - 128:
            raise ValueError("Distractor must be evicted before the question")
        for variant in ("neutral", "competing"):
            ids = list(original["input_ids"])
            if variant == "competing":
                ids[boundary["fact"]:stop] = competitor
            rows.append({**original, "example_id": f"{original['example_id']}-{variant}",
                "input_ids": ids, "filler_variant": variant, "distractor_token_id": target,
                "distractor_answer": word, "distractor_end": stop,
                "distractor_record": "Record R9999", "paired_example_id": original["example_id"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    rows = build_interference_examples(tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "working_memory_size": 128, "fact_to_answer_gap": 192,
        "comparison": "Paired neutral/competing filler, equal length and recipient query; both facts evicted.",
        "distractor_mapping": "Family-specific balanced derangement; not exhaustive label-pair counterbalance.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
