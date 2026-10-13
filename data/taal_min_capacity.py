"""Paired post-eviction capacity datasets, not generalization tests."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION, build_examples


def build_capacity_examples(tokenizer, *, pairs: int = 1, gap: int = 64) -> list[dict]:
    if type(pairs) is not int or pairs < 1:
        raise ValueError("Capacity datasets require a positive number of record pairs")
    # Keep the attention window fixed while varying distance or task diversity.
    # The shared generator requires two pairs, so select one complete pair for
    # the minimum case and rebuild its candidate list from the selected rows.
    examples = build_examples(tokenizer, pairs=max(2, pairs), window=32, gap=gap)
    examples = examples[:2 * pairs]
    candidates = [example["target_token_id"] for example in examples]
    for example in examples:
        example["candidate_token_ids"] = candidates.copy()
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/taal_min_capacity.jsonl"))
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--gap", type=int, default=64)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = build_capacity_examples(tokenizer, pairs=args.pairs, gap=args.gap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "pairs": args.pairs, "examples": len(examples), "working_memory_size": 32,
        "fact_to_answer_gap": args.gap, "fact_visible_in_kv": False,
        "prompt_tokens": sorted({len(row["input_ids"]) for row in examples}),
        "comparison": "Within each pair, record/question/filler match; only the source label differs.",
    }, indent=2) + "\n")
    print(f"Prepared {len(examples)} capacity records: {args.output}")


if __name__ == "__main__":
    main()
