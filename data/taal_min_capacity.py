"""One opposing-label pair for minimum-capacity training, not generalization."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION, build_examples


def build_capacity_examples(tokenizer) -> list[dict]:
    examples = build_examples(tokenizer, pairs=2, window=32, gap=64)[:2]
    candidates = [example["target_token_id"] for example in examples]
    for example in examples:
        example["candidate_token_ids"] = candidates.copy()
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/taal_min_capacity.jsonl"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = build_capacity_examples(tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "pairs": 1, "examples": len(examples), "working_memory_size": 32,
        "fact_to_answer_gap": 64, "fact_visible_in_kv": False,
        "comparison": "Same record/question/filler. Only the source label differs.",
    }, indent=2) + "\n")
    print(f"Prepared {len(examples)} minimum-capacity records: {args.output}")


if __name__ == "__main__":
    main()
