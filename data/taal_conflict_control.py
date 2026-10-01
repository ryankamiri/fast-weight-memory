"""Generate paired episodes that differ only in an evicted fact value."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from evaluation.bridge_data import single_token_labels


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def build_examples(tokenizer, *, pairs: int, window: int, gap: int) -> list[dict]:
    if pairs < 2 or window < 32 or gap <= window:
        raise ValueError("Need at least two pairs and a fact-to-answer gap beyond KV")
    labels = single_token_labels(tokenizer, pairs * 2)
    candidates = [token_id for _, token_id in labels]
    filler_seed = tokenizer.encode(
        " A neutral passage continues without mentioning any record or label.",
        add_special_tokens=False,
    )
    if not filler_seed:
        raise ValueError("Tokenizer produced empty filler")
    examples = []
    for pair in range(pairs):
        record = f"Record R{pair:04d}"
        fact_start = tokenizer.encode(
            f"User: Remember that {record}'s archive label is",
            add_special_tokens=False,
        )
        fact_end = tokenizer.encode(
            ".\nAssistant: Understood.\n\n", add_special_tokens=False,
        )
        query = tokenizer.encode(
            f"User: What is {record}'s archive label?\n"
            "Assistant: The archive label is",
            add_special_tokens=False,
        )
        fact_position = len(fact_start)
        filler_length = gap - 1 - len(fact_end) - len(query)
        if filler_length < 1:
            raise ValueError("Gap leaves no room for filler")
        filler = (filler_seed * (filler_length // len(filler_seed) + 1))[:filler_length]
        for variant in range(2):
            word, answer = labels[2 * pair + variant]
            fact = fact_start + [answer] + fact_end
            prefix = fact + filler
            input_ids = prefix + query
            assert len(input_ids) - fact_position == gap
            examples.append({
                "example_id": f"conflict-{pair:02d}-{variant}",
                "fact_id": 2 * pair + variant,
                "record_name": record,
                "answer": word,
                "condition": "micro_conflict",
                "query_variant": "exact",
                "input_ids": input_ids,
                "target_token_id": answer,
                "candidate_token_ids": candidates,
                "fact_position": fact_position,
                "final_query_position": len(prefix),
                "final_answer_position": len(input_ids),
                "prompt_length": len(input_ids),
            })
    return examples


def write_dataset(output: Path, *, pairs: int, window: int, gap: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = build_examples(tokenizer, pairs=pairs, window=window, gap=gap)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "tokenizer": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "pairs": pairs,
        "working_memory_size": window,
        "fact_to_answer_gap": gap,
        "comparison": "Within each pair, only the evicted answer token differs.",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--gap", type=int, default=192)
    args = parser.parse_args()
    write_dataset(args.output, pairs=args.pairs, window=args.window, gap=args.gap)


if __name__ == "__main__":
    main()
