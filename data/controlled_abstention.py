"""Build counterbalanced present/absent queries for controlled retrieval evaluation."""

import argparse
import random
from pathlib import Path

from evaluation.storage import append_result, atomic_json, file_sha256

ABSTENTION_RESPONSE = "UNKNOWN"
GENERATOR_VERSION = "controlled-abstention-v1"


def build_examples(
    pairs: int = 32, fact_count: int = 2, filler_repetitions: int = 64, seed: int = 42
) -> list[dict]:
    if pairs < 1 or fact_count < 1 or filler_repetitions < 0:
        raise ValueError(
            "pairs/fact_count must be positive; filler_repetitions must be nonnegative"
        )

    rng = random.Random(seed)
    rows = []
    instruction = (
        "Answer using only the records in this context. Return only the archive label. "
        f"If the requested record is not in the context, return exactly {ABSTENTION_RESPONSE}.\n\n"
    )
    filler = (
        "The neutral passage continues without additional record information.\n"
        * filler_repetitions
    )
    for pair in range(pairs):
        identifiers = rng.sample(range(10_000_000), 4 * fact_count)
        keys = []
        labels = []
        for index in range(2 * fact_count):
            keys.append(f"R{identifiers[index]:07d}")
            labels.append(f"V{identifiers[2 * fact_count + index]:07d}")

        # Each queried key is present in one context and absent in the other.
        query_indices = [0, fact_count]
        for variant in range(2):
            start = variant * fact_count
            facts = []
            for index in range(start, start + fact_count):
                facts.append(
                    f"Record {keys[index]}'s archive label is {labels[index]}.\n"
                )
            rng.shuffle(facts)
            prefix = instruction + "".join(facts) + "\n" + filler
            group = f"abstention-{pair}-context-{variant}"
            for query_index in query_indices:
                answerable = start <= query_index < start + fact_count
                answer = ABSTENTION_RESPONSE
                if answerable:
                    answer = labels[query_index]
                query = f"What is Record {keys[query_index]}'s archive label?\nAnswer:"
                rows.append(
                    {
                        "example_id": f"{group}-query-{query_index}",
                        "prompt": prefix + query,
                        "answer": answer,
                        "group_id": group,
                        "metadata": {
                            "task": "controlled_abstention",
                            "answerable": answerable,
                            "abstention_response": ABSTENTION_RESPONSE,
                            "fact_count": fact_count,
                            "pair_id": pair,
                            "query_key": keys[query_index],
                            "filler_repetitions": filler_repetitions,
                        },
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/controlled_abstention/evaluation.jsonl"),
    )
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--fact-count", type=int, default=2)
    parser.add_argument("--filler-repetitions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Use a new output path rather than overwriting an evaluation set"
        )

    rows = build_examples(
        args.pairs, args.fact_count, args.filler_repetitions, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        append_result(args.output, row)
    atomic_json(
        args.output.with_suffix(".metadata.json"),
        {
            "generator_version": GENERATOR_VERSION,
            "seed": args.seed,
            "pairs": args.pairs,
            "records": len(rows),
            "fact_count": args.fact_count,
            "filler_repetitions": args.filler_repetitions,
            "sha256": file_sha256(args.output),
            "split_unit": "pair_id; keep both context variants and every query together",
        },
    )
    print(f"Wrote {len(rows)} evaluation examples to {args.output}")


if __name__ == "__main__":
    main()
