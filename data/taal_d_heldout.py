"""Isolated held-out binding, context and question tests for the fixed D checkpoint."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_min_capacity import build_capacity_examples


EXPERIMENTS = ("bindings", "context", "questions")
FILLERS = {
    "weather": " Outside, the sky was clear. A soft wind moved across the hills. Nothing unusual happened that afternoon.",
    "room": " Inside, the room was quiet. Someone adjusted a chair and then continued working. The clock kept ticking.",
}
QUESTIONS = {
    "recall": "Please recall {record}'s archive label.",
    "give": "Give me {record}'s archive label.",
    "belongs": "Which archive label belongs to {record}?",
}
D_CHECKPOINT_SHA256 = "285dab0ad8f638d95caefb283aeb67598ca959105974e57453a8e925a4359100"
D_DATASET_SHA256 = "685041cd94587585da055371c0460576637f0d23f4169b6b53dcfcf4071e4d93"


def repeat_tokens(tokens: list[int], length: int) -> list[int]:
    if not tokens or length < 0:
        raise ValueError("Filler requires nonempty tokens and a nonnegative length")

    return (tokens * (length // len(tokens) + 1))[:length]


def tagged_example(original: dict, arm: str, rows: list[dict]) -> dict:
    row = copy.deepcopy(original)
    row.update({
        "example_id": f"d-{arm}-{original['example_id']}",
        "source_example_id": original["example_id"],
        "fact_id": len(rows),
        "condition": arm,
        "pair_group": f"{arm}/{original['record_name']}",
        "evaluation_split": "reference" if arm == "reference" else "heldout",
        "fact_start_position": 0,
    })
    rows.append(row)
    return row


def build_heldout_examples(tokenizer, experiment: str) -> list[dict]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown held-out experiment: {experiment}")

    base = build_capacity_examples(tokenizer, pairs=8, gap=192)
    rows = []
    for original in base:
        tagged_example(original, "reference", rows)
    if experiment == "bindings":
        for shift in range(1, 8):
            for index, original in enumerate(base):
                row = tagged_example(original, f"binding_shift_{shift}", rows)
                replacement = base[(index + 2 * shift) % len(base)]
                row["answer"] = replacement["answer"]
                row["target_token_id"] = replacement["target_token_id"]
                row["input_ids"][row["fact_position"]] = row["target_token_id"]
    elif experiment == "context":
        fact_end = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
        for name, text in FILLERS.items():
            filler = tokenizer.encode(text, add_special_tokens=False)
            for original in base:
                row = tagged_example(original, f"filler_{name}", rows)
                start = row["fact_position"] + 1 + len(fact_end)
                end = row["final_query_position"]
                row["input_ids"][start:end] = repeat_tokens(filler, end - start)
        neutral = tokenizer.encode(
            " A neutral passage continues without mentioning any record or label.",
            add_special_tokens=False,
        )
        separator = tokenizer.encode("\n\n", add_special_tokens=False)
        for offset in (32, 64):
            prefix = repeat_tokens(neutral, offset - len(separator)) + separator
            for original in base:
                row = tagged_example(original, f"position_{offset}", rows)
                row["input_ids"] = prefix + row["input_ids"]
                row["fact_start_position"] = offset
                for field in ("fact_position", "final_query_position", "final_answer_position", "prompt_length"):
                    row[field] += offset
    else:
        for name, template in QUESTIONS.items():
            for original in base:
                row = tagged_example(original, f"question_{name}", rows)
                text = "User: " + template.format(record=row["record_name"])
                text += "\nAssistant: The archive label is"
                query = tokenizer.encode(text, add_special_tokens=False)
                start = row["final_query_position"]
                if len(query) != len(row["input_ids"]) - start:
                    raise ValueError("Paraphrase must preserve query length without changing filler")

                row["input_ids"][start:] = query
                row["query_variant"] = name
    return rows


def dataset_text(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def dataset_metadata(rows: list[dict], experiment: str) -> dict:
    arms = sorted({row["condition"] for row in rows})
    return {
        "tokenizer": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "experiment": experiment,
        "dataset_sha256": hashlib.sha256(dataset_text(rows).encode()).hexdigest(),
        "parent_checkpoint_sha256": D_CHECKPOINT_SHA256,
        "training_dataset_sha256": D_DATASET_SHA256,
        "working_memory_size": 32,
        "memory_chunk_size": 1,
        "fact_to_answer_gap": 192,
        "arms": {arm: sum(row["condition"] == arm for row in rows) for arm in arms},
        "reference_examples": 16,
        "heldout_examples": len(rows) - 16,
        "position_caveat": "Position arms prepend familiar filler, changing pre-fact history and total length while preserving the original fact-to-answer sequence.",
        "inference_only": True,
        "comparison": "Matched opposing labels within each arm and record; identical candidate set, no checkpoint fitting or selection on these examples.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    rows = build_heldout_examples(tokenizer, args.experiment)
    payload = dataset_text(rows)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata = json.dumps(dataset_metadata(rows, args.experiment), indent=2) + "\n"
    for path, content in ((args.output, payload), (metadata_path, metadata)):
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite a different dataset: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    metadata_path.write_text(metadata)
    print(json.dumps(dataset_metadata(rows, args.experiment)), flush=True)


if __name__ == "__main__":
    main()
