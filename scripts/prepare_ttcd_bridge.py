import argparse
import json
from pathlib import Path
import random
import shutil

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer

from evaluation.bridge_data import BridgeGeometry, build_bridge_example, single_token_labels


EVALUATION_TOKENIZER = "Qwen/Qwen3-0.6B-Base"
FILLER_DATASET = "ryankamiri/prolong-qwen"
FILLER_SOURCE_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
OUTPUT_ROOT = Path("datasets/ttcd-bridge-memory")
NUM_FACTS = 128
NUM_CANDIDATES = 16
SEED = 42
SCHEMA_VERSION = 1


def candidate_ids(codes, index):
    target = codes[index][1]
    decoys = [codes[(index + offset) % len(codes)][1] for offset in range(1, NUM_CANDIDATES)]
    values = [target, *decoys]
    random.Random(SEED + index).shuffle(values)
    return values


def build_fact_rows(
    tokenizer, source_ids, codes, fact_index, geometry, source_record_index=None,
):
    answer, target = codes[fact_index]
    candidates = candidate_ids(codes, fact_index)
    conditions = [
        ("visible", "exact"),
        ("no_bridge", "exact"),
        ("no_bridge", "paraphrased"),
        ("bridge", "exact"),
        ("bridge", "paraphrased"),
    ]

    # Search within the source record for a long filler span that does not
    # accidentally contain this example's answer token.
    stride = geometry.teacher_window_size * 3
    for start in range(0, len(source_ids), stride):
        filler = source_ids[start:]
        try:
            rows = [
                build_bridge_example(
                    tokenizer,
                    filler,
                    fact_index=fact_index,
                    answer_text=answer,
                    target_token_id=target,
                    candidate_token_ids=candidates,
                    condition=condition,
                    query_variant=query_variant,
                    geometry=geometry,
                )
                for condition, query_variant in conditions
            ]
        except ValueError:
            continue
        expected = {"visible": 1, "no_bridge": 1, "bridge": 2}
        if all(row["input_ids"].count(target) == expected[row["condition"]] for row in rows):
            for row in rows:
                row["filler_source_record"] = (
                    fact_index if source_record_index is None else source_record_index
                )
                row["filler_source_offset"] = start
            return rows
    raise ValueError(f"Source record {fact_index} has no usable filler span")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ryankamiri/ttcd-bridge-memory")
    parser.add_argument("--teacher-window-size", type=int, default=4096)
    parser.add_argument("--student-window-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    validate_repo_id(args.repo_id)
    if "/" not in args.repo_id:
        parser.error("--repo-id must include the namespace: username/dataset-name")

    geometry = BridgeGeometry(
        teacher_window_size=args.teacher_window_size,
        student_window_size=args.student_window_size,
        chunk_size=args.chunk_size,
    )
    geometry.validate()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    api = HfApi()
    api.whoami()
    tokenizer_revision = api.model_info(EVALUATION_TOKENIZER).sha
    filler_tokenizer_revision = api.model_info(FILLER_SOURCE_TOKENIZER).sha
    filler_revision = api.dataset_info(FILLER_DATASET).sha
    tokenizer = AutoTokenizer.from_pretrained(
        EVALUATION_TOKENIZER, revision=tokenizer_revision,
    )
    filler_tokenizer = AutoTokenizer.from_pretrained(
        FILLER_SOURCE_TOKENIZER, revision=filler_tokenizer_revision,
    )
    if tokenizer.get_vocab() != filler_tokenizer.get_vocab():
        raise ValueError("Prepared filler token IDs do not use the evaluation tokenizer vocabulary")
    del filler_tokenizer
    codes = single_token_labels(tokenizer, NUM_FACTS)
    source = iter(load_dataset(
        FILLER_DATASET, split="train", revision=filler_revision, streaming=True,
    ))

    rows = []
    source_record_index = 0
    for fact_index in range(NUM_FACTS):
        while True:
            source_row = next(source)
            try:
                fact_rows = build_fact_rows(
                    tokenizer, list(source_row["input_ids"]), codes, fact_index, geometry,
                    source_record_index=source_record_index,
                )
            except ValueError:
                source_record_index += 1
                continue
            source_record_index += 1
            rows.extend(fact_rows)
            break
        if (fact_index + 1) % 16 == 0 or fact_index + 1 == NUM_FACTS:
            print(f"Prepared {fact_index + 1}/{NUM_FACTS} facts", flush=True)

    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")
    lengths = {condition: [] for condition in ("visible", "no_bridge", "bridge")}
    for row in rows:
        lengths[row["condition"]].append(row["prompt_length"])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config_name": geometry.config_name,
        "records": len(rows),
        "facts": NUM_FACTS,
        "candidates_per_fact": NUM_CANDIDATES,
        "seed": SEED,
        "tokenizer": EVALUATION_TOKENIZER,
        "tokenizer_revision": tokenizer_revision,
        "filler_dataset": FILLER_DATASET,
        "filler_dataset_revision": filler_revision,
        "filler_tokenizer": FILLER_SOURCE_TOKENIZER,
        "filler_tokenizer_revision": filler_tokenizer_revision,
        "teacher_window_size": geometry.teacher_window_size,
        "student_window_size": geometry.student_window_size,
        "chunk_size": geometry.chunk_size,
        "bridge_distance": geometry.bridge_distance,
        "eviction_distance": geometry.eviction_distance,
        "maximum_prompt_length": max(row["prompt_length"] for row in rows),
        "conditions": {name: len(values) for name, values in lengths.items()},
    }

    output = OUTPUT_ROOT / geometry.config_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(output / "test.parquet")
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    repo_url = api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading to {repo_url} as config {geometry.config_name}...", flush=True)
    dataset.push_to_hub(
        args.repo_id, config_name=geometry.config_name, split="test", private=True,
    )
    api.upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo=f"{geometry.config_name}/metadata.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"Ready: https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
