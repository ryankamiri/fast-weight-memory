"""Offline preparation: python -m scripts.prepare_longmemeval [--upload]."""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from evaluation.data import FILES, PREFIX, PROMPT_VERSION, SOURCE, SOURCE_REVISION, TOKENIZER, prepare_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=FILES, default="oracle")
    parser.add_argument("--repo-id", default="ryankamiri/longmemeval-qwen")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    api = HfApi()
    revision = api.model_info(TOKENIZER).sha
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=revision)
    source = hf_hub_download(SOURCE, FILES[args.variant], repo_type="dataset", revision=SOURCE_REVISION)
    with open(source) as file:
        examples = json.load(file)
    rows = []
    for index, example in enumerate(examples):
        rows.append(prepare_example(example, tokenizer))
        if (index + 1) % 25 == 0 or index + 1 == len(examples):
            print(f"Tokenized {index + 1}/{len(examples)}", flush=True)
    if len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    metadata = {
        "schema_version": 3,  # Original source columns plus a single input_ids sequence.
        "source": SOURCE, "source_revision": SOURCE_REVISION,
        "tokenizer": TOKENIZER, "tokenizer_revision": revision,
        "variant": args.variant, "prompt_version": PROMPT_VERSION, "prefix": PREFIX,
        "records": len(rows), "max_prompt_length": max(row["prompt_length"] for row in rows),
    }
    output = Path("datasets/longmemeval-qwen") / args.variant
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(output / "test.parquet")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.upload:
        print(f"Uploading to https://huggingface.co/datasets/{args.repo_id}", flush=True)
        dataset.push_to_hub(args.repo_id, config_name=args.variant, split="test")
        api.upload_file(
            path_or_fileobj=output / "metadata.json", path_in_repo=f"{args.variant}/metadata.json",
            repo_id=args.repo_id, repo_type="dataset",
        )
        print(f"Ready: https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
