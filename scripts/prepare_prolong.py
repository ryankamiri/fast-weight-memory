"""Convert locally, then upload: uv run python -m scripts.prepare_prolong"""

import argparse
from collections import deque
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer

from data.prolong import stream_prolong


DATASET_ID = "princeton-nlp/prolong-data-64K"
SUBSET = "book-65536"
SOURCE_TOKENIZER = "princeton-nlp/Llama-3-8B-ProLong-64k-Base"
TOKENIZER = "Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = Path("datasets/prolong-qwen")
MAX_LENGTH = 65536
RECORDS_PER_SHARD = 1024
ROW_GROUP_SIZE = 8
NUM_WORKERS = min(4, os.cpu_count() or 1)
_tokenizers = None
_tokenizer_error = None


def initialize_worker(source_revision, target_revision):
    global _tokenizers, _tokenizer_error
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tokenizers = None
    _tokenizer_error = None
    try:
        source = AutoTokenizer.from_pretrained(SOURCE_TOKENIZER, revision=source_revision)
        target = AutoTokenizer.from_pretrained(TOKENIZER, revision=target_revision)
        target.model_max_length = sys.maxsize
        _tokenizers = source, target
    except Exception as error:
        # Report through a task instead of making Pool endlessly restart workers.
        _tokenizer_error = str(error)


def convert_task(sample, source_index):
    if _tokenizer_error is not None:
        raise RuntimeError(f"Worker tokenizer initialization failed: {_tokenizer_error}")
    return convert_record(sample, *_tokenizers, MAX_LENGTH, source_index)


def parallel_rows(stream, start_record, source_revision, target_revision):
    """Bound queued work and preserve source order; stop workers on interruption."""
    with mp.get_context("spawn").Pool(
        NUM_WORKERS, initializer=initialize_worker,
        initargs=(source_revision, target_revision),
    ) as pool:
        pending = deque()
        for source_index, sample in enumerate(stream, start=start_record):
            pending.append(pool.apply_async(
                convert_task, (sample, source_index),
            ))
            if len(pending) >= NUM_WORKERS * 2:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()


def load_checkpoint(directory):
    path = directory / "manifest.json"
    if not path.exists():
        if any(directory.glob("train-*.parquet")):
            raise ValueError("Existing shards have no manifest; cannot safely resume")
        return None
    manifest = json.loads(path.read_text())
    expected = {
        "dataset_id": DATASET_ID, "subset": SUBSET,
        "source_tokenizer": SOURCE_TOKENIZER, "tokenizer": TOKENIZER,
        "max_length": MAX_LENGTH, "records_per_shard": RECORDS_PER_SHARD,
        "row_group_size": ROW_GROUP_SIZE, "compression": "zstd",
        "skip_source_special_tokens": True, "clean_up_tokenization_spaces": False,
        "add_target_special_tokens": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cannot resume: {key} differs from the saved conversion")
    records = 0
    for index, shard in enumerate(manifest["shards"]):
        if shard["file"] != f"train-{index:05d}.parquet":
            raise ValueError("Checkpoint shards are not contiguous")
        metadata = pq.read_metadata(directory / shard["file"])
        if metadata.num_rows != shard["records"]:
            raise ValueError(f"Checkpoint row count mismatch: {shard['file']}")
        records += shard["records"]
    if records != manifest["records"]:
        raise ValueError("Checkpoint record count does not match its shards")
    return manifest


def convert_record(sample, source_tokenizer, tokenizer, max_length, source_record_index):
    """Preserve one document segment; never pack it together with another record."""
    source_ids = sample["input_ids"]
    indices = np.asarray(sample["indices"])
    if indices.shape != (1, 2) or indices.tolist() != [[0, len(source_ids)]]:
        raise ValueError(
            f"Record {source_record_index}: expected one document interval covering the record; "
            "multi-document packing is not supported by this books baseline"
        )
    text = source_tokenizer.decode(
        source_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    original_length = len(ids)
    ids = ids[:max_length]
    # Do not add EOS: a source record may end in the middle of a book.
    return {
        "input_ids": ids,
        "indices": [[0, len(ids)]],
        "length": len(ids),
        "domain": sample.get("domain"),
        "source_record_index": source_record_index,
        "source_length": len(source_ids),
        "length_before_truncation": original_length,
    }


def write_manifest(directory, manifest):
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ryankamiri/prolong-qwen",
                        help="Destination Hugging Face dataset (default: ryankamiri/prolong-qwen)")
    args = parser.parse_args()
    validate_repo_id(args.repo_id)
    if "/" not in args.repo_id:
        parser.error("--repo-id must include the namespace: username/dataset-name")
    print(f"Destination: https://huggingface.co/datasets/{args.repo_id}", flush=True)
    output_dir = OUTPUT_DIR.resolve()

    api = HfApi()
    # Check auth
    api.whoami()
    checkpoint = load_checkpoint(output_dir)
    dataset_revision = checkpoint["dataset_revision"] if checkpoint else api.dataset_info(DATASET_ID).sha
    source_revision = checkpoint["source_tokenizer_revision"] if checkpoint else api.model_info(SOURCE_TOKENIZER).sha
    target_revision = checkpoint["tokenizer_revision"] if checkpoint else api.model_info(TOKENIZER).sha

    schema = pa.schema([
        ("input_ids", pa.list_(pa.uint32())),
        ("indices", pa.list_(pa.list_(pa.uint32(), 2))),
        ("length", pa.uint32()),
        ("domain", pa.string()),
        ("source_record_index", pa.uint64()),
        ("source_length", pa.uint64()),
        ("length_before_truncation", pa.uint64()),
    ])
    manifest = {
        "status": "incomplete",
        "destination_repo_id": args.repo_id,
        "dataset_id": DATASET_ID, "dataset_revision": dataset_revision,
        "subset": SUBSET,
        "source_tokenizer": SOURCE_TOKENIZER, "source_tokenizer_revision": source_revision,
        "tokenizer": TOKENIZER, "tokenizer_revision": target_revision,
        "max_length": MAX_LENGTH,
        "records_per_shard": RECORDS_PER_SHARD, "row_group_size": ROW_GROUP_SIZE,
        "compression": "zstd", "skip_source_special_tokens": True,
        "clean_up_tokenization_spaces": False, "add_target_special_tokens": False,
        "record_policy": "independent; no padding or cross-record packing",
        "source_record_index": "zero-based index in the pinned subset's sequential MDS stream",
        "records": 0, "tokens": 0, "truncated_records": 0, "dropped_tokens": 0,
        "shards": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        manifest = checkpoint
        manifest["destination_repo_id"] = args.repo_id
        print(f"Resuming after {manifest['records']:,} completed records", flush=True)
    write_manifest(output_dir, manifest)
    stream = stream_prolong(
        subset=SUBSET, revision=dataset_revision, dataset_id=DATASET_ID,
        start_record=manifest["records"],
    )
    rows = None
    if manifest["status"] != "complete":
        print(f"Converting with {NUM_WORKERS} worker processes", flush=True)
        rows = parallel_rows(stream, manifest["records"], source_revision, target_revision)

    writer = None
    buffer = []
    shard_records = 0

    def finish_shard():
        nonlocal writer, buffer, shard_records
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=ROW_GROUP_SIZE)
            buffer = []
        writer.close()
        writer = None
        temporary.replace(final_path)
        manifest["shards"].append({"file": final_path.name, "records": shard_records})
        write_manifest(output_dir, manifest)
        print(f"Wrote {final_path.name}: {shard_records} records; total {manifest['records']}", flush=True)
        shard_records = 0

    try:
        for row in rows if rows is not None else ():
            if writer is None:
                final_path = output_dir / f"train-{len(manifest['shards']):05d}.parquet"
                temporary = final_path.with_suffix(".parquet.tmp")
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            buffer.append(row)
            shard_records += 1
            manifest["records"] += 1
            manifest["tokens"] += row["length"]
            dropped = row["length_before_truncation"] - row["length"]
            manifest["dropped_tokens"] += dropped
            manifest["truncated_records"] += int(dropped > 0)
            if len(buffer) >= ROW_GROUP_SIZE:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema), row_group_size=ROW_GROUP_SIZE)
                buffer = []
            if shard_records == RECORDS_PER_SHARD:
                finish_shard()
        if writer is not None:
            finish_shard()
    finally:
        if rows is not None:
            rows.close()
        if writer is not None:
            writer.close()  # An interrupted shard stays .tmp, not a completed Parquet file.

    manifest["status"] = "complete"
    write_manifest(output_dir, manifest)
    print(f"Done: {manifest['records']} records, {manifest['tokens']} Qwen tokens", flush=True)

    # Explicitly identify the Parquet files so manifest.json is not loaded as data.
    (output_dir / "README.md").write_text(
        "---\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: train-*.parquet\n"
        "---\n\n"
        "# ProLong books tokenized for Qwen\n\n"
        f"Source: [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID}), subset `{SUBSET}`.\n\n"
        f"Decoded with `{SOURCE_TOKENIZER}` and encoded with `{TOKENIZER}`. "
        f"Each source record is independent and truncated to at most {MAX_LENGTH:,} target tokens. "
        "Shorter records are unpadded. Source special tokens are removed; target BOS/EOS are not added.\n\n"
        "See `manifest.json` for pinned revisions, settings, and counts. "
        "Reset model session state between records. No tokenizer round-trip validation was performed.\n",
        encoding="utf-8",
    )
    repo_url = api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading to {repo_url}...", flush=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        allow_patterns=[shard["file"] for shard in manifest["shards"]] + ["manifest.json", "README.md"],
        # Remove old generated shards that are absent from this completed run.
        delete_patterns=["train-*.parquet"],
        commit_message="Upload Qwen-tokenized ProLong books",
    )
    print(f"Uploaded: {repo_url}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run the same command to resume; only the unfinished shard is redone.", flush=True)
        sys.exit(130)
