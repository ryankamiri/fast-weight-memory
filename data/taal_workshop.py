"""Versioned workshop pilot: matched continuation data and two-fact selection.

Natural text comes from the already cached, pinned ProLong document pools.
Document hashes are split before constructing episodes; no new download or
model-generated filler is involved. These are familiar-label context tests.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_d_heldout import D_CHECKPOINT_SHA256, D_DATASET_SHA256, repeat_tokens
from data.taal_min_capacity import build_capacity_examples


ROOT = Path("data/taal_workshop")
PARENT = "checkpoints/taal-capacity-d-distance-diversity-11021676/final"
SOURCES = ("book-65536", "fineweb-edu", "dolmawiki")
SOURCE_REVISION = "1447128c751f0fbc7d2082b2a113a938426a9bba"
SCHEMA = "taal-workshop-pilot/v1"


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def split_documents(pools: dict[str, list[dict]], forbidden: set[int]) -> dict[str, list[dict]]:
    """Deterministic, globally hash-disjoint text splits, stratified by source."""
    splits = {"train": [], "dev": [], "test": []}
    seen = set()
    for source in SOURCES:
        candidates = []
        for row in sorted(pools[source], key=lambda row: row["document_sha256"]):
            digest = row["document_sha256"]
            # Exclude every candidate label token from all filler actually used.
            if digest in seen or len(row["input_ids"]) < 512 or forbidden.intersection(row["input_ids"][:512]):
                continue
            seen.add(digest)
            candidates.append(row)
        if len(candidates) < 24:
            raise ValueError(f"Need 24 label-free documents for {source}, got {len(candidates)}")
        for split, start, end in (("train", 0, 16), ("dev", 16, 20), ("test", 20, 24)):
            splits[split].extend(candidates[start:end])
    # Early bounded subsets must cover all sources, rather than taking only
    # the first source's documents from a concatenated pool.
    for split, documents in splits.items():
        per_source = len(documents) // len(SOURCES)
        interleaved = []
        for index in range(per_source):
            for source_index in range(len(SOURCES)):
                interleaved.append(documents[source_index * per_source + index])
        splits[split] = interleaved
    return splits


def tag(original: dict, name: str, split: str) -> dict:
    row = copy.deepcopy(original)
    row.update(
        example_id=f"{split}-{name}-{original['example_id']}",
        condition=name, evaluation_split=split,
        source_example_id=original["example_id"],
        pair_group=f"{split}/{name}/{original['record_name']}",
        prefix_group=f"{split}/{name}/{original['example_id']}",
        selection_group="", fact_count=1,
        fact_positions=[original["fact_position"]],
        fact_spans=copy.deepcopy(original["fact_spans"]),
        filler_document_sha256="", filler_source="original_repeated_sentence",
        pre_fact_tokens=0, fact_to_answer_gap=len(row["input_ids"]) - row["fact_position"],
    )
    return row


def context_example(tokenizer, original: dict, *, name: str, split: str,
                    document: dict | None, pre_fact_tokens: int, gap: int) -> dict:
    row = tag(original, name, split)
    fact_end_ids = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
    fact_end = original["fact_position"] + 1 + len(fact_end_ids)
    fact = original["input_ids"][:fact_end]
    query = original["input_ids"][original["final_query_position"]:]
    filler_length = gap - 1 - len(fact_end_ids) - len(query)
    if filler_length < 33:
        raise ValueError("Both facts must leave the 32-token KV before the query")
    if document is None:
        neutral = tokenizer.encode(" A neutral passage continues without mentioning any record or label.", add_special_tokens=False)
        tokens = repeat_tokens(neutral, pre_fact_tokens + filler_length)
    else:
        tokens = document["input_ids"][:pre_fact_tokens + filler_length]
        if len(tokens) != pre_fact_tokens + filler_length:
            raise ValueError("Insufficient natural filler")
        row.update(filler_document_sha256=document["document_sha256"], filler_source=document["source"])
    prefix = tokens[:pre_fact_tokens] + fact + tokens[pre_fact_tokens:]
    row.update(
        input_ids=prefix + query, final_query_position=len(prefix),
        final_answer_position=len(prefix) + len(query), prompt_length=len(prefix) + len(query),
        fact_position=pre_fact_tokens + original["fact_position"],
        fact_positions=[pre_fact_tokens + original["fact_position"]],
        fact_spans=[[pre_fact_tokens, pre_fact_tokens + fact_end]],
        pre_fact_tokens=pre_fact_tokens, fact_to_answer_gap=gap,
    )
    return row


def selection_examples(tokenizer, base: list[dict]) -> list[dict]:
    """Two records, two assignments, two orders, two queries per identical prefix."""
    rows = []
    neutral = tokenizer.encode(" A neutral passage continues without mentioning any record or label.", add_special_tokens=False)
    for pair in range(2):
        records = [base[4 * pair], base[4 * pair + 2]]
        labels = [base[4 * pair], base[4 * pair + 1]]
        for order in ((0, 1), (1, 0)):
            for assignment in range(2):
                prefix, positions, spans = [], {}, []
                for record_index in order:
                    start = len(prefix)
                    head = tokenizer.encode(f"User: Remember that {records[record_index]['record_name']}'s archive label is", add_special_tokens=False)
                    tail = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
                    positions[record_index] = len(prefix) + len(head)
                    prefix += head + [labels[(record_index + assignment) % 2]["target_token_id"]] + tail
                    spans.append([start, len(prefix)])
                prefix += repeat_tokens(neutral, 128)
                for query_index, source in enumerate(records):
                    label = labels[(query_index + assignment) % 2]
                    query = source["input_ids"][source["final_query_position"]:]
                    group = f"selection/{pair}/{order[0]}"
                    row = tag(source, "selection", "test")
                    row.update(
                        example_id=f"selection-{pair}-{order[0]}-{assignment}-q{query_index}",
                        pair_group=f"{group}/q{query_index}",
                        prefix_group=f"{group}/assignment{assignment}",
                        selection_group=group,
                        answer=label["answer"], target_token_id=label["target_token_id"],
                        input_ids=prefix + query, final_query_position=len(prefix),
                        final_answer_position=len(prefix) + len(query), prompt_length=len(prefix) + len(query),
                        fact_position=positions[query_index], fact_positions=sorted(positions.values()),
                        fact_spans=spans, fact_count=2,
                        fact_to_answer_gap=len(prefix) + len(query) - positions[query_index],
                    )
                    rows.append(row)
    return rows


def build_bundle(tokenizer, pools: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    base = build_capacity_examples(tokenizer, pairs=8, gap=192)
    tail = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
    for row in base:
        row["fact_spans"] = [[0, row["fact_position"] + 1 + len(tail)]]
    documents = split_documents(pools, set(base[0]["candidate_token_ids"]))
    narrow = [tag(row, "reference", "train") for row in base]
    diverse = copy.deepcopy(narrow)
    # 16/128 records retain the original layout. Other cells vary only context.
    layouts = ((0, 192), (32, 192), (64, 192), (0, 96), (0, 320), (32, 256), (64, 320))
    for layout, (offset, gap) in enumerate(layouts):
        for index, original in enumerate(base):
            document = documents["train"][(layout * 8 + index // 2) % len(documents["train"])]
            # Match the shuffled record/answer sequence exactly across arms.
            # Repeating the original layout does not add context diversity.
            narrow.append(tag(original, f"repeat_{layout}", "train"))
            diverse.append(context_example(tokenizer, original, name=f"varied_{layout}", split="train",
                                           document=document, pre_fact_tokens=offset, gap=gap))
    dev = []
    for index, original in enumerate(base):
        dev.append(context_example(tokenizer, original, name="development", split="dev",
                                   document=documents["dev"][index // 2],
                                   pre_fact_tokens=32, gap=224))
    evaluation = [tag(row, "reference", "test") for row in base]
    cells = (
        ("filler", 0, 192, True), ("position", 64, 192, False),
        ("distance", 0, 256, False), ("combined", 48, 288, True),
    )
    for name, offset, gap, natural in cells:
        for index, original in enumerate(base[:8]):
            evaluation.append(context_example(tokenizer, original, name=name, split="test",
                                              document=documents["test"][index // 2] if natural else None,
                                              pre_fact_tokens=offset, gap=gap))
    evaluation += selection_examples(tokenizer, base)
    files = {"narrow.jsonl": narrow + copy.deepcopy(dev),
             "diverse.jsonl": diverse + copy.deepcopy(dev), "evaluation.jsonl": evaluation}
    for rows in files.values():
        for index, row in enumerate(rows):
            row["fact_id"] = index
    metadata = {
        "schema": SCHEMA, "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "source_repo": "princeton-nlp/prolong-data-64K", "source_revision": SOURCE_REVISION,
        "parent_checkpoint": PARENT, "parent_checkpoint_sha256": D_CHECKPOINT_SHA256,
        "parent_dataset_sha256": D_DATASET_SHA256,
        "documents": {split: [{key: doc[key] for key in (
            "document_sha256", "source", "source_record", "document_index",
        )} for doc in selected] for split, selected in documents.items()},
        "splits": {"narrow": {"train": [0, 128], "dev": [128, 144]},
                   "diverse": {"train": [0, 128], "dev": [128, 144]}},
        "evaluation_cells": dict(Counter(row["condition"] for row in evaluation)),
        "limits": "One parent checkpoint, familiar labels/queries, small context-transfer pilot. Exact source-document isolation, not guaranteed book-level isolation. Reference cases are deliberate regression controls.",
    }
    return files, metadata


def write_bundle(output: Path, files: dict[str, list[dict]], metadata: dict) -> dict:
    payloads = {name: jsonl(rows) for name, rows in files.items()}
    manifest = metadata | {"files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in payloads.items()}}
    payloads["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    for name in files:
        payloads[name.replace(".jsonl", ".metadata.json")] = json.dumps(metadata, sort_keys=True) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = output / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite a different versioned dataset: {path}")
    for name, content in payloads.items():
        (output / name).write_text(content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=Path("data/taal_cpt"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    source_manifest = json.loads((args.pool_root / "manifest.json").read_text())
    if source_manifest["source_revision"] != SOURCE_REVISION or source_manifest["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Source pools have different pinned dataset/tokenizer revisions")
    pools = {}
    for source in SOURCES:
        path = args.pool_root / f"pool-{source}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_manifest["files"][path.name]:
            raise ValueError(f"Source pool checksum mismatch: {path}")
        pools[source] = [json.loads(line) for line in path.read_text().splitlines()]
    files, metadata = build_bundle(tokenizer, pools)
    manifest = write_bundle(args.output, files, metadata)
    print(json.dumps({"files": manifest["files"], "evaluation_cells": manifest["evaluation_cells"]}, indent=2))


if __name__ == "__main__":
    main()
