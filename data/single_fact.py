"""Single-fact acquisition and context robustness, with independent test records."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

from data.balanced_retrieval import FILLER, LABELS, render_example
from data.retrieval_exposure import write_rows
from evaluation.storage import ensure_manifest, file_sha256


ROOT = Path("data/single_fact")
VERSION = "single-fact-retrieval-v1"
REGIMES = ("familiar", "filler", "position", "combined")
PASSAGES = {
    "train": (
        "The morning bus stopped near the market. Passengers waited beside the timetable.",
        "The cook washed the dishes after lunch. A clean towel hung beside the sink.",
        "Workers inspected the roof before sunset. Their equipment stayed by the entrance.",
    ),
    "val": (
        "The museum opened its doors at noon. Visitors walked quietly through the exhibition.",
        "The gardener watered the plants before breakfast. Small birds gathered near the fence.",
        "The mechanic checked the brakes carefully. Several tools lay on the workbench.",
    ),
    "test": (
        "The weather station reported steady rain. Clouds covered the hills throughout the afternoon.",
        "The librarian sorted returned books. Readers waited quietly beside the front desk.",
        "The orchestra rehearsed in the hall. Musicians checked their instruments before the concert.",
    ),
}


@dataclass(frozen=True)
class SingleFactConfig:
    seed: int = 426
    kv_window: int = 128
    min_gap_tokens: int = 192
    max_sequence_tokens: int = 1024
    train_pairs: int = 16
    val_per_regime: int = 32
    test_per_regime: int = 128


def make_example(tokenizer, config, split, index, regime, key, label, assignment=0):
    identity = f"single-{split}-{index:06d}-{regime}"
    rng = random.Random(f"{config.seed}/{identity}")
    filler = FILLER
    if regime in ("filler", "combined"):
        passages = rng.sample(list(PASSAGES[split]), len(PASSAGES[split]))
        filler = "\n".join(passages) + "\n"
    prefact = ""
    if regime in ("position", "combined"):
        prefact = (rng.choice(PASSAGES[split]) + "\n") * rng.choice((2, 4, 6))
    gap = config.min_gap_tokens
    if regime == "combined":
        gap = rng.choice((256, 384, 512))
    row = render_example(tokenizer, replace(config, min_gap_tokens=gap), identity,
                         [key], [label], [0], assignment, 0, prefact, filler)
    row["metadata"].update(split=split, regime=regime, requested_gap_tokens=gap,
                           filler_kind="synthetic split-specific paragraph bank",
                           prefact_text=prefact)
    return row


def record_key(config, split, index):
    digest = hashlib.sha256(f"{config.seed}/{split}/{index}".encode()).hexdigest()
    return "R" + digest[:16]


def build(tokenizer, config):
    if config.train_pairs != len(LABELS):
        raise ValueError("Acquisition uses sixteen counterbalanced record pairs")
    if any(n < 16 or n % 16 for n in (config.val_per_regime, config.test_per_regime)):
        raise ValueError("Each evaluation stratum must balance all sixteen labels")
    if config.kv_window != 128 or config.min_gap_tokens < config.kv_window:
        raise ValueError("The experiment requires verified eviction beyond KV128")
    if not tokenizer.is_fast:
        raise ValueError("Token offsets require a fast tokenizer")
    label_ids = [tokenizer.encode(" " + value, add_special_tokens=False) for value in LABELS]
    if any(not ids for ids in label_ids) or len({tuple(ids) for ids in label_ids}) != len(LABELS):
        raise ValueError("Labels must have distinct nonempty token sequences")
    control_labels = [label for label, ids in zip(LABELS, label_ids) if len(ids) == 1]
    if len(control_labels) < 8:
        raise ValueError("The causal audit needs eight single-token labels")
    bundle = {"narrow": [], "mixed": [], "val": [], "test": [], "controls": []}
    for regime in REGIMES:
        for index in range(config.train_pairs):
            key = record_key(config, "train", index)
            for assignment in (0, 1):
                label = LABELS[(index + assignment) % len(LABELS)]
                row = make_example(tokenizer, config, "train", index, regime, key, label, assignment)
                # Assignment variants and context changes are one dependent family.
                row["metadata"]["family_id"] = f"single-train-family{index:04d}"
                bundle["mixed"].append(row)
                if regime == "familiar":
                    bundle["narrow"].append(row)
    for split, per_regime in (("val", config.val_per_regime), ("test", config.test_per_regime)):
        for regime_index, regime in enumerate(REGIMES):
            for index in range(per_regime):
                episode = regime_index * per_regime + index
                key = record_key(config, split, episode)
                label = LABELS[index % len(LABELS)]
                row = make_example(tokenizer, config, split, episode, regime, key, label)
                bundle[split].append(row)
                if split == "test" and index < len(LABELS) and label in control_labels[:8]:
                    # Thirty-two recipients plus their matched alternative labels.
                    # These extra variants are mechanism controls, not extra n.
                    donor_label = control_labels[(control_labels.index(label) + 4) % len(control_labels)]
                    donor = make_example(tokenizer, config, split, episode, regime, key, donor_label, 1)
                    for member in (row, donor):
                        bundle["controls"].append(audit_example(tokenizer, member, episode))
    random.Random(config.seed).shuffle(bundle["val"])
    for rows in bundle.values():
        if len({row["example_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate example IDs")
    return bundle


def audit_example(tokenizer, row, index):
    metadata = row["metadata"]
    answer_ids = tokenizer.encode(" " + row["answer"], add_special_tokens=False)
    if len(answer_ids) != 1:
        raise ValueError("The legacy causal audit scores only one-token answers")
    return {
        "example_id": row["example_id"], "fact_id": index,
        "answer": row["answer"], "condition": metadata["regime"], "query_variant": "original",
        "input_ids": row["prompt_input_ids"], "final_query_position": metadata["query_start_token"],
        "target_token_id": answer_ids[0],
        "candidate_token_ids": [ids[0] for label in LABELS
                                if len(ids := tokenizer.encode(" " + label, add_special_tokens=False)) == 1],
        "fact_spans": metadata["fact_token_spans"],
        "pair_group": metadata["family_id"], "prefix_group": row["example_id"],
    }


def prepare(tokenizer, config, destination):
    bundle = build(tokenizer, config)
    common = {
        "generator_version": VERSION, "generator_sha256": file_sha256(Path(__file__)),
        "config": asdict(config) | {"label_pool": list(LABELS)},
        "tokenizer": tokenizer.name_or_path,
        "tokenizer_backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "special_tokens": tokenizer.special_tokens_map,
        "test_unit": "512 independent record scenarios; donor controls reuse 32 of them",
        "filler": PASSAGES,
    }
    files = {}
    for name, rows in bundle.items():
        files[name] = write_rows(destination / f"{name}.jsonl", rows)
    ensure_manifest(destination / "manifest.json", common | {"files": files})
    for arm in ("narrow", "mixed"):
        cohort_files = {
            "train": write_rows(destination / arm / "train.jsonl", bundle[arm]),
            "val": write_rows(destination / arm / "val.jsonl", bundle["val"]),
        }
        ensure_manifest(destination / arm / "manifest.json", common | {"files": cohort_files})
    ensure_manifest(destination / "controls.metadata.json", {
        "generator_version": VERSION, "files": files, "labels": list(LABELS),
        "tokenizer": "Qwen/Qwen3-0.6B-Base",
        "tokenizer_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "pairing": "same record, filler, position and requested gap; alternate label",
    })
    summary = {name: {"rows": len(rows)} for name, rows in bundle.items()}
    for name in ("narrow", "mixed", "val", "test"):
        rows = bundle[name]
        summary[name].update(
            labels=dict(Counter(row["answer"] for row in rows)),
            regimes=dict(Counter(row["metadata"]["regime"] for row in rows)),
            sequence_tokens=[min(len(row["input_ids"]) for row in rows), max(len(row["input_ids"]) for row in rows)],
            minimum_verified_gap=min(row["metadata"]["verified_gap_tokens"] for row in rows),
        )
    ensure_manifest(destination / "validation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(tokenizer, SingleFactConfig(), args.output), indent=2))


if __name__ == "__main__":
    main()
