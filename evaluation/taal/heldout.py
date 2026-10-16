"""Guard the fixed D checkpoint and report each held-out arm separately."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer
import yaml

from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_d_heldout import (
    D_CHECKPOINT_SHA256, D_DATASET_SHA256, EXPERIMENTS,
    build_heldout_examples, dataset_metadata, dataset_text,
)
from evaluation.scoring import aggregate_scores
from evaluation.taal.interventions import audit_conditions
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.state_audit import paired_differences, select_examples, swap_sources


def verify_inputs(
    experiment: str, config_path: Path, checkpoint: Path, tokenizer,
    reference_path: Path = Path("data/taal_capacity_16_gap192.jsonl"),
) -> list[dict]:
    settings = yaml.safe_load(config_path.read_text())
    expected = build_heldout_examples(tokenizer, experiment)
    dataset = Path(settings["dataset"]["path"])
    if dataset.read_text() != dataset_text(expected):
        raise ValueError("Held-out dataset differs from the deterministic, isolated setup")

    if json.loads(dataset.with_suffix(".metadata.json").read_text()) != dataset_metadata(expected, experiment):
        raise ValueError("Held-out metadata differs from the deterministic setup")

    selected = select_examples(expected, settings["dataset"])
    if {row["example_id"] for row in selected} != {row["example_id"] for row in expected}:
        raise ValueError("Held-out config must select every reference and test row")

    required = {
        "execution_block_size": 32, "state_bank_batch_size": 1, "seed": 42,
        "save_traces": True, "swap_group_field": "pair_group", "audit_suite": "standard",
        "persistent_writes_enabled": True, "query_updates_enabled": True,
    }
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("Held-out audits require matched native controls and all traces")

    swap_sources(selected, "pair_group")
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    model = json.loads((checkpoint / "config.json").read_text())
    if metadata["progress"]["step"] != 449 or metadata["reason"] == "exception":
        raise ValueError("Held-out audits require the original successful D final449")

    if (model["num_hidden_layers"], model["working_memory_size"], model["memory_chunk_size"]) != (28, 32, 1):
        raise ValueError("Held-out audits require unchanged D layers, KV and memory cadence")

    if checkpoint_fingerprint(checkpoint) != D_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the verified D final449")

    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != D_DATASET_SHA256:
        raise ValueError("Reference source differs from D's verified training data")

    print("D_HELDOUT_INPUTS " + json.dumps({
        "experiment": experiment, "checkpoint_sha256": D_CHECKPOINT_SHA256,
        **dataset_metadata(expected, experiment),
    }), flush=True)
    return selected


def summarize_heldout(rows: list[dict], examples: list[dict]) -> dict:
    conditions = audit_conditions("standard")
    expected = {
        f"{example['example_id']}/{condition.name}"
        for example in examples for condition in conditions
    }
    if len(rows) != len(expected) or {row["evaluation_id"] for row in rows} != expected:
        raise ValueError("Held-out summary requires every unique example/condition result")

    by_id = {row["evaluation_id"]: row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset_condition"]].append(row)
    arms = {}
    for arm, selected in groups.items():
        metrics = {}
        for condition in conditions:
            values = [row for row in selected if row["audit_condition"] == condition.name]
            aggregate = aggregate_scores(values)
            loss = -aggregate["mean_target_log_probability"]
            metrics[condition.name] = {
                **aggregate,
                "answer_loss": loss,
                "answer_perplexity": math.exp(loss),
                "donor_greedy_rate": sum(row["donor_target_vocabulary_top_1"] for row in values) / len(values),
                "mean_own_minus_donor_label_log_probability": sum(
                    row["own_minus_donor_label_log_probability"] for row in values
                ) / len(values),
            }
        pairs = defaultdict(list)
        for row in selected:
            if row["audit_condition"] == "correct_full":
                swapped = by_id[f"{row['example_id']}/swapped"]
                pairs[row["record_group"]].append(
                    row["vocabulary_top_1_correct"] and swapped["donor_target_vocabulary_top_1"]
                )
        arms[arm] = {
            "metrics": metrics,
            "paired_vs_correct_full": paired_differences(selected, conditions),
            "record_pairs": len(pairs),
            "pairs_both_directions_own_and_donor_correct": sum(all(values) for values in pairs.values()),
        }
    reference = {row["source_example_id"]: row for row in examples if row["condition"] == "reference"}
    contrasts = []
    for example in examples:
        if example["condition"] == "reference":
            continue

        source = reference[example["source_example_id"]]
        baseline = by_id[f"{source['example_id']}/correct_full"]
        heldout = by_id[f"{example['example_id']}/correct_full"]
        contrasts.append({
            "example_id": example["example_id"], "arm": example["condition"],
            "reference_example_id": source["example_id"],
            "same_target": source["target_token_id"] == example["target_token_id"],
            "reference_correct": baseline["vocabulary_top_1_correct"],
            "heldout_correct": heldout["vocabulary_top_1_correct"],
            "heldout_minus_reference_target_log_probability": heldout["target_log_probability"] - baseline["target_log_probability"],
        })
    result = {"arms": arms, "reference_contrasts": contrasts, "rows": len(rows)}
    json.dumps(result, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = Path(f"evaluation/configs/taal/d_heldout_{args.experiment}.yaml")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    examples = verify_inputs(args.experiment, config, args.checkpoint, tokenizer)
    if args.check_only:
        return

    subprocess.run([
        sys.executable, "-m", "utils.explorer_preflight", "--config", str(config),
        "--checkpoint", str(args.checkpoint),
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "evaluation.taal.state_audit", "--config", str(config),
        "--checkpoint", str(args.checkpoint), "--output-dir", str(args.output_dir),
    ], check=True)
    rows = [json.loads(line) for line in (args.output_dir / "results.jsonl").read_text().splitlines()]
    summary = summarize_heldout(rows, examples)
    (args.output_dir / "heldout_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("D_HELDOUT_COMPLETE " + json.dumps({
        "experiment": args.experiment, "examples": len(examples),
        "rows": len(rows), "arms": list(summary["arms"]),
    }), flush=True)


if __name__ == "__main__":
    main()
