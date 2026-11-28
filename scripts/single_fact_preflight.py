"""Hash and cohort checks safe to run on an Explorer login node."""

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from evaluation.storage import file_sha256


def check_inputs(root=Path("data/single_fact")):
    manifest = json.loads((root / "manifest.json").read_text())
    expected_counts = {"narrow": 32, "mixed": 128, "val": 128, "test": 512, "controls": 64}
    cohorts = {}
    for name, count in expected_counts.items():
        path = root / f"{name}.jsonl"
        if file_sha256(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Changed dataset: {name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != count or len({r["example_id"] for r in rows}) != count:
            raise ValueError(f"Unexpected unique row count: {name}")
        cohorts[name] = rows
        if name == "controls":
            continue
        for row in rows:
            metadata = row["metadata"]
            for start, end in metadata["fact_token_spans"]:
                if not 0 <= start < end <= metadata["query_start_token"] - 128:
                    raise ValueError("Fact still visible in KV128")
            if metadata["verified_gap_tokens"] < metadata["requested_gap_tokens"]:
                raise ValueError("Incorrect token gap")
            if len(row["input_ids"]) > 1024:
                raise ValueError("Episode exceeds approved training budget")
    for arm in ("narrow", "mixed"):
        child = json.loads((root / arm / "manifest.json").read_text())
        for split, source in (("train", arm), ("val", "val")):
            digest = file_sha256(root / arm / f"{split}.jsonl")
            if digest != child["files"][split]["sha256"] or digest != manifest["files"][source]["sha256"]:
                raise ValueError("Training cohort and top-level manifest disagree")
    keys = {name: {key for row in cohorts[name] for key in row["metadata"]["bindings"]}
            for name in ("narrow", "val", "test")}
    if keys["narrow"] & keys["val"] or keys["narrow"] & keys["test"] or keys["val"] & keys["test"]:
        raise ValueError("Train/val/test record overlap")
    if len(keys["test"]) != 512:
        raise ValueError("Test questions must represent 512 distinct record scenarios")
    labels = manifest["config"]["label_pool"]
    for regime in ("familiar", "filler", "position", "combined"):
        rows = [row for row in cohorts["test"] if row["metadata"]["regime"] == regime]
        if Counter(row["answer"] for row in rows) != Counter({label: 8 for label in labels}):
            raise ValueError("Test stratum is not label balanced")
    pairs = {}
    for row in cohorts["controls"]:
        pairs.setdefault(row["pair_group"], []).append(row)
    if len(pairs) != 32 or any(len(rows) != 2 or rows[0]["target_token_id"] == rows[1]["target_token_id"] for rows in pairs.values()):
        raise ValueError("Causal audit needs 32 opposing-label pairs")
    print("SINGLE_FACT_INPUTS_VALID " + json.dumps(expected_counts), flush=True)
    return manifest


def check_precision_recipe(reference=Path("training/configs/taal/single_fact.yaml"),
                           candidate=Path("training/configs/taal/single_fact_fp32.yaml")):
    baseline = yaml.safe_load(reference.read_text())
    fp32 = yaml.safe_load(candidate.read_text())
    if baseline["model"].get("execution_precision", "bf16") != "bf16":
        raise ValueError("The matched acquisition reference must remain BF16")
    if fp32["model"].get("execution_precision") != "fp32":
        raise ValueError("The new acquisition arm must request FP32")

    baseline["model"]["execution_precision"] = "fp32"
    baseline["wandb"]["name"] = fp32["wandb"]["name"]
    if baseline != fp32:
        raise ValueError("FP32 acquisition must differ only in precision and run name")
    if fp32["training"]["max_steps"] != 768 or not fp32["training"]["save_traces"]:
        raise ValueError("Expected 768 updates and saved traces")

    return {"reference_sha256": file_sha256(reference), "fp32_sha256": file_sha256(candidate),
            "changed_fields": ["model.execution_precision", "wandb.name"], "updates": 768}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    check_inputs()
    if args.fp32:
        print("SINGLE_FACT_FP32_MATCH_VALID " + json.dumps(check_precision_recipe()), flush=True)
