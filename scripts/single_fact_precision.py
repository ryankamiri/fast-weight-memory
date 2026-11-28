"""Matched FP32 acquisition and a dependent audit in both execution precisions."""

import argparse
import json
import os
from pathlib import Path
import signal

import torch

from evaluation.shared.__main__ import load_rows
from evaluation.storage import atomic_json, ensure_manifest, file_sha256, read_results
from scripts.retrieval_acquisition import validate_checkpoint
from scripts.retrieval_exposure_audit import Allocation
from scripts.retrieval_parallel import needs_training
from scripts.retrieval_training_segment import run_segment
from scripts.single_fact import ROOT, audit_checkpoint
from scripts.single_fact_preflight import check_inputs, check_precision_recipe
from scripts.single_fact_scoring import predicted_label
from training.config import load_config
from utils.seed import seed_everything


RECIPE = Path("training/configs/taal/single_fact_fp32.yaml")
PHASE_ALLOCATIONS = {"train": 6, "audit": 4}


def compare_precisions(output, bf16_output=None, training_precision="fp32"):
    if bf16_output is None:
        bf16_output = output / "bf16"
    directories = {"fp32": output / "fp32", "bf16": bf16_output}
    comparisons = {}
    for cohort in ("narrow", "val"):
        rows = load_rows(ROOT / f"{cohort}.jsonl")
        predictions = {}
        for precision in ("fp32", "bf16"):
            predictions[precision] = read_results(directories[precision] / cohort / "predictions.jsonl", "example_id")
            if set(predictions[precision]) != {row["example_id"] for row in rows}:
                raise ValueError("Precision comparison needs both complete cohorts")

        gains = 0
        losses = 0
        changed = 0
        records = []
        for row in rows:
            example_id = row["example_id"]
            fp32 = predicted_label(predictions["fp32"][example_id]["hypothesis"])
            bf16 = predicted_label(predictions["bf16"][example_id]["hypothesis"])
            fp32_correct = fp32 == row["answer"]
            bf16_correct = bf16 == row["answer"]
            gains += int(fp32_correct and not bf16_correct)
            losses += int(bf16_correct and not fp32_correct)
            changed += int(fp32 != bf16)
            records.append({"example_id": example_id, "expected": row["answer"],
                            "fp32_label": fp32, "bf16_label": bf16})
        comparisons[cohort] = {
            "questions": len(rows), "fp32_gains": gains, "fp32_losses": losses,
            "changed_label": changed, "records": records,
            "scope": f"Same {training_precision.upper()}-trained checkpoint; change evaluation precision only.",
        }
    atomic_json(output / "precision-comparison.json", comparisons)


def pipeline(output, phase, job_id, restart, device):
    budget = PHASE_ALLOCATIONS[phase]
    if restart >= budget:
        raise ValueError("Precision arm allocation budget exhausted")

    check_inputs()
    match = check_precision_recipe()
    config = load_config(RECIPE)
    output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output / "recipe.json", config.to_dict())
    ensure_manifest(output / "match.json", match)
    ensure_manifest(output / "dataset.json", {"sha256": file_sha256(ROOT / "manifest.json")})
    job_output = output / "jobs" / phase
    ensure_manifest(job_output / "job.json", {
        "job_id": job_id, "phase": phase, "max_allocations": budget,
        "allocation_hours": 2, "save_traces": True,
    })

    run_id = f"{output.name}-acquisition"
    checkpoint = Path(config.checkpoints.output_dir) / run_id / "final"
    allocation = Allocation(budget)
    previous = {}
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        previous[signum] = signal.signal(signum, allocation.signal)
    try:
        print(f"SINGLE_FACT_PRECISION phase={phase} training=fp32 target=768 checkpoint={checkpoint}", flush=True)
        if phase == "train":
            if allocation.stopped():
                allocation.finish(False)
            if needs_training(checkpoint, config.training.max_steps):
                run_segment(RECIPE, run_id, job_id, restart, budget)
            metadata = validate_checkpoint(RECIPE, checkpoint)
            # afterok can release the audit only after the exact training budget is verified.
            allocation.finish(True)
            atomic_json(job_output / "complete.json", {
                "complete": True, "checkpoint": str(checkpoint),
                "step": metadata["progress"]["step"], "save_traces": True,
            })
            return

        receipt = output / "jobs/train/complete.json"
        if not receipt.is_file():
            raise ValueError("FP32 training receipt is missing; refusing dependent evaluation")

        training = json.loads(receipt.read_text())
        if not training.get("complete") or training.get("step") != 768 or training.get("checkpoint") != str(checkpoint):
            raise ValueError("FP32 training receipt does not identify the completed parent")

        # Each precision gets its own manifest, predictions, losses and token/layer traces.
        for precision in ("fp32", "bf16"):
            if allocation.stopped():
                allocation.finish(False)
            complete = audit_checkpoint(RECIPE, checkpoint, output / "audit" / precision,
                                        allocation, device, False, precision)
            allocation.finish(complete)
        compare_precisions(output / "audit")
        atomic_json(job_output / "complete.json", {
            "complete": True, "step": 768, "save_traces": True,
            "evaluation_precisions": ["fp32", "bf16"],
            "questions_per_precision": {"seen": 32, "validation": 128},
        })
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASE_ALLOCATIONS, required=True)
    args = parser.parse_args()
    job_id = os.environ["SLURM_JOB_ID"]
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise ValueError("Expected a CUDA Slurm allocation")

    seed_everything(42)
    pipeline(args.output_dir, args.phase, job_id, restart, torch.device("cuda"))


if __name__ == "__main__":
    main()
