"""Evaluate the existing BF16 acquisition in FP32 without changing its artifacts."""

import argparse
import json
import os
from pathlib import Path
import signal

import torch

from evaluation.storage import atomic_json, ensure_manifest, file_sha256
from scripts.retrieval_exposure_audit import Allocation
from scripts.single_fact import ROOT, audit_checkpoint
from scripts.single_fact_precision import compare_precisions
from scripts.single_fact_preflight import check_inputs, check_parent_submission, check_precision_recipe
from training.config import load_config
from utils.seed import seed_everything


MAX_ALLOCATIONS = 2


def check_completed_parent(parent, parent_job):
    check_parent_submission(parent, parent_job)
    job = json.loads((parent / "jobs/acquisition/job.json").read_text())
    complete = json.loads((parent / "jobs/acquisition/complete.json").read_text())
    audit = json.loads((parent / "acquisition/audit/complete.json").read_text())
    if job["job_id"] != parent_job or not complete.get("complete"):
        raise ValueError("Acquisition job has not completed")
    if not audit.get("complete") or audit.get("step") != 768:
        raise ValueError("Expected the completed step768 acquisition audit")

    recipe = parent / "recipes/acquisition.yaml"
    actual = load_config(recipe).to_dict()
    expected = load_config("training/configs/taal/single_fact.yaml").to_dict()
    actual.pop("wandb")
    expected.pop("wandb")
    if actual != expected:
        raise ValueError("Parent is not the matched BF16 acquisition recipe")

    return recipe


def compare_parent(output, baseline):
    fp32 = json.loads((output / "fp32/checkpoint.json").read_text())
    bf16 = json.loads((baseline / "checkpoint.json").read_text())
    if fp32.pop("precision") != "fp32" or bf16.pop("precision") != "bf16":
        raise ValueError("Cross-precision audit has incorrect execution policies")
    if fp32 != bf16:
        raise ValueError("Cross-precision comparison requires the same checkpoint and provenance")

    compare_precisions(output, baseline, "bf16")


def pipeline(parent, parent_job, output, job_id, restart, device):
    if restart >= MAX_ALLOCATIONS:
        raise ValueError("Cross-precision audit allocation budget exhausted")
    if parent.resolve() == output.resolve() or parent.resolve() in output.resolve().parents:
        raise ValueError("Cross-precision results need a separate output directory")

    check_inputs()
    match = check_precision_recipe()
    recipe = check_completed_parent(parent, parent_job)
    checkpoint = Path("checkpoints") / f"{parent.name}-acquisition" / "final"
    ensure_manifest(output / "job.json", {
        "job_id": job_id, "parent_job": parent_job, "parent_output": str(parent),
        "checkpoint": str(checkpoint), "training_precision": "bf16", "evaluation_precision": "fp32",
        "max_allocations": MAX_ALLOCATIONS, "allocation_hours": 2, "save_traces": True,
    })
    ensure_manifest(output / "match.json", match)
    ensure_manifest(output / "dataset.json", {"sha256": file_sha256(ROOT / "manifest.json")})
    allocation = Allocation(MAX_ALLOCATIONS)
    previous = {}
    for signum in (signal.SIGUSR1, signal.SIGTERM):
        previous[signum] = signal.signal(signum, allocation.signal)
    try:
        print(f"SINGLE_FACT_CROSS_PRECISION parent={parent_job} checkpoint={checkpoint} training=bf16 evaluation=fp32", flush=True)
        if allocation.stopped():
            allocation.finish(False)
        complete = audit_checkpoint(recipe, checkpoint, output / "fp32", allocation, device, False, "fp32")
        allocation.finish(complete)
        # Read the parent's saved BF16 answers; never rerun or rewrite that evaluation.
        compare_parent(output, parent / "acquisition/audit")
        atomic_json(output / "complete.json", {
            "complete": True, "step": 768, "save_traces": True,
            "training_precision": "bf16", "evaluation_precision": "fp32",
            "questions": {"seen": 32, "validation": 128}, "parent_job": parent_job,
        })
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = os.environ["SLURM_JOB_ID"]
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise ValueError("Expected a CUDA Slurm allocation")

    seed_everything(42)
    pipeline(args.parent_output, args.parent_job, args.output_dir, job_id, restart, torch.device("cuda"))


if __name__ == "__main__":
    main()
