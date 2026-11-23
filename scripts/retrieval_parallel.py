"""One bounded, resumable Slurm allocation for each parallel experiment branch."""

import argparse
import json
import os
from pathlib import Path
import signal

import torch

from evaluation.storage import atomic_json, ensure_manifest
from scripts.retrieval_exposure_audit import Allocation
from scripts.retrieval_parallel_audit import ARMS, audit_trained, recipe_path, run_baselines
from scripts.retrieval_training_segment import checkpoint_progress, run_segment
from training.config import load_config
from utils.seed import seed_everything


def needs_training(checkpoint, target):
    if not checkpoint.exists():
        return True
    step = checkpoint_progress(checkpoint)["progress"]["step"]
    if step > target:
        raise ValueError("Checkpoint exceeds the approved update budget")
    return step < target


def run_pipeline(arm, output, job_id, restart_count, device):
    allocation = Allocation()
    prior_usr1 = signal.signal(signal.SIGUSR1, allocation.signal)
    prior_term = signal.signal(signal.SIGTERM, allocation.signal)
    try:
        output.mkdir(parents=True, exist_ok=True)
        ensure_manifest(output / "job.json", {"arm": arm, "job_id": job_id, "save_traces": True,
                        "max_allocations": 6, "allocation_hours": 2})
        if arm in ARMS:
            recipe = recipe_path(arm)
            config = load_config(recipe)
            ensure_manifest(output / "recipe.json", config.to_dict())
            run_id = f"taal-parallel-{arm}-{job_id}"
            checkpoint = Path(config.checkpoints.output_dir) / run_id / "final"
            if needs_training(checkpoint, config.training.max_steps):
                warned = run_segment(recipe, run_id, job_id, restart_count)
                # A warning received by the child trainer still applies to this
                # allocation. Start the audit only after a fresh allocation.
                allocation.timed_out = allocation.timed_out or warned
            if allocation.stopped():
                allocation.finish(False)
            complete = audit_trained(arm, checkpoint, output / "audit", allocation, device)
        elif arm == "baselines":
            complete = run_baselines(output, allocation, device)
        elif arm == "diagnostic":
            from scripts.retrieval_backward_parity import run_diagnostics
            complete = run_diagnostics(output, allocation, device)
        else:
            raise ValueError(f"Unknown branch {arm}")
        allocation.finish(complete)
        atomic_json(output / "complete.json", {"arm": arm, "complete": True, "save_traces": True})
    finally:
        signal.signal(signal.SIGUSR1, prior_usr1)
        signal.signal(signal.SIGTERM, prior_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, "baselines", "diagnostic"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = os.environ["SLURM_JOB_ID"]
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise ValueError("Expected a numeric Slurm job and CUDA allocation")
    seed_everything(42)
    run_pipeline(args.arm, args.output_dir, job_id, int(os.environ.get("SLURM_RESTART_COUNT", "0")), torch.device("cuda"))


if __name__ == "__main__":
    main()
