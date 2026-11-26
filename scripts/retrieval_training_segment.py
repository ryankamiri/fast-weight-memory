"""Run one allocation of a resumable acquisition job; release audits only at budget."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from training.config import load_config


def checkpoint_progress(checkpoint):
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if not (checkpoint / "trainer_state.pt").is_file():
        raise ValueError("Continuation requires a full trainer checkpoint")
    return metadata


def finish_segment(checkpoint, previous_step, target_step, timed_out, restart_count, max_allocations=6):
    metadata = checkpoint_progress(checkpoint)
    step = metadata["progress"]["step"]
    if step == target_step:
        return "complete"
    if step > target_step or step <= previous_step:
        raise ValueError("Training did not make valid forward progress")
    if not timed_out or metadata["reason"] != "signal":
        raise ValueError("Incomplete training without the allocation warning; refusing success")
    if restart_count >= max_allocations - 1:
        raise ValueError(f"{max_allocations} allocations exhausted; inspect progress before continuing")
    return "requeue"


def run_segment(config_path, run_id, job_id, restart_count, max_allocations=6):
    config = load_config(config_path)
    checkpoint = Path(config.checkpoints.output_dir) / run_id / "final"
    previous_step = 0
    command = [sys.executable, "-m", "training.train", "--config", str(config_path)]
    if checkpoint.exists():
        previous_step = checkpoint_progress(checkpoint)["progress"]["step"]
        command.extend(["--resume", str(checkpoint)])
    elif restart_count:
        raise ValueError("Restarted job has no final checkpoint; refusing a fresh run")

    # USR1 is the walltime warning. TERM is cancellation and must never restart a job.
    timed_out = False
    cancelled = False
    process = None

    def forward_signal(signum, frame):
        nonlocal timed_out, cancelled
        if signum == signal.SIGUSR1:
            timed_out = True
        else:
            cancelled = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    prior_usr1 = signal.signal(signal.SIGUSR1, forward_signal)
    prior_term = signal.signal(signal.SIGTERM, forward_signal)
    try:
        if cancelled or timed_out:
            raise RuntimeError("Allocation interrupted before trainer startup")
        environment = dict(os.environ, CHECKPOINT_RUN_ID=run_id)
        print(f"RETRIEVAL_SEGMENT job={job_id} restart={restart_count} resume_step={previous_step} target={config.training.max_steps}", flush=True)
        process = subprocess.Popen(command, env=environment)
        if cancelled or timed_out:
            process.send_signal(signal.SIGTERM)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        if cancelled:
            raise RuntimeError("Training cancelled; no automatic continuation")
        action = finish_segment(checkpoint, previous_step, config.training.max_steps, timed_out, restart_count, max_allocations)
        print(f"RETRIEVAL_SEGMENT action={action}", flush=True)
        if action == "requeue":
            # Requeue the still-running job: afterok remains blocked on this same ID.
            # Exit nonzero if Slurm rejects it; never turn an incomplete run into success.
            subprocess.run(["scontrol", "requeue", job_id], check=True, timeout=60)
            raise SystemExit(75)
        return timed_out
    finally:
        signal.signal(signal.SIGUSR1, prior_usr1)
        signal.signal(signal.SIGTERM, prior_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    job_id = os.environ["SLURM_JOB_ID"]
    if not job_id.isdigit():
        raise ValueError("Expected a numeric Slurm job ID")
    run_segment(args.config, args.run_id, job_id, int(os.environ.get("SLURM_RESTART_COUNT", "0")))


if __name__ == "__main__":
    main()
