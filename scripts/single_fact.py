"""Resumable acquisition, matched continuations and predeclared single-fact audits."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.models import load_model
from evaluation.shared.__main__ import load_rows, source_fingerprint, tokenizer_fingerprint
from evaluation.shared.backends import QwenBackend
from evaluation.shared.runner import run
from evaluation.storage import atomic_json, ensure_manifest, file_sha256, read_results
from evaluation.taal.prefix_store import checkpoint_fingerprint
from scripts.retrieval_acquisition import validate_checkpoint
from scripts.retrieval_exposure_audit import Allocation, token_diagnostics
from scripts.retrieval_parallel import needs_training
from scripts.retrieval_parallel_audit import load_baseline, release
from scripts.retrieval_training_segment import run_segment
from scripts.single_fact_preflight import check_inputs
from scripts.single_fact_scoring import SingleFactBenchmark, accuracy_summary, assignment_summary, paired_comparison
from utils.precision import precision_context
from utils.seed import seed_everything


ROOT = Path("data/single_fact")
MAX_ALLOCATIONS = 16
PHASE_ALLOCATIONS = {"baselines": 2, "acquisition": 6, "narrow": 4, "mixed": 4}


def prepare_recipes(output, arms=("acquisition", "narrow", "mixed")):
    common = yaml.safe_load(Path("training/configs/taal/single_fact.yaml").read_text())
    parent = Path("checkpoints") / f"{output.name}-acquisition" / "final"
    recipes = {}
    for arm in arms:
        config = json.loads(json.dumps(common))
        if arm != "acquisition":
            config["data"]["dataset_id"] = f"data/single_fact/{arm}"
            config["data"]["split_files"] = {split: f"data/single_fact/{arm}/{split}.jsonl" for split in ("train", "val")}
            config["data"]["manifest_path"] = f"data/single_fact/{arm}/manifest.json"
            config["training"].update(max_steps=512, initial_taal_gate=None,
                                      initialize_from_checkpoint=str(parent))
            config["scheduler"]["schedule_steps"] = 512
            config["checkpoints"]["save_steps"] = [128, 256, 512]
        config["wandb"]["name"] = f"{output.name}-{arm}"
        path = output / "recipes" / f"{arm}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError(f"Stage recipe changed: {path}")
        # JSON's 1e-06 is a string to PyYAML's loader; YAML preserves numeric types.
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        recipes[arm] = path
    return recipes


def generation(model, tokenizer, provenance, rows, output, allocation, read_scale=1.0):
    precision = getattr(model.config, "execution_precision", "bf16")
    if provenance.get("precision", precision) != precision:
        raise ValueError("Generation precision disagrees with its provenance")

    benchmark = SingleFactBenchmark()
    examples = [benchmark.prepare(row, tokenizer) for row in rows]
    backend = QwenBackend(model, tokenizer, provenance,
                         {"do_sample": False, "max_new_tokens": 8, "eos_token_id": tokenizer.eos_token_id},
                         memory_read_scale=read_scale)
    device = next(model.parameters()).device
    with precision_context(device, precision):
        # Record the active policy, rather than inferring execution from parameter dtype.
        execution = {
            "precision": precision, "device_type": device.type,
            "autocast_enabled": torch.is_autocast_enabled(device.type),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        ensure_manifest(output / "execution.json", execution)
        summary = run(backend, benchmark, examples, output,
                      {"manifest_sha256": file_sha256(ROOT / "manifest.json")},
                      seed=42, save_traces=True, should_stop=allocation.stopped)
    if summary is None:
        return False
    predictions = read_results(output / "predictions.jsonl", "example_id")
    atomic_json(output / "accuracy.json", accuracy_summary(rows, predictions))
    atomic_json(output / "assignments.json", assignment_summary(rows, predictions))
    return True


def audit_checkpoint(recipe, checkpoint, output, allocation, device, final, precision=None):
    metadata = validate_checkpoint(recipe, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = TaalQwen3Config.from_pretrained(checkpoint)
    if precision is not None:
        if precision not in ("bf16", "fp32"):
            raise ValueError("Expected bf16 or fp32 evaluation precision")

        config.execution_precision = precision
    model = load_model(str(checkpoint), config, dtype=torch.float32).to(device)
    provenance = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "tokenizer": str(checkpoint), "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "source_sha256": source_fingerprint(), "precision": config.execution_precision, "progress": metadata["progress"],
    }
    try:
        # Loss-only resumes must also refuse a different checkpoint or precision.
        ensure_manifest(output / "checkpoint.json", provenance)
        for cohort in ("narrow", "val"):
            rows = load_rows(ROOT / f"{cohort}.jsonl")
            if not generation(model, tokenizer, provenance, rows, output / cohort, allocation):
                return False
            # Label-only and EOS likelihood remain distinct from free-answer accuracy.
            if not token_diagnostics(model, rows, output / f"{cohort}-loss", "loss", allocation):
                return False
        if final:
            rows = load_rows(ROOT / "test.jsonl")
            if not generation(model, tokenizer, provenance, rows, output / "test", allocation):
                return False
            # A matched whole-episode reads-off slice; query-only controls live in the causal audit.
            subset = [row for row in rows if int(row["metadata"]["family_id"].split("-")[2]) % 128 < 16]
            if len(subset) != 64:
                raise ValueError("Expected 64 fixed reads-off questions")
            if not generation(model, tokenizer, provenance, subset, output / "test-reads-off", allocation, 0.0):
                return False
        return True
    finally:
        del model
        release()


def run_controls(checkpoint, output, allocation):
    if (output / "complete.json").is_file():
        return True
    command = [sys.executable, "-m", "evaluation.taal.state_audit",
               "--config", "evaluation/configs/taal/single_fact.yaml",
               "--checkpoint", str(checkpoint), "--output-dir", str(output)]
    process = None

    def stop(signum, frame):
        allocation.signal(signum, frame)
        if process is not None and process.poll() is None:
            process.terminate()

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        if allocation.stopped():
            return False
        process = subprocess.Popen(command)
        if allocation.stopped():
            process.terminate()
        code = process.wait()
        if allocation.stopped():
            return False
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    records = read_results(output / "results.jsonl", "evaluation_id")
    trace_index = json.loads((output / "memory_traces/manifest.json").read_text())["episodes"]
    if len(records) != 384 or set(records) != set(trace_index):
        raise ValueError("Causal audit requires 64 episodes x six complete score/trace conditions")
    atomic_json(output / "complete.json", {"rows": 384, "independent_recipient_scenarios": 32,
                "precision": "float32 parameters, legacy state-audit execution; separate from BF16 free-generation scores"})
    return True


def compare(output):
    rows = load_rows(ROOT / "test.jsonl")
    summaries = {}
    for arm in ("narrow", "mixed"):
        predictions = read_results(output / arm / "audit/test/predictions.jsonl", "example_id")
        for reference in ("u_full", "u_sw128"):
            baseline = read_results(output / reference / "test/predictions.jsonl", "example_id")
            summaries[f"{arm}-vs-{reference}"] = paired_comparison(rows, predictions, baseline)
    narrow = read_results(output / "narrow/audit/test/predictions.jsonl", "example_id")
    mixed = read_results(output / "mixed/audit/test/predictions.jsonl", "example_id")
    summaries["mixed-vs-narrow"] = paired_comparison(rows, mixed, narrow)
    atomic_json(output / "comparisons.json", summaries)


def finish_comparison(output):
    # Independent jobs publish their own receipts first. The last finisher
    # computes the comparison under a lock, without occupying a fifth Slurm slot.
    with (output / "comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((output / "jobs" / phase / "complete.json").is_file() for phase in PHASE_ALLOCATIONS):
            return
        if not (output / "complete.json").is_file():
            compare(output)
            atomic_json(output / "complete.json", {"complete": True, "save_traces": True})


def pipeline(output, job_id, restart, device, phase="all"):
    if phase != "all" and phase not in PHASE_ALLOCATIONS:
        raise ValueError(f"Unknown single-fact phase: {phase}")
    check_inputs()
    output.mkdir(parents=True, exist_ok=True)
    budget = MAX_ALLOCATIONS if phase == "all" else PHASE_ALLOCATIONS[phase]
    job_output = output if phase == "all" else output / "jobs" / phase
    ensure_manifest(job_output / "job.json", {"job_id": job_id, "max_allocations": budget,
                    "hours_per_allocation": 2, "save_traces": True,
                    "selection": "fixed final checkpoints; no test-guided tuning or checkpoint selection"})
    if phase == "all":
        arms = ("acquisition", "narrow", "mixed")
    elif phase == "baselines":
        arms = ()
    else:
        arms = (phase,)
    if phase in ("narrow", "mixed"):
        # afterok is the scheduler dependency; these receipts also prevent a
        # manually launched continuation from accepting a partial parent.
        if not (output / "jobs/acquisition/complete.json").is_file():
            raise ValueError("Shared acquisition has not completed")
        validate_checkpoint(output / "recipes/acquisition.yaml",
                            Path("checkpoints") / f"{output.name}-acquisition" / "final")
    recipes = prepare_recipes(output, arms)
    allocation = Allocation(budget)
    baseline_arms = ("u_full", "u_sw128") if phase in ("all", "baselines") else ()
    previous = {signum: signal.signal(signum, allocation.signal) for signum in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        for arm in baseline_arms:
            marker = output / arm / "complete.json"
            if marker.is_file():
                continue
            if allocation.stopped():
                allocation.finish(False)
            model, tokenizer, provenance = load_baseline(arm, device, output / arm, load_rows(ROOT / "val.jsonl")[0])
            try:
                complete = generation(model, tokenizer, provenance, load_rows(ROOT / "test.jsonl"),
                                      output / arm / "test", allocation)
            finally:
                del model
                release()
            allocation.finish(complete)
            atomic_json(marker, {"complete": True, "test_questions": 512})
        for arm in arms:
            run_id = f"{output.name}-{arm}"
            checkpoint = Path("checkpoints") / run_id / "final"
            target = 768 if arm == "acquisition" else 512
            start_path = output / arm / "allocation-start.json"
            if not start_path.exists():
                atomic_json(start_path, {"restart": restart})
            start = json.loads(start_path.read_text())["restart"]
            if allocation.stopped():
                allocation.finish(False)
            if needs_training(checkpoint, target):
                training_budget = min(6, budget - start)
                warned = run_segment(recipes[arm], run_id, job_id, restart - start, training_budget)
                allocation.timed_out = allocation.timed_out or warned
            if allocation.stopped():
                allocation.finish(False)
            marker = output / arm / "audit/complete.json"
            if not marker.exists():
                complete = audit_checkpoint(recipes[arm], checkpoint, output / arm / "audit", allocation, device, arm != "acquisition")
                allocation.finish(complete)
                atomic_json(marker, {"complete": True, "step": target})
            if arm != "acquisition":
                complete = run_controls(checkpoint, output / arm / "controls", allocation)
                allocation.finish(complete)
        if phase == "all":
            compare(output)
            atomic_json(output / "complete.json", {"complete": True, "save_traces": True})
        else:
            atomic_json(job_output / "complete.json", {"complete": True, "save_traces": True})
            finish_comparison(output)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", *PHASE_ALLOCATIONS], default="all")
    args = parser.parse_args()
    job_id = os.environ["SLURM_JOB_ID"]
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise ValueError("Expected a CUDA Slurm allocation")
    budget = MAX_ALLOCATIONS if args.phase == "all" else PHASE_ALLOCATIONS[args.phase]
    if restart >= budget:
        raise ValueError("Pipeline allocation budget exhausted")
    seed_everything(42)
    pipeline(args.output_dir, job_id, restart, torch.device("cuda"), args.phase)


if __name__ == "__main__":
    main()
