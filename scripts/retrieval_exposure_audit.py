"""Resumable acquisition diagnostics and matched repetition/coverage audits."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import signal
import subprocess

import torch
from transformers import AutoTokenizer

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.models import load_model
from evaluation.shared.__main__ import load_rows, source_fingerprint, tokenizer_fingerprint
from evaluation.shared.backends import QwenBackend
from evaluation.shared.benchmarks import ControlledBenchmark, PreparedControlledBenchmark
from evaluation.shared.retrieval_diagnostics import path_parity, score_answer, selection_counts
from evaluation.shared.runner import run
from evaluation.storage import append_result, atomic_json, ensure_manifest, file_sha256, read_results
from evaluation.taal.prefix_store import checkpoint_fingerprint
from scripts.retrieval_acquisition import validate_checkpoint
from training.engine import precision_context
from training.train import configure_trainable_parameters
from utils.seed import seed_everything


COHORT = Path("data/retrieval_exposure")
OLD_ARMS = {
    "A": ("lr3e5", "11868529"),
    "B": ("lr1e4", "11872832"),
    "C": ("lr3e4", "11872834"),
}


class Allocation:
    def __init__(self):
        self.timed_out = False
        self.cancelled = False

    def signal(self, signum, frame):
        if signum == signal.SIGUSR1:
            self.timed_out = True
        else:
            self.cancelled = True

    def stopped(self):
        return self.timed_out or self.cancelled

    def finish(self, complete):
        if self.cancelled:
            raise RuntimeError("Audit cancelled; refusing automatic requeue")
        if complete:
            return
        if not self.timed_out:
            raise RuntimeError("Incomplete audit without allocation warning")
        if int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= 5:
            raise RuntimeError("Six allocations exhausted; inspect partial diagnostics")
        job_id = os.environ["SLURM_JOB_ID"]
        if not job_id.isdigit():
            raise ValueError("Expected numeric Slurm ID")
        subprocess.run(["scontrol", "requeue", job_id], check=True, timeout=60)
        raise SystemExit(75)


def checked_rows(name):
    selection = json.loads((COHORT / "selection.json").read_text())
    path = COHORT / f"{name}.jsonl"
    rows = load_rows(path)
    expected = selection["files"][name]
    if file_sha256(path) != expected["sha256"] or len(rows) != expected["rows"]:
        raise ValueError("Diagnostic cohort changed")
    return rows


def load_checkpoint(arm, train_job, output, device):
    if arm in OLD_ARMS:
        name, root_job = OLD_ARMS[arm]
        suffix = "" if name == "lr1e4" else f"_{name}"
        recipe = Path(f"training/configs/taal/retrieval_acquisition{suffix}.yaml")
        checkpoint = Path(f"checkpoints/taal-retrieval-{name}-{root_job}/final")
    else:
        if not train_job or not train_job.isdigit():
            raise ValueError("New-arm audit requires its training job ID")
        recipe = Path(f"training/configs/taal/retrieval_exposure_{arm}.yaml")
        checkpoint = Path(f"checkpoints/taal-exposure-{arm}-{train_job}/final")
    metadata = validate_checkpoint(recipe, checkpoint)
    if arm in OLD_ARMS:
        rows = checked_rows(f"{arm}-consumed")
        progress = metadata["progress"]
        expected_skips = int(arm == "C")
        if progress["examples_seen"] != len(rows) or progress["skipped_updates"] != expected_skips:
            raise ValueError("Checkpoint exposure differs from diagnostic reconstruction")
        if progress["tokens_seen"] != sum(len(row["input_ids"]) for row in rows):
            raise ValueError("Reconstructed exposure does not match token counter")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    manifest = json.loads((COHORT / "repeat/manifest.json").read_text())
    if json.loads(tokenizer.backend_tokenizer.to_str()) != manifest["tokenizer_backend"]:
        raise ValueError("Cohort tokenizer does not match checkpoint")
    if tokenizer.special_tokens_map != manifest["special_tokens"]:
        raise ValueError("Cohort special tokens do not match checkpoint")
    provenance = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "tokenizer": str(checkpoint),
        "source_sha256": source_fingerprint(), "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "dtype": "float32", "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
        "device": str(device), "torch_version": torch.__version__,
    }
    ensure_manifest(output / "checkpoint.json", provenance | {"progress": metadata["progress"]})
    model = load_model(str(checkpoint), TaalQwen3Config.from_pretrained(checkpoint), dtype=torch.float32)
    model.to(device)
    configure_trainable_parameters(model, "taal_only")
    return model, tokenizer, provenance


def token_diagnostics(model, rows, output, mode, allocation):
    output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output / "examples.json", rows)
    path = output / "records.jsonl"
    records = read_results(path, "example_id")
    for row in rows:
        if allocation.stopped():
            return False
        if row["example_id"] in records:
            continue
        if mode == "parity":
            metrics = path_parity(model, row)
        else:
            metrics = score_answer(model, row)
        result = {"example_id": row["example_id"], **metrics}
        append_result(path, result)
        records[row["example_id"]] = result
        print(json.dumps({"phase": mode, **result}), flush=True)
    if set(records) != {row["example_id"] for row in rows}:
        raise ValueError("Token diagnostic has unexpected or missing rows")
    summary = {"examples": len(rows)}
    if mode == "parity":
        summary["passed"] = all(record["passed"] for record in records.values())
    else:
        for metric in ("label_nll", "eos_nll", "combined_nll"):
            summary[metric] = sum(record[metric] for record in records.values()) / len(rows)
        summary["averaging"] = "mean of per-episode means; label/EOS also retained per token"
        summary["execution"] = "full sequence, no cache, teacher-forced label and EOS"
        for metric in ("first_label_token_correct", "all_label_tokens_correct", "label_and_eos_correct"):
            summary[metric] = {
                "correct": sum(record[metric] for record in records.values()),
                "total": len(rows),
            }
    atomic_json(output / "summary.json", summary)
    return True


def generation_audit(model, tokenizer, provenance, rows, output, allocation, read_scale=1.0, workshop=False):
    benchmark = PreparedControlledBenchmark() if workshop else ControlledBenchmark()
    examples = [benchmark.prepare(row, tokenizer) for row in rows]
    backend = QwenBackend(model, tokenizer, provenance,
                         {"do_sample": False, "max_new_tokens": 8, "eos_token_id": tokenizer.eos_token_id},
                         memory_read_scale=read_scale)
    backend.validate([example.prompt for example in examples], save_traces=True)
    device = next(model.parameters()).device
    with precision_context(device, getattr(model.config, "execution_precision", "bf16")):
        summary = run(backend, benchmark, examples, output,
                      {"cohort": str(output.name), "selection_sha256": file_sha256(COHORT / "selection.json")},
                      seed=42, save_traces=True, should_stop=allocation.stopped)
    if summary is not None and not workshop and all("family_id" in row["metadata"] for row in rows):
        predictions = read_results(output / "predictions.jsonl", "example_id")
        atomic_json(output / "selection.json", selection_counts(rows, predictions))
    return summary is not None


def audit_arm(phase, arm, train_job, output, allocation, device):
    model, tokenizer, provenance = load_checkpoint(arm, train_job, output, device)
    try:
        if phase == "parity":
            rows = checked_rows(f"{arm}-parity")
            if not token_diagnostics(model, rows, output / "parity", "parity", allocation):
                return False
            if not generation_audit(model, tokenizer, provenance, rows, output / "generation", allocation):
                return False
            summary = json.loads((output / "parity/summary.json").read_text())
            if not summary["passed"]:
                raise ValueError(f"{arm} parity screen failed; inspect the recorded differences")
            return True

        if phase == "diagnostic":
            consumed = checked_rows(f"{arm}-consumed")
            # C's example 355 was observed, but its infinite-gradient update was skipped.
            receipt = {"consumed": len(consumed), "successful_updates": 400,
                       "skipped_example_ids": [consumed[354]["example_id"]] if arm == "C" else []}
            ensure_manifest(output / "exposure.json", receipt)
            if not token_diagnostics(model, checked_rows(f"{arm}-loss"), output / "loss", "loss", allocation):
                return False
            # Keep logical halves independently resumable inside one Slurm job.
            middle = (len(consumed) + 1) // 2
            for shard, rows in enumerate((consumed[:middle], consumed[middle:])):
                if not generation_audit(model, tokenizer, provenance, rows, output / f"consumed-{shard}", allocation):
                    return False
            predictions = {}
            for shard in range(2):
                predictions.update(read_results(output / f"consumed-{shard}/predictions.jsonl", "example_id"))
            atomic_json(output / "consumed-selection.json", selection_counts(consumed, predictions))
        else:
            fit = checked_rows("common-fit")
            if not token_diagnostics(model, fit, output / "fit-loss", "loss", allocation):
                return False
            if not generation_audit(model, tokenizer, provenance, fit, output / "fit", allocation):
                return False

        workshop = load_rows(Path("data/taal_workshop/evaluation.jsonl"))
        if Counter(row["condition"] for row in workshop) != Counter(reference=16, selection=16, distance=8, filler=8, position=8, combined=8):
            raise ValueError("Historical workshop slice counts changed")
        if not generation_audit(model, tokenizer, provenance, workshop, output / "workshop", allocation, workshop=True):
            return False
        if phase == "audit":
            val = load_rows(COHORT / arm / "val.jsonl")
            if not token_diagnostics(model, val, output / "val-loss", "loss", allocation):
                return False
            for read_scale, name in ((1.0, "val"), (0.0, "val-reads-off")):
                if not generation_audit(model, tokenizer, provenance, val, output / name, allocation, read_scale):
                    return False
        return True
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["parity", "diagnostic", "audit"], required=True)
    parser.add_argument("--arm", choices=["A", "B", "C", "repeat", "broad"])
    parser.add_argument("--train-job")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "diagnostic" and args.arm not in OLD_ARMS:
        parser.error("Diagnostics require A, B, or C")
    if args.phase == "audit" and args.arm not in ("repeat", "broad"):
        parser.error("Final audits require repeat or broad")
    seed_everything(42)
    allocation = Allocation()
    signal.signal(signal.SIGUSR1, allocation.signal)
    signal.signal(signal.SIGTERM, allocation.signal)
    arms = list(OLD_ARMS) if args.phase == "parity" else [args.arm]
    complete = True
    for arm in arms:
        if allocation.stopped() or not audit_arm(args.phase, arm, args.train_job,
                                               args.output_dir / arm, allocation, torch.device(args.device)):
            complete = False
            break
    if complete:
        atomic_json(args.output_dir / "complete.json", {"phase": args.phase, "arms": arms, "passed": True})
    allocation.finish(complete)


if __name__ == "__main__":
    main()
