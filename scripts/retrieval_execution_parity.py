"""Locate full/cached CUDA differences on two fixed, previously failing examples."""

import argparse
from dataclasses import asdict
import gc
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import traceback

import torch

from evaluation.shared.retrieval_diagnostics import answer_loss, compare_logits
from evaluation.storage import atomic_json, ensure_manifest, file_sha256
from evaluation.taal.execution_parity import (
    PRECISIONS, compare_execution, execute_path, tensor_difference,
)
from scripts.retrieval_exposure_audit import Allocation, checked_rows, generation_audit, load_checkpoint
from utils.seed import seed_everything


EXAMPLE_IDS = (
    "train-family000184-assignment1-order1-query1",
    "train-family000207-assignment0-order1-query0",
)


def diagnostic_rows(arm):
    available = {}
    for row in checked_rows(f"{arm}-parity"):
        available[row["example_id"]] = row
    rows = []
    for example_id in EXAMPLE_IDS:
        if example_id not in available:
            raise ValueError(f"Fixed example missing from {arm} parity cohort: {example_id}")
        rows.append(available[example_id])
    return rows


def prediction_details(logits, labels, tokenizer):
    values, indices = logits.topk(2, dim=-1)
    tokens = []
    for position in range(len(labels)):
        first, second = indices[position].tolist()
        tokens.append({
            "answer_index": position, "target_id": int(labels[position]),
            "target_text": tokenizer.decode([int(labels[position])]),
            "top_ids": [first, second],
            "top_text": [tokenizer.decode([first]), tokenizer.decode([second])],
            "top_logits": values[position].tolist(),
            "top_two_margin": float(values[position, 0] - values[position, 1]),
        })
    return {"loss": asdict(answer_loss(logits, labels)), "tokens": tokens}


def save_path(output, name, result, labels, tokenizer):
    output.mkdir(parents=True, exist_ok=True)
    payload = {"logits": result.logits, "labels": labels, "tokens_seen": result.tokens_seen}
    torch.save(payload, output / f"{name}-logits.pt")
    report = prediction_details(result.logits, labels, tokenizer)
    report.update({"settings": result.settings, "fused_loss": result.fused_loss,
                   "logits_sha256": file_sha256(output / f"{name}-logits.pt")})
    atomic_json(output / f"{name}.json", report)
    return report


def compare_kv(reference, actual):
    if set(reference) != set(actual):
        raise ValueError("Final KV layer coverage differs")

    result = {}
    for layer in reference:
        left, right = reference[layer], actual[layer]
        if not torch.equal(left["positions"], right["positions"]):
            raise ValueError("Final KV retained positions differ")
        result[layer] = {"positions": left["positions"].tolist(),
                         "keys": tensor_difference(left["keys"], right["keys"]),
                         "values": tensor_difference(left["values"], right["values"])}
    return result


def run_case(model, tokenizer, row, precision, read_scale, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "error.json").unlink(missing_ok=True)
    labels = torch.tensor(row["input_ids"][len(row["prompt_input_ids"]):])
    reference = execute_path(model, row, "full_eval", precision, read_scale)
    reference_report = save_path(output, "full_eval", reference, labels, tokenizer)
    paths = ["full_train", "full_cache", "blocked", "blocked_alternate"]
    comparisons = {}
    cached_kv = {}
    blocked_logits = None
    for path in paths:
        actual = execute_path(model, row, path, precision, read_scale)
        report = save_path(output, path, actual, labels, tokenizer)
        comparison = compare_execution(reference, actual, labels)
        comparison["metrics"]["reference_path"] = "full_eval"
        comparison["metrics"]["candidate_path"] = path
        if path == "full_train":
            comparison["metrics"]["actual_fused_vs_explicit"] = abs(
                actual.fused_loss - report["loss"]["combined_nll"])
        if path == "full_cache":
            cached_kv = actual.kv
        if path.startswith("blocked"):
            comparison["metrics"]["kv_vs_full_cache"] = compare_kv(cached_kv, actual.kv)
        if path == "blocked":
            blocked_logits = actual.logits
        with gzip.open(output / f"{path}-stages.json.gz", "wt") as stream:
            json.dump(comparison["stages"], stream, allow_nan=False)
        torch.save(comparison["snapshots"], output / f"{path}-first-differences.pt")
        atomic_json(output / f"{path}-comparison.json", comparison["metrics"])
        comparisons[path] = comparison["metrics"]
        print(json.dumps({"phase": "path", "example": row["example_id"],
                          "precision": precision, "read_scale": read_scale, "path": path,
                          "max_abs": comparison["metrics"]["max_abs_logit_difference"],
                          "first": comparison["metrics"]["first_tolerance_difference"]}), flush=True)
        del actual, comparison

    # Check that synchronization/copying in our hooks did not change the measured logits.
    observer_checks = {}
    for path, observed_logits in (("full_eval", reference.logits), ("blocked", blocked_logits)):
        unobserved = execute_path(model, row, path, precision, read_scale, record=False)
        save_path(output, f"{path}_unobserved", unobserved, labels, tokenizer)
        observer_checks[path] = compare_logits(
            observed_logits, unobserved.logits, labels,
            answer_loss(observed_logits, labels).combined_nll)
    summary = {
        "example_id": row["example_id"], "precision": precision, "read_scale": read_scale,
        "full_eval": reference_report, "comparisons": comparisons,
        "observer_checks": observer_checks,
        "save_traces": True,
        "trace_kind": "all text positions and layers at observed modules; final memory/KV comparisons",
        "interpretation": "completed measurements; discrepancies do not release training jobs",
    }
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "summary.json":
            artifacts[path.name] = file_sha256(path)
    summary["artifacts"] = artifacts
    atomic_json(output / "summary.json", summary)
    return summary


def completed_case(output):
    path = output / "summary.json"
    if not path.exists():
        return False

    summary = json.loads(path.read_text())
    for name, checksum in summary["artifacts"].items():
        artifact = output / name
        if not artifact.is_file() or file_sha256(artifact) != checksum:
            raise ValueError(f"Incomplete or changed diagnostic artifact: {artifact}")
    return True


def run_diagnostic(model, tokenizer, provenance, rows, output, allocation):
    ensure_manifest(output / "examples.json", rows)
    ensure_manifest(output / "design.json", {
        "precisions": list(PRECISIONS), "examples": [row["example_id"] for row in rows],
        "reads_off_precisions": ["bf16_auto", "fp32_math"], "save_traces": True,
        "optimizer_updates": False, "full_sequence_includes_eos": True,
        "source_files": {
            "execution_parity": file_sha256(Path("evaluation/taal/execution_parity.py")),
            "runner": file_sha256(Path(__file__)),
        },
    })
    for row in rows:
        for precision in PRECISIONS:
            scales = [1.0]
            if precision in ("bf16_auto", "fp32_math"):
                scales.append(0.0)
            for scale in scales:
                if allocation.stopped():
                    return False
                directory = output / row["example_id"] / precision / f"reads-{int(scale)}"
                if completed_case(directory):
                    continue
                try:
                    run_case(model, tokenizer, row, precision, scale, directory)
                except Exception:
                    atomic_json(directory / "error.json", {"error": traceback.format_exc()})
                    raise
                gc.collect()
                if next(model.parameters()).is_cuda:
                    torch.cuda.empty_cache()
    # Normal greedy exports remain distinct from the forced-token diagnostic traces.
    return generation_audit(model, tokenizer, provenance, rows, output / "generation", allocation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA allocation has no usable GPU")

    allocation = Allocation()
    signal.signal(signal.SIGUSR1, allocation.signal)
    signal.signal(signal.SIGTERM, allocation.signal)
    rows = diagnostic_rows(args.arm)
    model, tokenizer, provenance = load_checkpoint(args.arm, None, args.output_dir, device)
    if model.config._attn_implementation != "sdpa":
        raise ValueError("The backend comparison requires the original SDPA attention path")
    receipt = {
        "arm": args.arm, "job": os.environ.get("SLURM_JOB_ID"),
        "restart": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "node": os.environ.get("SLURMD_NODENAME"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "attention_implementation": model.config._attn_implementation,
    }
    atomic_json(args.output_dir / f"runtime-{receipt['restart']}.json", receipt)
    print(json.dumps({"phase": "loaded", **receipt}), flush=True)
    complete = run_diagnostic(model, tokenizer, provenance, rows, args.output_dir, allocation)
    if complete:
        atomic_json(args.output_dir / "complete.json", {
            "arm": args.arm, "examples": len(rows), "cases": len(rows) * 6,
            "completed": True, "training_released": False,
        })
    allocation.finish(complete)


if __name__ == "__main__":
    main()
