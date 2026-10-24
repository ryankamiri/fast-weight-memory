"""Audit matched normalized training snapshots without changing their trained math."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.run import load_model
from evaluation.taal.decoder_diagnostic import load_replay_source
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.normalization_ablation import ARMS, run_fixed, run_generation
from evaluation.taal.prefix_store import checkpoint_fingerprint
from training.config import load_config
from utils.seed import seed_everything


AUDIT_CONFIG = Path("evaluation/configs/taal/capacity_d_distance_diversity_audit.yaml")
MATCHED_STEPS = (100, 400)


def validate_snapshot(checkpoint, arm, expected_step, recipe):
    """Require successful-step milestones while retaining skipped-attempt evidence."""
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    progress = metadata["progress"]
    if progress["step"] != expected_step:
        raise ValueError("Matched audit requires the exact successful optimizer-step budget")

    if type(progress["skipped_updates"]) is not int or progress["skipped_updates"] < 0:
        raise ValueError("Checkpoint skipped_updates must be a nonnegative integer")

    if metadata["reason"] != "matched optimizer-step snapshot":
        raise ValueError("Matched audit requires a fully validated milestone checkpoint")

    expected = recipe.to_dict()
    actual = metadata["training_config"]
    for section in ("model", "data", "loss", "optimizer", "scheduler", "training", "validation", "checkpoints"):
        if actual[section] != expected[section]:
            raise ValueError(f"Checkpoint training recipe differs in {section}")

    config = TaalQwen3Config.from_pretrained(checkpoint)
    flags = (config.memory_pre_rmsnorm, config.memory_mlp_output_layernorm)
    if flags != ARMS[arm] or config.memory_normalization_eps != 1e-6:
        raise ValueError("Serialized checkpoint normalization does not match the trained arm")

    for name in ("working_memory_size", "memory_dim", "memory_depth", "memory_chunk_size",
                 "memory_conv_kernel_size", "num_persistent_tokens", "max_persistent_kv_tokens"):
        if getattr(config, name) != getattr(recipe.model, name):
            raise ValueError(f"Serialized checkpoint differs in {name}")

    return metadata


def run_long_probe(checkpoint, arm, source_run, directory):
    """Keep trained normalization and replay the original failing prompt, then generate."""
    config = TaalQwen3Config.from_pretrained(checkpoint)
    layers = [layer for layer in (2, 3, 8, 12, 24, 27) if layer < config.num_hidden_layers]
    source = load_replay_source(source_run, 169, layers)
    if source.manifest["settings"]["model"].get("retain_task_prefix", False):
        raise ValueError("Expected the original unpinned source prompt")

    metadata = source.manifest["metadata"]
    tokens = source.example["input_ids"] + source.continuation
    directory.mkdir()
    manifest = {
        "arm": f"trained-{arm}", "checkpoint": str(checkpoint.resolve()),
        "fingerprint": checkpoint_fingerprint(checkpoint), "model_config": config.to_dict(),
        "source_run": str(source_run.resolve()), "source_fingerprint": source.manifest["fingerprint"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "optimizer_step": 400,
        "normalization": {"input_rmsnorm": config.memory_pre_rmsnorm,
                          "mlp_output_layernorm": config.memory_mlp_output_layernorm,
                          "eps": config.memory_normalization_eps, "affine": False},
        "prompt_length": source.example["prompt_length"], "forced_token_ids": source.continuation,
        "token_digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "layers": layers, "trace_start": max(0, source.example["prompt_length"] - 32),
        "execution_block_size": 32, "memory_read_scale": 1.0, "max_new_tokens": 256,
        "seed": 42, "save_traces": True, "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "interpretation": "Trained checkpoint, no normalization override. One diagnostic prompt, not a benchmark.",
    }
    write_report(directory / "manifest.json", manifest)
    write_report(directory / "example.json", source.example)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
    seed_everything(42)
    model = load_model(str(checkpoint), config).to("cuda").eval()
    manifest["attention_implementation"] = model.config._attn_implementation
    write_report(directory / "manifest.json", manifest)
    reports = {}
    for phase, runner in (("fixed", run_fixed), ("generation", run_generation)):
        print(f"TRAINED_NORMALIZATION_PROBE arm={arm} phase={phase}", flush=True)
        reports[phase] = runner(model, tokenizer, source, manifest, directory / phase)
        write_report(directory / "summary.json", reports)
        torch.cuda.empty_cache()

    return reports


def audit_training_run(run_dir, arm, source_run, output_dir):
    recipe_path = Path(f"training/configs/taal/qwen3_0_6b_norm_{arm}.yaml")
    recipe = load_config(recipe_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"arm": arm, "training_directory": str(run_dir), "save_traces": True, "snapshots": {}}
    write_report(output_dir / "summary.json", summary)
    for step in MATCHED_STEPS:
        checkpoint = run_dir / f"step-{step}"
        if not (checkpoint / "training_metadata.json").is_file():
            summary["snapshots"][str(step)] = {"status": "missing_matched_checkpoint"}
            write_report(output_dir / "summary.json", summary)
            continue

        parent = validate_snapshot(checkpoint, arm, step, recipe)
        progress = parent["progress"]
        warnings = []
        if progress["skipped_updates"] > 0:
            warnings.append(
                f"Checkpoint has {progress['skipped_updates']} skipped update attempts. "
                "Successful optimizer updates match, but training sample exposure does not "
                "match a run without skipped attempts."
            )
            print(f"NORMALIZATION_AUDIT_WARNING arm={arm} step={step}: {warnings[0]}", flush=True)

        record = {
            "status": "validated", "checkpoint": str(checkpoint), "progress": progress,
            "comparison_basis": "successful_optimizer_updates", "warnings": warnings,
            "attempted_updates": progress["step"] + progress["skipped_updates"],
        }
        summary["snapshots"][str(step)] = record
        write_report(output_dir / "summary.json", summary)
        snapshot_output = output_dir / f"step-{step}"
        snapshot_output.mkdir()
        write_report(snapshot_output / "training_progress.json", {
            "checkpoint": str(checkpoint), "progress": progress,
            "comparison_basis": record["comparison_basis"], "warnings": warnings,
            "attempted_updates": record["attempted_updates"],
        })
        subprocess.run([sys.executable, "-m", "utils.explorer_preflight", "--config", str(AUDIT_CONFIG),
                        "--checkpoint", str(checkpoint)], check=True)
        subprocess.run([sys.executable, "-m", "evaluation.taal.state_audit", "--config", str(AUDIT_CONFIG),
                        "--checkpoint", str(checkpoint), "--output-dir", str(snapshot_output)], check=True)
        record["status"] = "audit_completed"
        write_report(output_dir / "summary.json", summary)

    if summary["snapshots"]["400"]["status"] == "audit_completed":
        summary["long_probe"] = run_long_probe(run_dir / "step-400", arm, source_run, output_dir / "long-probe")
    else:
        summary["long_probe"] = {"status": "skipped_missing_step_400"}
    write_report(output_dir / "summary.json", summary)
    if any(record["status"] != "audit_completed" for record in summary["snapshots"].values()):
        raise RuntimeError("Matched evaluation incomplete; see summary.json. No final checkpoint substituted.")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-checkpoint matched audits require CUDA")

    audit_training_run(args.run_dir, args.arm, args.source_run, args.output_dir)


if __name__ == "__main__":
    main()
