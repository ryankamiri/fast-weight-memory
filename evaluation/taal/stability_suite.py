"""Matched stability training, checkpoint audits and long-context probes."""

import argparse
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from transformers import AutoTokenizer

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.run import load_model
from evaluation.taal.decoder_diagnostic import load_replay_source
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.normalization_ablation import run_fixed, run_generation
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.stability_observation import ReadRoundingRecorder
from training.config import load_config
from utils.seed import seed_everything


TRAINING_ARMS = {
    "baseline": {},
    "output_ln": {"model": {"memory_mlp_output_layernorm": True}},
    "read_ln": {"model": {"memory_read_layernorm": True}},
    "small_write_ln": {
        "model": {
            "memory_mlp_output_layernorm": True,
            "memory_initial_write_strength": 0.01,
        }
    },
    "scale_ln": {
        "model": {
            "memory_mlp_output_layernorm": True,
            "memory_norm_learned_scale": True,
        }
    },
    "residual_ln": {
        "model": {"memory_mlp_output_layernorm": True, "memory_norm_residual": True}
    },
    "scale_residual_ln": {
        "model": {
            "memory_mlp_output_layernorm": True,
            "memory_norm_learned_scale": True,
            "memory_norm_residual": True,
        }
    },
    "clip_ln": {
        "model": {"memory_mlp_output_layernorm": True},
        "optimizer": {"gate_max_grad_norm": 2.0},
    },
    "lr_ln": {
        "model": {"memory_mlp_output_layernorm": True},
        "optimizer": {"gate_lr_multiplier": 2.0},
    },
    "clip_lr_ln": {
        "model": {"memory_mlp_output_layernorm": True},
        "optimizer": {"gate_lr_multiplier": 2.0, "gate_max_grad_norm": 2.0},
    },
}
TRAINING_GROUPS = {
    "references": ["baseline", "output_ln"],
    "write_objective": ["read_ln", "small_write_ln"],
    "norm_structure": ["scale_ln", "residual_ln", "scale_residual_ln"],
    "gate_optimizer": ["clip_ln", "lr_ln", "clip_lr_ln"],
}
BASE_RECIPE = Path("training/configs/taal/qwen3_0_6b_norm_baseline.yaml")
AUDIT_CONFIG = Path("evaluation/configs/taal/capacity_d_distance_diversity_audit.yaml")


def arm_config_path(arm):
    return Path(f"training/configs/taal/qwen3_0_6b_stability_{arm}.yaml")


def write_recipes():
    baseline = yaml.safe_load(BASE_RECIPE.read_text())
    for arm, changes in TRAINING_ARMS.items():
        settings = copy.deepcopy(baseline)
        for section, values in changes.items():
            settings[section].update(values)
        settings["wandb"]["name"] = f"taal-stability-{arm}"
        arm_config_path(arm).write_text(yaml.safe_dump(settings, sort_keys=False))
        load_config(arm_config_path(arm))


def normalization_settings(config):
    return {
        name: getattr(config, name)
        for name in (
            "memory_pre_rmsnorm",
            "memory_mlp_output_layernorm",
            "memory_read_layernorm",
            "memory_norm_learned_scale",
            "memory_norm_residual",
            "memory_normalization_eps",
        )
    }


def long_probe(checkpoint, source_run, directory):
    config = TaalQwen3Config.from_pretrained(checkpoint)
    layers = [2, 3, 8, 12, 24, 27]
    source = load_replay_source(source_run, 169, layers)
    metadata = source.manifest["metadata"]
    directory.mkdir(parents=True)
    manifest = {
        "arm": directory.parent.name,
        "checkpoint": str(checkpoint.resolve()),
        "fingerprint": checkpoint_fingerprint(checkpoint),
        "save_traces": True,
        "source_fingerprint": source.manifest["fingerprint"],
        "model_config": config.to_dict(),
        "normalization": normalization_settings(config),
        "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "layers": layers,
        "trace_start": source.example["prompt_length"] - 32,
        "execution_block_size": 32,
        "max_new_tokens": 256,
        "source_run": str(source_run),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_report(directory / "manifest.json", manifest)
    write_report(directory / "example.json", source.example)
    seed_everything(42)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"], revision=metadata["tokenizer_revision"]
    )
    model = load_model(str(checkpoint), config).to("cuda").eval()
    reports = {}
    for phase, runner in [("fixed", run_fixed), ("generation", run_generation)]:
        with ReadRoundingRecorder(model, directory / f"{phase}-rounding.jsonl"):
            reports[phase] = runner(
                model, tokenizer, source, manifest, directory / phase
            )
        write_report(directory / "summary.json", reports)
    del model
    torch.cuda.empty_cache()
    return reports


def validate_checkpoint(checkpoint, recipe, step):
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if (
        metadata["progress"]["step"] != step
        or metadata["reason"] != "matched optimizer-step snapshot"
    ):
        raise ValueError("Require exact validated optimizer-step snapshot")

    expected = recipe.to_dict()
    for section in (
        "model",
        "data",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "validation",
        "checkpoints",
    ):
        if metadata["training_config"][section] != expected[section]:
            raise ValueError(f"Checkpoint recipe mismatch: {section}")

    config = TaalQwen3Config.from_pretrained(checkpoint)
    for name, value in expected["model"].items():
        if (
            name not in ("model_id", "revision", "architecture")
            and getattr(config, name) != value
        ):
            raise ValueError(f"Serialized architecture mismatch: {name}")
    return metadata["progress"]


def audit_arm(arm, run_dir, source_run, output):
    output.mkdir(parents=True, exist_ok=False)
    recipe = load_config(arm_config_path(arm))
    summary = {"arm": arm, "save_traces": True, "snapshots": {}}
    for step in (100, 400):
        checkpoint = run_dir / f"step-{step}"
        if not (checkpoint / "training_metadata.json").is_file():
            summary["snapshots"][str(step)] = {"status": "missing_exact_checkpoint"}
            write_report(output / "summary.json", summary)
            continue

        progress = validate_checkpoint(checkpoint, recipe, step)
        target = output / f"step-{step}"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.taal.stability_observed_audit",
                "--config",
                str(AUDIT_CONFIG),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(target),
            ]
        )
        summary["snapshots"][str(step)] = {
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "progress": progress,
            "comparison_basis": "successful_optimizer_updates; skipped attempts retained",
        }
        write_report(output / "summary.json", summary)
    if summary["snapshots"]["400"]["status"] == "complete":
        summary["long_probe"] = long_probe(
            run_dir / "step-400", source_run, output / "long-probe"
        )
    write_report(output / "summary.json", summary)
    if any(row["status"] != "complete" for row in summary["snapshots"].values()):
        raise RuntimeError(
            "Audit incomplete; retained results and exact checkpoint status"
        )
    return summary


def run_group(group, source_run, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for arm in TRAINING_GROUPS[group]:
        run_id = f"taal-stability-{arm}-{os.environ['SLURM_JOB_ID']}"
        env = os.environ | {"CHECKPOINT_RUN_ID": run_id}
        print(f"STABILITY_TRAIN_START arm={arm} run={run_id}", flush=True)
        training_log = (output / f"{arm}-training.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "training.train",
                "--config",
                str(arm_config_path(arm)),
            ],
            env=env,
            stdout=training_log,
            stderr=subprocess.STDOUT,
        )
        stopped = []

        def stop_training(signum, frame):
            stopped.append(signum)
            process.send_signal(signal.SIGTERM)

        previous = signal.signal(signal.SIGTERM, stop_training)
        try:
            returncode = process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
            training_log.close()
        record = {
            "training_returncode": returncode,
            "checkpoint_directory": f"checkpoints/{run_id}",
            "training_log": str(output / f"{arm}-training.log"),
        }
        summary[arm] = record
        write_report(output / "summary.json", summary)
        if stopped:
            record["audit_error"] = (
                "Allocation termination requested; training received TERM to save progress"
            )
            write_report(output / "summary.json", summary)
            raise RuntimeError(record["audit_error"])

        try:
            record["audit"] = audit_arm(
                arm, Path("checkpoints") / run_id, source_run, output / arm
            )
        except Exception as error:
            record["audit_error"] = repr(error)
        write_report(output / "summary.json", summary)
        print(f"STABILITY_ARM_DONE arm={arm} status={record}", flush=True)
    if any(
        row["training_returncode"] != 0 or "audit_error" in row
        for row in summary.values()
    ):
        raise RuntimeError(
            "One or more arms incomplete; inspect summary, no arm silently skipped"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=TRAINING_GROUPS, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_group(args.group, args.source_run, args.output_dir)


if __name__ == "__main__":
    main()
