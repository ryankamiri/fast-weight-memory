"""Matched length/content tests, transition interventions and input-run milestone checks."""

import argparse
import json
import os
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
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.normalization_ablation import run_fixed, run_generation
from evaluation.taal.stability_interventions import INTERVENTIONS, intervene
from evaluation.taal.stability_observation import ReadRoundingRecorder
from evaluation.taal.stability_suite import AUDIT_CONFIG, normalization_settings


def length_examples(rows, pool, length):
    if length < 206 or not pool:
        raise ValueError(
            "Need a nonempty filler and at least the original prompt length"
        )
    result = []
    for row in rows:
        head = row["input_ids"][:32]
        query = row["input_ids"][row["final_query_position"] :]
        count = length - len(head) - len(query)
        filler = (pool * (count // len(pool) + 1))[:count]
        ids = head + filler + query
        result.append(
            row
            | {
                "input_ids": ids,
                "final_query_position": length - len(query),
                "final_answer_position": length,
                "prompt_length": length,
            }
        )
    return result


def run_audit(checkpoint, config, output, arm="baseline"):
    env = os.environ | {"TAAL_STABILITY_INTERVENTION": arm}
    settings = yaml.safe_load(Path(config).read_text())
    settings["stability_intervention"] = arm
    selected_config = output.parent / f"{output.name}-config.yaml"
    selected_config.parent.mkdir(parents=True, exist_ok=True)
    selected_config.write_text(yaml.safe_dump(settings, sort_keys=False))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.taal.stability_observed_audit",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(selected_config),
            "--output-dir",
            str(output),
        ],
        env=env,
    )
    return {
        "returncode": result.returncode,
        "status": "complete" if result.returncode == 0 else "failed",
        "output": str(output),
        "intervention": arm,
    }


def run_probe(checkpoint, source_run, directory, arm="baseline", read_scale=1.0):
    config = TaalQwen3Config.from_pretrained(checkpoint)
    source = load_replay_source(source_run, 169, [2, 3, 8, 12, 24, 27])
    metadata = source.manifest["metadata"]
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "arm": f"dynamics-{arm}",
        "checkpoint": str(checkpoint),
        "source_run": str(source_run),
        "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "layers": [2, 3, 8, 12, 24, 27],
        "trace_start": source.example["prompt_length"] - 32,
        "execution_block_size": 32,
        "max_new_tokens": 256,
        "normalization": normalization_settings(config),
        "fingerprint": checkpoint_fingerprint(checkpoint),
        "source_fingerprint": source.manifest["fingerprint"],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "allow_transition_ablation": True,
        "intervention": arm,
        "save_traces": True,
        "memory_read_scale": read_scale,
    }
    write_report(directory / "manifest.json", manifest)
    write_report(directory / "example.json", source.example)
    model = load_model(str(checkpoint), config).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"], revision=metadata["tokenizer_revision"]
    )
    summary = {}
    with intervene(model, arm):
        for phase, runner in [("fixed", run_fixed), ("generation", run_generation)]:
            with ReadRoundingRecorder(model, directory / f"{phase}-rounding.jsonl"):
                summary[phase] = runner(
                    model, tokenizer, source, manifest, directory / phase
                )
            write_report(directory / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def run_dynamics(checkpoint, source_run, input_run, output, phase):
    output.mkdir(parents=True, exist_ok=False)
    summary = {"length_content": {}, "transitions": {}, "input_milestones": {}}
    source = json.loads((source_run / "example.json").read_text())
    rows = [
        json.loads(line)
        for line in Path("data/taal_capacity_16_gap192.jsonl").read_text().splitlines()
    ][:4]
    pools = {
        "training": rows[0]["input_ids"][32:186],
        "natural": source["input_ids"][:4096],
        "repetitive": [353],
    }
    selected_pools = pools if phase == "length" else {}
    for family, pool in selected_pools.items():
        for length in (206, 1024, 4096, 8888):
            name = f"{family}-{length}"
            folder = output / f"length-{name}"
            folder.mkdir()
            examples = length_examples(rows, pool, length)
            dataset = folder / "examples.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in examples))
            metadata = json.loads(
                Path("data/taal_capacity_16_gap192.metadata.json").read_text()
            )
            metadata.update(
                content_family=family,
                prompt_length=length,
                prompt_tokens=[length],
                pairs=2,
                examples=len(examples),
                working_memory_size=32,
                fact_to_answer_gap=length - examples[0]["fact_position"],
            )
            dataset.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            config = yaml.safe_load(AUDIT_CONFIG.read_text())
            config["dataset"]["path"] = str(dataset)
            config["dataset"]["end"] = 4
            path = folder / "audit.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            summary["length_content"][name] = run_audit(
                checkpoint, path, folder / "audit"
            )
            write_report(output / "summary.json", summary)
    arms = INTERVENTIONS if phase == "transitions" else ()
    for arm in arms:
        record = {
            "short": run_audit(checkpoint, AUDIT_CONFIG, output / f"{arm}-short", arm)
        }
        summary["transitions"][arm] = record
        try:
            record["long"] = run_probe(
                checkpoint, source_run, output / f"{arm}-long", arm
            )
        except Exception as error:
            record["long_error"] = repr(error)
        write_report(output / "summary.json", summary)
    if phase == "transitions":
        for scale in (0.5, 0.0):
            name = f"whole_episode_reads_{scale}"
            try:
                summary["transitions"][name] = {
                    "short": {"returncode": 0, "status": "not_planned"},
                    "long": run_probe(
                        checkpoint, source_run, output / name, read_scale=scale
                    ),
                }
            except Exception as error:
                summary["transitions"][name] = {
                    "short": {"returncode": 0},
                    "long_error": repr(error),
                }
            write_report(output / "summary.json", summary)
    milestones = (200, 300, 400) if phase == "input" else ()
    for step in milestones:
        snapshot = input_run / f"step-{step}"
        record = {
            "short": run_audit(snapshot, AUDIT_CONFIG, output / f"input-{step}-short")
        }
        summary["input_milestones"][str(step)] = record
        try:
            record["long"] = run_probe(
                snapshot, source_run, output / f"input-{step}-long"
            )
        except Exception as error:
            record["long_error"] = repr(error)
        write_report(output / "summary.json", summary)
    failures = []
    for name, row in summary["length_content"].items():
        if row["returncode"]:
            failures.append(name)
    for family in ("transitions", "input_milestones"):
        for name, row in summary[family].items():
            if row["short"]["returncode"] or "long_error" in row:
                failures.append(f"{family}/{name}")
    write_report(
        output / "completion.json",
        {"status": "complete" if not failures else "partial", "failed": failures},
    )
    if failures:
        raise RuntimeError(f"Incomplete diagnostics: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("length", "transitions", "input"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dynamics(
        args.checkpoint, args.source_run, args.input_run, args.output_dir, args.phase
    )


if __name__ == "__main__":
    main()
