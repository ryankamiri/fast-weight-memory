"""Bounded real-checkpoint diagnosis; stops and saves the first non-finite block."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.data import PROMPT_VERSION, prepare_example
from evaluation.run import load_model
from evaluation.taal.longmemeval import create_trace_exporter, generate_taal_example, resolve_generation_settings
from evaluation.taal.numerical_diagnostic import BlockMonitor, NumericalFailure, replay_block
from evaluation.taal.prefix_store import checkpoint_fingerprint
from utils.seed import seed_everything


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/taal/longmemeval_smoke.yaml"))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--read-scales", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-checkpoint numerical diagnostic expects CUDA")

    if args.output_dir.exists():
        raise ValueError("Use a fresh output directory for each diagnostic")

    settings = yaml.safe_load(args.config.read_text())
    source = str(args.checkpoint.resolve())
    config = TaalQwen3Config.from_pretrained(source)
    settings["generation"] = resolve_generation_settings(settings["generation"], config.working_memory_size)
    settings["checkpoint_fingerprint"] = checkpoint_fingerprint(args.checkpoint)
    dataset_settings = settings["dataset"]
    metadata_path = hf_hub_download(dataset_settings["repo_id"], f"{dataset_settings['variant']}/metadata.json",
                                    repo_type="dataset", revision=dataset_settings["revision"])
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Prepared prompt version differs from evaluator")

    dataset = load_dataset(dataset_settings["repo_id"], dataset_settings["variant"],
                           split=dataset_settings["split"], revision=dataset_settings["revision"])
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
    example = prepare_example(dataset[args.row], tokenizer, "completion")
    args.output_dir.mkdir(parents=True)
    write_report(args.output_dir / "example.json", example)
    manifest = {
        "checkpoint": source, "fingerprint": settings["checkpoint_fingerprint"],
        "model_config": config.to_dict(), "settings": settings, "metadata": metadata,
        "row": args.row, "question_id": example["question_id"], "read_scales": args.read_scales,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "purpose": "First nonfinite in actual execution order; not an accuracy benchmark",
    }
    write_report(args.output_dir / "manifest.json", manifest)
    seed_everything(settings["seed"])
    model = load_model(source, config).to("cuda").eval()
    for scale in args.read_scales:
        arm = args.output_dir / f"reads-{scale:g}"
        arm.mkdir()
        settings["model"]["memory_read_scale"] = scale
        exporter = create_trace_exporter(arm, source, config, metadata, settings, {})
        monitor = BlockMonitor(model, example["prompt_length"])
        started = time.perf_counter()
        report = {"scale": scale, "question_id": example["question_id"], "status": "started"}
        write_report(arm / "report.json", report)
        print(f"NUMERICAL_ARM scale={scale} tokens={example['prompt_length']}", flush=True)
        try:
            with monitor:
                report["outcome"] = generate_taal_example(model, tokenizer, example, settings,
                                                          torch.device("cuda"), exporter)
            report["status"] = "finite_generation"
        except NumericalFailure as error:
            report.update(status="nonfinite_detected", first_bad_boundary=error.report)
            write_report(arm / "report.json", report)
            block = monitor.last_block
            if block is not None and error.report.get("stage") == "taal":
                report["replay"] = replay_block(model.model.layers[block.layer].taal, block, arm / "reproducer.pt")
                first = report["replay"]["first_nonfinite"]
                if first is not None and first["position"] is not None:
                    offset = first["position"] - block.start
                    first["token_id"] = block.token_ids[offset]
                    first["token_text"] = tokenizer.decode([block.token_ids[offset]])
                    first["phase"] = "prompt" if first["position"] < example["prompt_length"] else "generated"
                print(f"NUMERICAL_FAILURE {json.dumps(first, allow_nan=False)}", flush=True)
        except Exception as error:
            report.update(status="diagnostic_error", error=repr(error))
            write_report(arm / "report.json", report)
            raise
        finally:
            report["seconds"] = time.perf_counter() - started
            report["block_samples"] = monitor.samples
            write_report(arm / "report.json", report)
        del monitor
        torch.cuda.empty_cache()
        print(f"NUMERICAL_ARM_DONE scale={scale} status={report['status']}", flush=True)


if __name__ == "__main__":
    main()
