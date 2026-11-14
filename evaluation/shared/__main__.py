"""Run shared benchmarks from local JSON/JSONL, or rescore saved predictions."""

import argparse
import asyncio
import json
import re
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3Config

from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from evaluation.judge import judge_all
from evaluation.models import configure_model, load_model
from evaluation.shared.backends import QwenBackend
from evaluation.shared.benchmarks import make_benchmark
from evaluation.shared.contracts import Example, Prompt
from evaluation.shared.runner import digest, rescore, run, validate_predictions
from evaluation.storage import ensure_manifest, file_sha256, read_results
from evaluation.taal.prefix_store import checkpoint_fingerprint
from utils.seed import seed_everything


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = []
    for folder in ("evaluation", "architectures", "inference"):
        files.extend((root / folder).rglob("*.py"))

    fingerprints = []
    for path in sorted(files):
        relative_path = str(path.relative_to(root))
        content_hash = file_sha256(path)
        fingerprints.append((relative_path, content_hash))
    return digest(fingerprints)


def tokenizer_fingerprint(tokenizer):
    # Token IDs alone cannot identify a tokenizer: special tokens and chat rendering matter.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(
            "Shared CLI requires a fast tokenizer for reproducible fingerprinting"
        )

    return digest(
        {
            "backend": backend.to_str(),
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template,
        }
    )


def load_rows(path):
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue

            rows.append(json.loads(line))

    if not isinstance(rows, list):
        raise ValueError("Dataset must be a JSON array or JSONL of objects")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Dataset rows must be objects")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--score-only", action="store_true")
    phase.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    benchmark = make_benchmark(settings["benchmark"])
    output = args.output_dir
    if args.score_only or args.judge_only:
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest["benchmark"] != benchmark.name
            or manifest["prompt_version"] != benchmark.prompt_version
        ):
            raise ValueError(
                "Scoring configuration does not match saved benchmark/prompt"
            )

        records = json.loads((output / "examples.json").read_text())
        if digest(records) != manifest["examples_sha256"]:
            raise ValueError("Saved evaluation examples changed")

        examples = []
        for row in records:
            prompt = Prompt(
                tuple(row["prompt"]["token_ids"]),
                row["prompt"]["persistent_prefix_length"],
            )
            example = Example(
                row["example_id"],
                prompt,
                tuple(row["references"]),
                row["metadata"],
                row["group_id"],
            )
            examples.append(example)
        predictions = read_results(output / "predictions.jsonl", "example_id")
        validate_predictions(
            output,
            predictions,
            examples,
            save_traces=manifest["save_traces"],
            memory_trace=manifest["backend"]["capabilities"]["memory_trace"],
        )
        if args.judge_only:
            if benchmark.name != "longmemeval":
                raise ValueError(
                    "External rubric judging is only supported for LongMemEval"
                )

            references = []
            for example in examples:
                reference = {
                    "question_id": example.example_id,
                    "answer": example.references[0],
                    **example.metadata,
                }
                references.append(reference)
            # judge_all consumes the historical question_id/hypothesis schema.
            judge_dir = output / "longmemeval_judge"
            judge_dir.mkdir(exist_ok=True)
            ensure_manifest(
                judge_dir / "source.json", {"predictions_sha256": digest(predictions)}
            )
            prediction_path = judge_dir / "predictions.jsonl"
            prediction_lines = []
            for example_id, prediction in predictions.items():
                row = {
                    "question_id": example_id,
                    "hypothesis": prediction["hypothesis"],
                }
                prediction_lines.append(json.dumps(row) + "\n")
            prediction_path.write_text("".join(prediction_lines))
            asyncio.run(judge_all(references, judge_dir, settings["judge"]))
        else:
            print(json.dumps(rescore(benchmark, examples, output), indent=2))
        return

    if type(settings.get("save_traces", True)) is not bool:
        raise ValueError("save_traces must be boolean")

    if type(settings.get("seed", 42)) is not int:
        raise ValueError("seed must be an integer")

    for key in ("revision", "split"):
        if (
            not isinstance(settings["dataset"].get(key), str)
            or not settings["dataset"][key]
        ):
            raise ValueError(f"dataset.{key} must be an explicit nonempty string")

    model_settings = settings["model"]
    mode = model_settings["mode"]
    if mode not in {"taal", "full", "swa", "fw_swa"}:
        raise ValueError(
            "Backend not implemented; Gated DeltaNet/static adapters require their model branch first"
        )

    source = Path(model_settings["checkpoint"]).expanduser()
    if not source.is_dir():
        revision = model_settings.get("revision")
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise ValueError(
                "Remote checkpoints require an immutable 40-character revision"
            )

        source = Path(
            snapshot_download(
                model_settings["checkpoint"],
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(source)
    token_hash = tokenizer_fingerprint(tokenizer)
    dataset_settings = settings["dataset"]
    path = Path(dataset_settings["path"])
    rows = load_rows(path)
    if (
        benchmark.name == "controlled_prepared"
        and dataset_settings.get("tokenizer_sha256") != token_hash
    ):
        raise ValueError(
            f"Prepared dataset tokenizer_sha256 must match tokenizer: {token_hash}"
        )

    examples = [benchmark.prepare(row, tokenizer) for row in rows]
    ids = [example.example_id for example in examples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation requires nonempty examples with unique IDs")

    raw_config = json.loads((source / "config.json").read_text())
    if mode == "taal":
        if raw_config.get("model_type") != TaalQwen3Config.model_type:
            raise ValueError("taal mode requires a TaaL checkpoint")

        config = TaalQwen3Config.from_pretrained(source)
    elif raw_config.get("model_type") == TaalQwen3Config.model_type:
        raise ValueError(
            "TaaL checkpoints cannot be loaded as attention-only baselines"
        )
    elif raw_config.get("model_type") == TTCDQwen3Config.model_type:
        config = TTCDQwen3Config.from_pretrained(source)
    else:
        config = TTCDQwen3Config(
            **Qwen3Config.from_pretrained(source).to_dict(), fast_weight_layers=[]
        )
    architecture = dict(model_settings.get("settings", {}))
    if mode != "taal":
        allowed = {
            "teacher_window_size",
            "student_window_size",
            "chunk_size",
            "fast_weight_read_scale",
            "retain_task_prefix",
        }
        if set(architecture) - allowed:
            raise ValueError(
                f"Unknown backend model settings: {sorted(set(architecture) - allowed)}"
            )

    if mode == "full":
        # Full attention must cover this entire run, not silently evict at a default window.
        maximum = max(len(item.prompt.token_ids) for item in examples) + settings[
            "generation"
        ].get("max_new_tokens", 256)
        architecture["teacher_window_size"] = max(maximum, config.student_window_size)
    config = configure_model(config, mode, architecture)
    dtype_name = model_settings.get("dtype", "bfloat16")
    if dtype_name not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16, or float16")

    device = torch.device(model_settings.get("device", "cuda"))
    seed = settings.get("seed", 42)
    seed_everything(seed)
    model = load_model(
        str(source),
        config,
        architecture.get("fast_weight_read_scale", 1.0),
        dtype=getattr(torch, dtype_name),
    )
    backend = QwenBackend(
        model,
        tokenizer,
        provenance={
            "checkpoint": str(source.resolve()),
            "checkpoint_sha256": checkpoint_fingerprint(source),
            "tokenizer": str(source.resolve()),
            "tokenizer_sha256": token_hash,
            "dtype": dtype_name,
            "device": str(device),
            "source_sha256": source_fingerprint(),
            "mode": mode,
            "torch_version": torch.__version__,
        },
        generation=settings["generation"],
        memory_read_scale=architecture.get("memory_read_scale", 1.0),
        retain_task_prefix=architecture.get("retain_task_prefix", False),
    )
    backend.validate(
        [example.prompt for example in examples],
        save_traces=settings.get("save_traces", True),
    )
    backend.require(settings.get("required_capabilities", []))
    model.to(device)
    summary = run(
        backend,
        benchmark,
        examples,
        output,
        dataset_identity={
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "revision": dataset_settings["revision"],
            "split": dataset_settings["split"],
        },
        seed=seed,
        save_traces=settings.get("save_traces", True),
        required_capabilities=settings.get("required_capabilities", []),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
