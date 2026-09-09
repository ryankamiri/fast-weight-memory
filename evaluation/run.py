"""Generate independent LongMemEval answers while grading saved answers in the background."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Int
import torch
from transformers import AutoTokenizer, Qwen3Config
import yaml

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from evaluation.data import PROMPT_VERSION, prepare_example
from evaluation.judge import BackgroundJudge, judge_all
from evaluation.storage import append_result, ensure_manifest, read_results
from utils.seed import seed_everything


def configure_model(config, mode, settings=None):
    if mode not in {"full", "swa", "fw_swa"}:
        raise ValueError("mode must be full, swa, or fw_swa")
    if bool(config.fast_weight_layers) != (mode == "fw_swa"):
        raise ValueError("Checkpoint fast-weight layers do not match evaluation mode")
    if settings:
        overrides = {
            name: settings[name]
            for name in ("teacher_window_size", "student_window_size", "chunk_size")
            if name in settings
        }
        config = FWQwen3Config(**{**config.to_dict(), **overrides})
    return config


def load_model(source, config, fast_weight_read_scale=1.0):
    if type(fast_weight_read_scale) not in (int, float) or not math.isfinite(fast_weight_read_scale) or fast_weight_read_scale < 0:
        raise ValueError("fast_weight_read_scale must be a finite nonnegative number")
    config.fast_weight_read_scale = float(fast_weight_read_scale)
    model, loading = FWQwen3ForCausalLM.from_pretrained(
        source, config=config, dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not load exactly: {loading}")
    return model


@torch.inference_mode()
def generate_example(model, tokenizer, example, config, device):
    input_ids: Int[torch.Tensor, "1 S"] = torch.tensor(
        [example["input_ids"]], dtype=torch.long, device=device,
    )
    persistent = torch.arange(input_ids.shape[1], device=device) < example["persistent_prefix_length"]
    # Every question starts fresh. Only history within this example persists.
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    generation_settings = {
        **config["generation"], "persistent_mask": persistent, "generator": generator,
    }
    result = model.generate(input_ids, **generation_settings)

    return {
        "question_id": example["question_id"],
        "hypothesis": tokenizer.decode(result.token_ids[0].tolist(), skip_special_tokens=True),
        "prompt_tokens": input_ids.shape[1], "generated_tokens": result.token_ids.numel(),
        "stop_reason": result.stop_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/longmemeval_fw_swa.yaml"))
    parser.add_argument("--mode", choices=("full", "swa", "fw_swa"), help="Override the YAML mode")
    parser.add_argument("--checkpoint", type=Path, help="Saved checkpoint; omit when YAML supplies model_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    if args.judge_only and args.generate_only:
        parser.error("Choose at most one phase override")
    if not args.generate_only and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in .env or your shell for judging, or use --generate-only")
    
    config = yaml.safe_load(args.config.read_text())
    model_settings = config.get("model", {})
    model_id = model_settings.get("model_id")
    if (args.checkpoint is None) == (model_id is None):
        parser.error("Supply either --checkpoint or model.model_id in the YAML, not both")
    model_source = str(args.checkpoint.resolve()) if args.checkpoint is not None else model_id
    prompt_format = config.get("prompt_format", "completion")
    if prompt_format not in {"completion", "chat"}:
        parser.error("prompt_format must be completion or chat")
    args.mode = args.mode or config["mode"]
    config["mode"] = args.mode
    output = args.output_dir
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    dataset_config = config["dataset"]
    revision = dataset_config["revision"]
    if revision is None:
        revision = previous["dataset_revision"] if previous else HfApi().dataset_info(dataset_config["repo_id"]).sha
    metadata_path = hf_hub_download(
        dataset_config["repo_id"], f"{dataset_config['variant']}/metadata.json", repo_type="dataset", revision=revision,
    )
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Prepared prompt version does not match this evaluator")

    
    dataset = load_dataset(dataset_config["repo_id"], dataset_config["variant"], split="test", revision=revision)
    ids = dataset["question_id"]
    if len(ids) != len(set(ids)) or len(ids) != metadata["records"]:
        raise ValueError("Prepared dataset has duplicate IDs or inconsistent record count")
    if args.checkpoint is not None:
        model_config = FWQwen3Config.from_pretrained(model_source)
    else:
        # Untouched Qwen weights: wrap the architecture with no added FW layers.
        base_config = Qwen3Config.from_pretrained(model_source)
        model_config = FWQwen3Config(**base_config.to_dict(), fast_weight_layers=[])
    model_config = configure_model(model_config, args.mode, model_settings)
    ensure_manifest(manifest_path, {
        "config": {key: value for key, value in config.items() if key != "judge"},
        "mode": args.mode, "checkpoint": model_source,
        "dataset_revision": revision,
        "metadata": metadata, "teacher_window_size": model_config.teacher_window_size,
    })

    predictions = read_results(output / "predictions.jsonl")
    if not set(predictions) <= set(ids):
        raise ValueError("Predictions contain unknown question IDs")
    
    # Only lightweight reference columns are needed for grading.
    references = dataset.select_columns(["question_id", "question_type", "question", "answer", "abstention"])
    if args.judge_only:
        asyncio.run(judge_all(list(references), output, config["judge"]))
        return

    def generate_remaining(judge=None):
        if len(predictions) == len(dataset):
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Evaluation expects a CUDA GPU")
        device = torch.device("cuda", 0)
        seed_everything(config["seed"])

        if prompt_format == "chat":
            tokenizer = AutoTokenizer.from_pretrained(model_source)
        else:
            tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
        examples = [prepare_example(example, tokenizer, prompt_format) for example in dataset]
        maximum = max(example["prompt_length"] for example in examples)
        if max(example["persistent_prefix_length"] for example in examples) > model_config.max_persistent_tokens:
            raise ValueError("Prepared task prefix exceeds checkpoint persistent-token budget")
        model = load_model(model_source, model_config, model_settings.get("fast_weight_read_scale", 1.0))
        model.to(device).eval()
        print(
            f"Mode={args.mode}, prompt={prompt_format}, maximum prompt={maximum}, "
            f"teacher={model_config.teacher_window_size}, student={model_config.student_window_size}, "
            f"chunk={model_config.chunk_size}, FW read scale={model_settings.get('fast_weight_read_scale', 1.0)}",
            flush=True,
        )

        for example in examples:
            if example["question_id"] in predictions:
                continue

            start = time.perf_counter()
            result = generate_example(model, tokenizer, example, config, device)
            result["seconds"] = time.perf_counter() - start
            append_result(output / "predictions.jsonl", result)
            predictions[result["question_id"]] = result
            if judge is not None:
                judge.submit(result)
            print(f"Generated {len(predictions)}/{len(dataset)}: {result['question_id']}", flush=True)
        del model
        torch.cuda.empty_cache()

    if args.generate_only:
        generate_remaining()
    else:
        with BackgroundJudge(list(references), output, config["judge"]) as judge:
            for prediction in predictions.values():
                judge.submit(prediction)
            generate_remaining(judge)


if __name__ == "__main__":
    main()
