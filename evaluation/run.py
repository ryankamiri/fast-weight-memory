"""Generate independent LongMemEval answers, then run resumable API grading."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import torch
from transformers import AutoTokenizer
import yaml

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from evaluation.data import PROMPT_VERSION
from evaluation.judge import judge_all
from evaluation.storage import append_result, ensure_manifest, read_results
from utils.seed import seed_everything


def configure_model(config, mode):
    if mode not in {"full", "swa", "fw_swa"}:
        raise ValueError("mode must be full, swa, or fw_swa")
    if bool(config.fast_weight_layers) != (mode == "fw_swa"):
        raise ValueError("Checkpoint fast-weight layers do not match evaluation mode")
    return config


@torch.inference_mode()
def generate_example(model, tokenizer, example, config, device):
    input_ids = torch.tensor(
        [example["history_ids"] + example["question_ids"]], dtype=torch.long, device=device,
    )
    persistent = torch.arange(input_ids.shape[1], device=device) < example["persistent_prefix_length"]
    # Every question is an independent experiment, not another turn of the
    # preceding evaluation question. Only history within this example persists.
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    result = model.generate(
        input_ids, persistent_mask=persistent, generator=generator,
        **config["generation"],
    )
    return {
        "question_id": example["question_id"],
        "hypothesis": tokenizer.decode(result.token_ids[0].tolist(), skip_special_tokens=True),
        "prompt_tokens": example["prompt_length"], "generated_tokens": result.token_ids.numel(),
        "stop_reason": result.stop_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/longmemeval_fw_swa.yaml"))
    parser.add_argument("--mode", choices=("full", "swa", "fw_swa"), help="Override the YAML mode")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved best/ or final/ checkpoint directory")
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
    maximum = max(dataset["prompt_length"])
    model_config = configure_model(
        FWQwen3Config.from_pretrained(args.checkpoint), args.mode,
    )
    if max(dataset["persistent_prefix_length"]) > model_config.max_persistent_tokens:
        raise ValueError("Prepared task prefix exceeds checkpoint persistent-token budget")
    ensure_manifest(manifest_path, {
        "config": {key: value for key, value in config.items() if key != "judge"},
        "mode": args.mode, "checkpoint": str(args.checkpoint.resolve()),
        "dataset_revision": revision,
        "metadata": metadata, "teacher_window_size": model_config.teacher_window_size,
    })

    predictions = read_results(output / "predictions.jsonl")
    if not set(predictions) <= set(ids):
        raise ValueError("Predictions contain unknown question IDs")
    
    if not args.judge_only and len(predictions) < len(dataset):
        if not torch.cuda.is_available():
            raise RuntimeError("Evaluation expects a CUDA GPU")
        device = torch.device("cuda", 0)
        seed_everything(config["seed"])

        tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
        model, loading = FWQwen3ForCausalLM.from_pretrained(
            args.checkpoint, config=model_config, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint did not load exactly: {loading}")
        model.to(device).eval()
        print(f"Mode={args.mode}, maximum prompt={maximum}, teacher window={model_config.teacher_window_size}", flush=True)

        for example in dataset:
            if example["question_id"] in predictions:
                continue

            start = time.perf_counter()
            result = generate_example(model, tokenizer, example, config, device)
            result["seconds"] = time.perf_counter() - start
            append_result(output / "predictions.jsonl", result)
            predictions[result["question_id"]] = result
        
            print(f"Generated {len(predictions)}/{len(dataset)}: {result['question_id']}", flush=True)
        del model
        torch.cuda.empty_cache()
    if not args.generate_only:
        # Only lightweight reference columns are needed for grading.
        references = dataset.select_columns(["question_id", "question_type", "question", "answer", "abstention"])
        asyncio.run(judge_all(list(references), output, config["judge"]))


if __name__ == "__main__":
    main()
