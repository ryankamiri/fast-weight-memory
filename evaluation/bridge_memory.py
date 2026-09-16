import argparse
import json
import math
from pathlib import Path
import time

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Float, Int
import torch
from transformers import AutoTokenizer, Qwen3Config
import yaml

from architectures.qwen.configuration import FWQwen3Config
from evaluation.run import configure_model, load_model
from evaluation.storage import append_result, ensure_manifest, read_results
from utils.seed import seed_everything


SYSTEMS = {
    "base_full": {"mode": "full", "read_scale": 0.0},
    "swa": {"mode": "swa", "read_scale": 0.0},
    "fw_0": {"mode": "fw_swa", "read_scale": 0.0},
    "fw_0_5": {"mode": "fw_swa", "read_scale": 0.5},
    "fw_1": {"mode": "fw_swa", "read_scale": 1.0},
}


def score_logits(logits, example, tokenizer):
    logits: Float[torch.Tensor, "vocab_size"] = logits.float()
    target = int(example["target_token_id"])
    candidates: Int[torch.Tensor, "num_candidates"] = torch.tensor(
        example["candidate_token_ids"], dtype=torch.long, device=logits.device,
    )
    target_score = logits[target]
    target_rank = int((logits > target_score).sum().item()) + 1
    candidate_scores = logits[candidates]
    candidate_choice = int(candidates[candidate_scores.argmax()].item())
    decoy_scores = candidate_scores[candidates != target]
    top_token = int(logits.argmax().item())
    return {
        "target_rank": target_rank,
        "target_reciprocal_rank": 1.0 / target_rank,
        "target_log_probability": float((target_score - torch.logsumexp(logits, dim=0)).item()),
        "target_vs_best_decoy_margin": float((target_score - decoy_scores.max()).item()),
        "candidate_choice_token_id": candidate_choice,
        "candidate_correct": candidate_choice == target,
        "vocabulary_top_token_id": top_token,
        "vocabulary_top_token": tokenizer.decode([top_token]),
        "vocabulary_top_1_correct": top_token == target,
    }


def summarize(rows):
    groups = {"all": list(rows)}
    for row in rows:
        key = f"{row['condition']}/{row['query_variant']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for name, group in groups.items():
        count = len(group)
        summary[name] = {
            "examples": count,
            "candidate_accuracy": sum(row["candidate_correct"] for row in group) / count,
            "vocabulary_top_1_accuracy": sum(row["vocabulary_top_1_correct"] for row in group) / count,
            "mean_target_log_probability": math.fsum(row["target_log_probability"] for row in group) / count,
            "mean_target_reciprocal_rank": math.fsum(row["target_reciprocal_rank"] for row in group) / count,
            "mean_target_vs_best_decoy_margin": math.fsum(
                row["target_vs_best_decoy_margin"] for row in group
            ) / count,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/bridge_memory.yaml"))
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    system = SYSTEMS[args.system]
    if (args.system == "base_full") == (args.checkpoint is not None):
        parser.error("base_full takes no checkpoint; every other system requires one")

    dataset_settings = settings["dataset"]
    revision = dataset_settings.get("revision") or HfApi().dataset_info(dataset_settings["repo_id"]).sha
    metadata_file = hf_hub_download(
        dataset_settings["repo_id"],
        f"{dataset_settings['variant']}/metadata.json",
        repo_type="dataset",
        revision=revision,
    )
    metadata = json.loads(Path(metadata_file).read_text())
    geometry = settings["geometry"]
    for name in ("teacher_window_size", "student_window_size", "chunk_size"):
        if geometry[name] != metadata[name]:
            raise ValueError(f"Configured {name} does not match the prepared dataset")

    dataset = load_dataset(
        dataset_settings["repo_id"], dataset_settings["variant"],
        split="test", revision=revision,
    )
    ids = dataset["example_id"]
    if len(ids) != len(set(ids)) or len(ids) != metadata["records"]:
        raise ValueError("Prepared dataset has duplicate IDs or an inconsistent record count")

    if args.system == "base_full":
        source = settings["model"]["base_model_id"]
        base = Qwen3Config.from_pretrained(source)
        full_window = settings["model"]["full_attention_window_size"]
        if full_window < metadata["maximum_prompt_length"]:
            raise ValueError("full_attention_window_size must cover every prepared prompt")
        model_config = FWQwen3Config(
            **base.to_dict(), fast_weight_layers=[],
            teacher_window_size=full_window,
            student_window_size=geometry["student_window_size"],
            chunk_size=geometry["chunk_size"],
        )
        model_config = configure_model(model_config, "full")
    else:
        source = str(args.checkpoint.resolve())
        model_config = FWQwen3Config.from_pretrained(source)
        model_config = configure_model(model_config, system["mode"], geometry)

    ensure_manifest(args.output_dir / "manifest.json", {
        "system": args.system,
        "checkpoint": source,
        "config": settings,
        "dataset_revision": revision,
        "metadata": metadata,
        "fast_weight_read_scale": system["read_scale"],
    })
    results = read_results(args.output_dir / "results.jsonl", id_field="example_id")
    if not set(results) <= set(ids):
        raise ValueError("Saved results contain unknown example IDs")
    if len(results) == len(dataset):
        summary = {
            "system": args.system,
            "fast_weight_read_scale": system["read_scale"],
            "metrics": summarize(results.values()),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Bridge-memory evaluation expects a CUDA GPU")

    seed_everything(settings["seed"])
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"], revision=metadata["tokenizer_revision"],
    )
    model = load_model(source, model_config, system["read_scale"])
    model.to(device).eval()
    print(
        f"System={args.system}, examples={len(dataset)}, teacher={model_config.teacher_window_size}, "
        f"student={model_config.student_window_size}, chunk={model_config.chunk_size}, "
        f"alpha={system['read_scale']}",
        flush=True,
    )

    for example in dataset:
        if example["example_id"] in results:
            continue
        input_ids: Int[torch.Tensor, "1 S"] = torch.tensor(
            [example["input_ids"]], dtype=torch.long, device=device,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.prefill(
                input_ids,
                execution_block_size=settings["execution_block_size"],
            )
        result = {
            "example_id": example["example_id"],
            "fact_id": example["fact_id"],
            "condition": example["condition"],
            "query_variant": example["query_variant"],
            "prompt_tokens": input_ids.shape[1],
            "seconds": time.perf_counter() - start,
            **score_logits(output.logits[0, -1], example, tokenizer),
        }
        append_result(args.output_dir / "results.jsonl", result)
        results[result["example_id"]] = result
        print(f"Scored {len(results)}/{len(dataset)}: {result['example_id']}", flush=True)

    summary = {
        "system": args.system,
        "fast_weight_read_scale": system["read_scale"],
        "metrics": summarize(results.values()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
