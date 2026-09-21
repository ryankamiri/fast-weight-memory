import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Int
import torch
from transformers import AutoTokenizer, Qwen3Config
import yaml

from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from evaluation.run import configure_model, load_model
from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.storage import append_result, ensure_manifest, read_results
from utils.seed import seed_everything


SYSTEMS = {
    "base_full": {"mode": "full", "read_scale": 0.0},
    "swa": {"mode": "swa", "read_scale": 0.0},
    "fw_0": {"mode": "fw_swa", "read_scale": 0.0},
    "fw_0_5": {"mode": "fw_swa", "read_scale": 0.5},
    "fw_1": {"mode": "fw_swa", "read_scale": 1.0},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation/configs/ttcd/bridge_memory.yaml"))
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
        model_config = TTCDQwen3Config(
            **base.to_dict(), fast_weight_layers=[],
            teacher_window_size=full_window,
            student_window_size=geometry["student_window_size"],
            chunk_size=geometry["chunk_size"],
        )
        model_config = configure_model(model_config, "full")
    else:
        source = str(args.checkpoint.resolve())
        model_config = TTCDQwen3Config.from_pretrained(source)
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
            "metrics": grouped_score_summary(
                results.values(),
                ("condition", "query_variant"),
            ),
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
        "metrics": grouped_score_summary(
            results.values(),
            ("condition", "query_variant"),
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
