"""Fail early on the allocated GPU if pinned caches/proxy/inputs are unavailable."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import requests
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Preflight requires a BF16-capable CUDA GPU")
    for name in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy"):
        if not os.environ.get(name):
            raise RuntimeError(f"Missing launch environment: {name}")
    if args.checkpoint is None:
        model = settings["model"]
        data = settings["data"]
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("Training model revision must be an immutable commit")
        model_config = hf_hub_download(
            model["model_id"], "config.json", revision=model["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"], local_files_only=True,
        )
        json.loads(Path(model_config).read_text())
        hf_hub_download(
            model["model_id"], "model.safetensors", revision=model["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"], local_files_only=True,
        )
        repo, variant, revision, split = (
            data["dataset_id"], data["dataset_config"], data.get("revision"), data["train"]["split"]
        )
    else:
        for filename in ("config.json", "model.safetensors", "training_metadata.json"):
            if not (args.checkpoint / filename).is_file():
                raise FileNotFoundError(args.checkpoint / filename)
        data = settings["dataset"]
        if "path" in data:
            repo, variant, revision, split = data["path"], None, None, "train"
        else:
            repo, variant, revision, split = data["repo_id"], data["variant"], data["revision"], data["split"]
    if repo.endswith(".jsonl"):
        if not Path(repo).is_file():
            raise FileNotFoundError(repo)
        metadata = Path(repo).with_suffix(".metadata.json")
        json.loads(metadata.read_text())
        dataset = load_dataset("json", data_files=str(repo), split="train", streaming=True)
        first = next(iter(dataset))
        if not first.get("input_ids"):
            raise ValueError("Local bridge-memory record has no input_ids")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        print("EXPLORER_PREFLIGHT " + json.dumps({
            "status": "pass", "commit": commit,
            "gpu": torch.cuda.get_device_name(0), "dataset_path": repo,
            "first_record_tokens": len(first["input_ids"]), "config": str(args.config),
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        }), flush=True)
        return
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be an immutable commit")
    metadata = hf_hub_download(
        repo, f"{variant}/metadata.json", repo_type="dataset", revision=revision, local_files_only=True,
    )
    json.loads(Path(metadata).read_text())
    # Streaming can need the Hub even with cached weights. Prove the compute
    # node's exported proxy works before spending hours inside the model.
    requests.get("https://huggingface.co", timeout=30).raise_for_status()
    dataset = load_dataset(repo, variant, split=split, revision=revision, streaming=True)
    first = next(iter(dataset))
    if not first.get("input_ids"):
        raise ValueError("Pinned dataset record has no input_ids")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("EXPLORER_PREFLIGHT " + json.dumps({
        "status": "pass", "commit": commit,
        "gpu": torch.cuda.get_device_name(0), "dataset_revision": revision,
        "first_record_tokens": len(first["input_ids"]), "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
    }), flush=True)


if __name__ == "__main__":
    main()
