"""Prepare a matched full-validation audit without launching a model or touching test."""

import argparse
import json
from pathlib import Path

import yaml
import torch

from data.retrieval import load_retrieval_splits
from evaluation.storage import atomic_json, ensure_manifest, file_sha256
from training.config import load_config


def prepare_evaluation(train_config, checkpoint, output, read_scale=1.0):
    if read_scale not in (0.0, 1.0):
        raise ValueError("Acquisition audits require reads on or off")
    config = load_config(train_config)
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if metadata["progress"]["step"] != config.training.max_steps:
        raise ValueError("Incomplete training budget; resume training before matched evaluation")

    expected = config.to_dict()
    actual = dict(metadata["training_config"])
    expected.pop("wandb")
    actual.pop("wandb")
    if actual != expected:
        raise ValueError("Checkpoint does not match this LR arm's training recipe")

    if not (checkpoint / "tokenizer.json").is_file():
        raise ValueError("Checkpoint must contain its training tokenizer")

    for name, checksum in metadata.get("files", {}).items():
        if Path(name).name != name or file_sha256(checkpoint / name) != checksum:
            raise ValueError("Checkpoint artifact checksum mismatch")

    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if state["data"]["identity"]["manifest_sha256"] != file_sha256(Path(config.data.manifest_path)):
        raise ValueError("Dataset manifest changed since training")

    splits, manifest = load_retrieval_splits(config.data)
    settings = yaml.safe_load(Path("evaluation/configs/shared/retrieval_acquisition_val.yaml").read_text())
    settings["dataset"]["path"] = config.data.split_files["val"]
    settings["model"]["checkpoint"] = str(checkpoint)
    settings["model"]["settings"]["memory_read_scale"] = read_scale
    output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output / "config.yaml", settings)
    atomic_json(output / "training_receipt.json", {
        "checkpoint": str(checkpoint), "progress": metadata["progress"],
        "validation_questions": len(splits["val"]),
        "dataset_manifest_sha256": file_sha256(Path(config.data.manifest_path)),
        "val_sha256": manifest["files"]["val"]["sha256"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--read-scale", type=float, default=1.0, choices=[0.0, 1.0])
    args = parser.parse_args()
    prepare_evaluation(args.train_config, args.checkpoint, args.output_dir, args.read_scale)


if __name__ == "__main__":
    main()
