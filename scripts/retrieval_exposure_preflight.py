"""Lightweight file checks, safe on the login node; never imports a model."""

import argparse
import json
from pathlib import Path

from evaluation.storage import file_sha256


def check_data():
    root = Path("data/retrieval_exposure")
    selection = json.loads((root / "selection.json").read_text())
    if selection["source_manifest_sha256"] != file_sha256(Path("data/shared_retrieval/manifest.json")):
        raise ValueError("Source dataset changed after cohort selection")
    for name, expected in selection["files"].items():
        path = root / f"{name}.jsonl"
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Cohort checksum mismatch: {name}")
    for arm, count in (("repeat", 64), ("broad", 512)):
        manifest = json.loads((root / arm / "manifest.json").read_text())
        for split, expected in manifest["files"].items():
            path = root / arm / f"{split}.jsonl"
            if file_sha256(path) != expected["sha256"]:
                raise ValueError(f"Training cohort checksum mismatch: {arm}/{split}")
            if expected["rows"] != (count if split == "train" else 128):
                raise ValueError("Wrong cohort size")
    small = (root / "repeat/train.jsonl").read_text().splitlines()
    broad = (root / "broad/train.jsonl").read_text().splitlines()
    if small != broad[:64]:
        raise ValueError("Training arms do not share the first 64 episodes")
    if not Path("data/taal_workshop/evaluation.jsonl").is_file():
        raise ValueError("Historical workshop suite is missing")


def check_gate(directory):
    complete = json.loads((directory / "complete.json").read_text())
    if complete != {"phase": "parity", "arms": ["A", "B", "C"], "passed": True}:
        raise ValueError("Parity job did not finish successfully")
    for arm in ("A", "B", "C"):
        summary = json.loads((directory / arm / "parity/summary.json").read_text())
        if summary != {"examples": 16, "passed": True}:
            raise ValueError(f"Missing or failed {arm} parity screen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-dir", type=Path)
    args = parser.parse_args()
    check_data()
    if args.parity_dir is not None:
        check_gate(args.parity_dir)
    print("Exposure cohorts and requested parity receipts validated", flush=True)


if __name__ == "__main__":
    main()
