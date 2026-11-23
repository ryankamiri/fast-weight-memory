"""Login-node-safe input checks; no model imports or GPU execution."""

import json
from pathlib import Path

from evaluation.storage import file_sha256
from scripts.retrieval_exposure_preflight import check_data


def check_inputs():
    check_data()
    root = Path("data/shared_retrieval")
    manifest = json.loads((root / "manifest.json").read_text())
    if file_sha256(root / "val.jsonl") != manifest["files"]["val"]["sha256"]:
        raise ValueError("Full validation checksum mismatch")
    if manifest["files"]["val"]["rows"] != 512:
        raise ValueError("Expected all 512 validation questions")
    for run in ("taal-retrieval-lr3e4-11872834", "taal-exposure-repeat-12045460", "taal-exposure-broad-12045462"):
        checkpoint = Path("checkpoints") / run / "final"
        for name in ("training_metadata.json", "trainer_state.pt", "tokenizer.json", "config.json"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
    for arm in ("repeat", "broad"):
        directory = Path(f"output/retrieval-exposure-20261121-restart/audit-{arm}/{arm}/val")
        if not (directory / "predictions.jsonl").is_file() or not (directory / "manifest.json").is_file():
            raise FileNotFoundError(directory)
    print("Parallel experiment inputs and historical audit paths verified", flush=True)


if __name__ == "__main__":
    check_inputs()
