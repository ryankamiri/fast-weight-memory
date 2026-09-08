"""Durable JSONL results and a simple settings check for resuming."""

import json
import os


def ensure_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Run settings changed. Use a new output directory: {path.parent}")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def read_results(path):
    """Repair only an interrupted final line; never ignore corrupt middle rows."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("rb+") as file:
        while line := file.readline():
            end = file.tell()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line.endswith(b"\n") or file.read(1):
                    raise ValueError(f"Corrupt result inside {path}")
                file.truncate(end - len(line))
                break
            if row["question_id"] in rows:
                raise ValueError(f"Duplicate question ID in {path}")
            rows[row["question_id"]] = row
            if not line.endswith(b"\n"):
                file.write(b"\n")
    return rows


def append_result(path, row):
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
