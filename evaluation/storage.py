"""Durable JSONL results and a simple settings check for resuming."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def ensure_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(
                f"Run settings changed. Use a new output directory: {path.parent}"
            )
    else:
        atomic_json(path, manifest)


def read_results(path, id_field="question_id"):
    """Repair only an interrupted final line; never ignore corrupt middle rows."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("rb+") as file:
        while True:
            line = file.readline()
            if not line:
                break

            end = file.tell()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line.endswith(b"\n") or file.read(1):
                    raise ValueError(f"Corrupt result inside {path}")
                file.truncate(end - len(line))
                break
            if row[id_field] in rows:
                raise ValueError(f"Duplicate {id_field} in {path}")
            rows[row[id_field]] = row
            if not line.endswith(b"\n"):
                file.write(b"\n")
    return rows


def append_result(path, row):
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    """Hash artifact contents without loading large traces into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()
