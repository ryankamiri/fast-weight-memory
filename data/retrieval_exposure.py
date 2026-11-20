"""Select matched repetition/coverage cohorts without opening the locked test split."""

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random

from data.retrieval import load_retrieval_splits
from evaluation.storage import ensure_manifest, file_sha256
from training.config import load_config


def balanced_blocks(rows, labels, count=8, seed=42):
    families = defaultdict(list)
    for row in rows:
        families[row["metadata"]["family_id"]].append(row)
    if any(len(group) != 8 for group in families.values()):
        raise ValueError("Expected complete two-fact, two-assignment, two-order families")

    candidates = list(families)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    pairs = {name: set(families[name][0]["metadata"]["bindings"].values()) for name in candidates}

    def select(remaining, available):
        if not remaining:
            return []
        label = min(remaining)
        for family in available:
            pair = pairs[family]
            if len(pair) == 2 and label in pair and pair <= remaining:
                rest = select(remaining - pair, [item for item in available if item != family])
                if rest is not None:
                    return [family, *rest]
        return None

    blocks = []
    for _ in range(count):
        chosen = select(set(labels), candidates)
        if chosen is None:
            raise ValueError("Cannot form enough disjoint label-balanced family blocks")
        candidates = [name for name in candidates if name not in chosen]
        block = [row for name in chosen for row in families[name]]
        rng.shuffle(block)
        if Counter(row["answer"] for row in block) != Counter({label: 4 for label in labels}):
            raise ValueError("Family labels are not balanced across all queries")
        blocks.append(block)
    return blocks


def stratified_rows(rows, per_cell=2):
    cells = defaultdict(list)
    for row in rows:
        cells[(row["answer"], row["metadata"]["queried_fact_order"])].append(row)
    if len(cells) != 32 or any(len(items) < per_cell for items in cells.values()):
        raise ValueError("Insufficient consumed rows for label-by-query-position strata")
    return [row for cell in sorted(cells) for row in cells[cell][:per_cell]]


def write_rows(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Refusing to replace an existing cohort: {path}")
    path.write_text(text)
    return {"rows": len(rows), "sha256": file_sha256(path)}


def prepare(config_path, output):
    config = load_config(config_path)
    splits, original = load_retrieval_splits(config.data)
    blocks = balanced_blocks(splits["train"], original["config"]["label_pool"])
    val_families = list(dict.fromkeys(row["metadata"]["family_id"] for row in splits["val"]))[:16]
    val = [row for row in splits["val"] if row["metadata"]["family_id"] in val_families]
    provenance = {"source_manifest_sha256": file_sha256(Path(config.data.manifest_path)),
                  "selection": "disjoint label-balanced family blocks; seed42; first block shared"}
    for arm, rows in (("repeat", blocks[0]), ("broad", sum(blocks, []))):
        manifest = copy.deepcopy(original)
        manifest["config"].update(train_families=len(rows) // 8, val_families=16)
        manifest["selection"] = provenance
        manifest["files"] = {split: write_rows(output / arm / f"{split}.jsonl", selected)
                             for split, selected in (("train", rows), ("val", val))}
        ensure_manifest(output / arm / "manifest.json", manifest)
    files = {"common-fit": write_rows(output / "common-fit.jsonl", blocks[0])}
    # This exactly reproduces epoch zero of the old seed42 shuffled sampler.
    consumed = copy.deepcopy(splits["train"])
    random.Random(config.training.seed).shuffle(consumed)
    for index, row in enumerate(consumed):
        row["metadata"]["consumed_index"] = index + 1
    for arm, count in (("A", 400), ("B", 400), ("C", 401)):
        cohort = consumed[:count]
        sample = stratified_rows(cohort)
        # One example per label, alternating the queried fact position.
        parity = [sample[index * 4 + (index % 2) * 2] for index in range(16)]
        for name, selected in (("consumed", cohort), ("loss", sample), ("parity", parity)):
            key = f"{arm}-{name}"
            files[key] = write_rows(output / f"{key}.jsonl", selected)
    ensure_manifest(output / "selection.json", provenance | {"files": files})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/taal/retrieval_acquisition.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/retrieval_exposure"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
