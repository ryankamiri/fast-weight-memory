"""Prepare one matched conflict audit at a specified fact-to-answer distance."""

import argparse
from pathlib import Path

import yaml

from data.taal_conflict_control import write_dataset


def prepare(*, gap: int, output_dir: Path) -> tuple[Path, Path]:
    if gap not in (64, 192, 512, 1024):
        raise ValueError("The declared distance sweep uses gaps 64, 192, 512, 1024")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "conflict_distance.jsonl"
    config_path = output_dir / "audit_config.yaml"
    write_dataset(
        dataset_path,
        pairs=8,
        window=128,
        gap=gap,
        allow_visible=gap <= 128,
    )
    base = Path("evaluation/configs/taal/conflict_control_audit.yaml")
    config = yaml.safe_load(base.read_text())
    config["dataset"]["path"] = str(dataset_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return dataset_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset_path, config_path = prepare(gap=args.gap, output_dir=args.output_dir)
    print(f"DISTANCE_SWEEP gap={args.gap} dataset={dataset_path} config={config_path}")


if __name__ == "__main__":
    main()
