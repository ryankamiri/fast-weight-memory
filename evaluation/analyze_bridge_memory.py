import argparse
import json
from pathlib import Path
import random

from evaluation.scoring import grouped_score_summary
from evaluation.storage import read_results


SYSTEM_NAMES = ("base_full", "swa", "fw_0", "fw_0_5", "fw_1")


def percentile(values, probability):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def interaction(fw_zero, treatment, metric, query_variant=None, samples=5000):
    values = []
    fact_ids = sorted({row["fact_id"] for row in fw_zero.values()})
    variants = [query_variant] if query_variant else ["exact", "paraphrased"]
    for fact_id in fact_ids:
        for variant in variants:
            keys = {
                condition: next(
                    row["example_id"] for row in fw_zero.values()
                    if row["fact_id"] == fact_id
                    and row["condition"] == condition
                    and row["query_variant"] == variant
                )
                for condition in ("no_bridge", "bridge")
            }
            values.append(
                (treatment[keys["bridge"]][metric] - treatment[keys["no_bridge"]][metric])
                - (fw_zero[keys["bridge"]][metric] - fw_zero[keys["no_bridge"]][metric])
            )
    mean = sum(values) / len(values)
    generator = random.Random(42)
    bootstrap = [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    return {
        "paired_units": len(values),
        "mean": mean,
        "bootstrap_95_percent_interval": [
            percentile(bootstrap, 0.025), percentile(bootstrap, 0.975),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SYSTEM_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("bridge-memory-analysis.json"))
    args = parser.parse_args()

    runs = {
        name: read_results(getattr(args, name) / "results.jsonl", id_field="example_id")
        for name in SYSTEM_NAMES
    }
    expected = set(runs["base_full"])
    if not expected or any(set(run) != expected for run in runs.values()):
        raise ValueError("Every run must contain the same complete example IDs")

    report = {
        "systems": {
            name: grouped_score_summary(
                run.values(),
                ("condition", "query_variant"),
            )
            for name, run in runs.items()
        },
        "bridge_interaction_relative_to_fw_0": {},
    }
    for name in ("fw_0_5", "fw_1"):
        report["bridge_interaction_relative_to_fw_0"][name] = {
            variant: {
                metric: interaction(runs["fw_0"], runs[name], metric, variant)
                for metric in ("candidate_correct", "target_log_probability")
            }
            for variant in ("exact", "paraphrased", None)
        }
    # JSON object keys cannot represent None clearly.
    for result in report["bridge_interaction_relative_to_fw_0"].values():
        result["combined"] = result.pop(None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
