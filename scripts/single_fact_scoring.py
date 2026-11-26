"""Predeclared label scoring and paired comparisons for single-fact retrieval."""

from collections import Counter
import math
import re

import numpy as np

from data.balanced_retrieval import LABELS
from evaluation.shared.benchmarks import ControlledBenchmark


def predicted_label(hypothesis):
    match = re.match(r"^\s*[\"'“‘]*([A-Za-z]+)\b", hypothesis)
    if match is None or match[1].lower() not in LABELS:
        return None
    label = match[1].lower()
    mentioned = set(re.findall(r"\b[a-z]+\b", hypothesis.lower())) & set(LABELS)
    if mentioned != {label}:
        return None
    return label


class SingleFactBenchmark(ControlledBenchmark):
    name = "single-fact"
    scorer_version = "leading-whole-label-no-conflicting-labels-v1"

    def score(self, example, hypothesis):
        scores = super().score(example, hypothesis)
        label = predicted_label(hypothesis)
        scores["label_correct"] = float(label in example.references)
        scores["valid_label"] = float(label is not None)
        return scores


def wilson(correct, total):
    if total <= 0:
        raise ValueError("An accuracy interval needs a nonempty cohort")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def accuracy_summary(rows, predictions):
    if {r["example_id"] for r in rows} != set(predictions):
        raise ValueError("Scoring requires every unique planned prediction")
    groups = {"all": rows}
    for regime in sorted({r["metadata"]["regime"] for r in rows}):
        groups[regime] = [r for r in rows if r["metadata"]["regime"] == regime]
    result = {}
    for name, selected in groups.items():
        labels = [predicted_label(predictions[r["example_id"]]["hypothesis"]) for r in selected]
        correct = sum(label == row["answer"] for label, row in zip(labels, selected))
        result[name] = {
            "correct": correct, "total": len(selected), "accuracy": correct / len(selected),
            "wilson_95": wilson(correct, len(selected)),
            "strict_correct": sum(predictions[r["example_id"]]["hypothesis"].strip().lower() == r["answer"] for r in selected),
            "predicted_labels": dict(Counter(label or "<invalid>" for label in labels)),
        }
    return result


def paired_comparison(rows, intervention, baseline):
    keys = {r["example_id"] for r in rows}
    if set(intervention) != keys or set(baseline) != keys:
        raise ValueError("Paired comparisons require identical complete example IDs")
    difference = np.array([
        int(predicted_label(intervention[r["example_id"]]["hypothesis"]) == r["answer"])
        - int(predicted_label(baseline[r["example_id"]]["hypothesis"]) == r["answer"])
        for r in rows
    ])
    rng = np.random.default_rng(42)
    indices = np.arange(len(rows))
    strata = [indices[[r["metadata"]["regime"] == name for r in rows]]
              for name in sorted({r["metadata"]["regime"] for r in rows})]
    means = []
    for _ in range(10000):
        sample = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        means.append(float(difference[sample].mean()))
    lower, upper = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "questions": len(rows), "gains": int((difference == 1).sum()),
        "losses": int((difference == -1).sum()), "accuracy_difference": float(difference.mean()),
        "paired_stratified_bootstrap_95": [lower, upper], "bootstrap_samples": 10000,
        "within_five_points_exploratory": lower > -0.05,
        "scope": "fixed trained checkpoint; independent records, conditional on these synthetic template banks",
        "inference_note": "Exploratory interval; no multiple-comparison correction or training-seed replication. Degenerate bootstrap samples require a conservative paired interval before a formal non-inferiority claim.",
    }
