"""Durable benchmark execution and independently versioned offline scoring."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from evaluation.shared.contracts import Backend, Benchmark, Example
from evaluation.storage import (
    append_result,
    atomic_json,
    ensure_manifest,
    file_sha256,
    read_results,
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_predictions(
    output: Path,
    predictions: dict,
    examples: list[Example],
    save_traces: bool,
    memory_trace: bool,
):
    known = {example.example_id: example for example in examples}
    if not predictions.keys() <= known.keys():
        raise ValueError("Saved predictions contain unknown example IDs")

    for key, row in predictions.items():
        example = known[key]
        if row.get("status") != "completed":
            raise ValueError(f"Saved prediction {key} is not completed")

        if row["prompt_tokens"] != len(example.prompt.token_ids) or row[
            "generated_tokens"
        ] != len(row["generated_token_ids"]):
            raise ValueError(f"Saved token counts disagree for {key}")

        artifacts = row.get("artifacts", [])
        kinds = {item["kind"] for item in artifacts}
        required = set()
        if save_traces:
            required.add("token_trace")
            if memory_trace:
                required.add("memory_trace")
        if not required <= kinds:
            raise ValueError(
                f"Saved prediction {key} has no required trace; use a new output directory"
            )

        for item in artifacts:
            path = (output / item["path"]).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError(
                    f"Missing or unsafe artifact for {key}: {item['path']}"
                )

            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"Artifact checksum mismatch for {key}")

            if item["kind"] == "token_trace":
                trace = json.loads(path.read_text())
                if (
                    trace.get("schema") != "evaluation-token-trace/v1"
                    or trace.get("example_id") != key
                    or trace.get("prompt_token_ids") != list(example.prompt.token_ids)
                    or trace.get("generated_token_ids") != row["generated_token_ids"]
                    or trace.get("hypothesis") != row["hypothesis"]
                    or trace.get("stop_reason") != row["stop_reason"]
                ):
                    raise ValueError(
                        f"Saved prediction and token trace disagree for {key}"
                    )


def run(
    backend: Backend,
    benchmark: Benchmark,
    examples: list[Example],
    output: Path,
    dataset_identity: dict,
    seed: int = 42,
    save_traces: bool = True,
    required_capabilities: list[str] | None = None,
    should_stop=lambda: False,
):
    if type(save_traces) is not bool or type(seed) is not int:
        raise ValueError("save_traces must be boolean and seed must be an integer")

    backend.require(required_capabilities or [])
    ids = [example.example_id for example in examples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation requires nonempty examples with unique IDs")

    backend.validate([example.prompt for example in examples], save_traces=save_traces)
    records = []
    for example in examples:
        records.append(asdict(example))
    records = json.loads(json.dumps(records))
    # Exact prompt IDs and references are stored once; references never enter generate.
    manifest = {
        "schema": "shared-evaluation/v1",
        "benchmark": benchmark.name,
        "prompt_version": benchmark.prompt_version,
        "dataset": dataset_identity,
        "examples_sha256": digest(records),
        "backend": backend.identity(),
        "seed": seed,
        "seed_policy": "same-seed-fresh-generator-per-example",
        "save_traces": save_traces,
        "required_capabilities": required_capabilities or [],
        "measurement_mode": "traced" if save_traces else "trace-disabled",
    }
    # Normalize tuples/config enum values exactly as they will be persisted.
    manifest = json.loads(json.dumps(manifest))
    ensure_manifest(output / "manifest.json", manifest)
    ensure_manifest(output / "examples.json", records)
    predictions = read_results(output / "predictions.jsonl", "example_id")
    validate_predictions(
        output,
        predictions,
        examples,
        save_traces=save_traces,
        memory_trace=backend.capabilities.memory_trace,
    )
    backend.validate_saved(output, predictions)
    for example in examples:
        if should_stop():
            return None

        if example.example_id in predictions:
            continue
        try:
            prediction = backend.generate(
                example.example_id,
                example.prompt,
                output,
                seed=seed,
                save_traces=save_traces,
            )
            row = asdict(prediction)
            if prediction.example_id != example.example_id:
                raise ValueError("Backend returned a different example ID")

            validate_predictions(
                output,
                {example.example_id: row},
                [example],
                save_traces=save_traces,
                memory_trace=backend.capabilities.memory_trace,
            )
            backend.validate_saved(output, {example.example_id: row})
            append_result(output / "predictions.jsonl", row)
            predictions[example.example_id] = row
            print(
                f"Generated {len(predictions)}/{len(examples)}: {example.example_id}",
                flush=True,
            )
        except Exception as error:
            append_result(
                output / "failures.jsonl",
                {
                    "example_id": example.example_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
    if should_stop():
        return None

    return rescore(benchmark, examples, output)


def average_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over the examples where it was measured."""
    totals = {}
    counts = {}
    for metrics in values:
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    averages = {}
    for name in sorted(totals):
        averages[name] = totals[name] / counts[name]
    return averages


def rescore(benchmark: Benchmark, examples: list[Example], output: Path):
    predictions = read_results(output / "predictions.jsonl", "example_id")
    if set(predictions) != {example.example_id for example in examples}:
        raise ValueError("Scoring requires exactly one saved prediction per example")

    rows = []
    groups = {}
    strata = {}
    for example in examples:
        metrics = benchmark.score(
            example, predictions[example.example_id]["hypothesis"]
        )
        rows.append({"example_id": example.example_id, "metrics": metrics})
        if example.group_id is not None and "exact_match" in metrics:
            groups.setdefault(example.group_id, []).append(metrics["exact_match"])
        for field in ("task", "length", "gap", "fact_count", "condition", "answerable"):
            if field in example.metadata and example.metadata[field] is not None:
                strata.setdefault(f"{field}={example.metadata[field]}", []).append(
                    metrics
                )

    # Keep per-example scores separate from paired-question correctness.
    all_metrics = []
    scored_examples = 0
    for row in rows:
        all_metrics.append(row["metrics"])
        if row["metrics"]:
            scored_examples += 1

    summary = {
        "examples": len(rows),
        "scored_examples": scored_examples,
        "metrics": average_metrics(all_metrics),
        "strata": {},
    }
    for stratum, values in strata.items():
        summary["strata"][stratum] = {
            "examples": len(values),
            "metrics": average_metrics(values),
        }

    if groups:
        correct_groups = 0
        group_sizes = {}
        for group_id, values in groups.items():
            group_sizes[group_id] = len(values)
            group_correct = True
            for value in values:
                if not value:
                    group_correct = False
                    break

            if group_correct:
                correct_groups += 1

        summary["group_all_correct"] = correct_groups / len(groups)
        summary["groups"] = len(groups)
        summary["group_sizes"] = group_sizes
    score_key = digest(
        {"benchmark": benchmark.name, "scorer": benchmark.scorer_version}
    )[:16]
    atomic_json(
        output / "scores" / f"{score_key}.json",
        {
            "benchmark": benchmark.name,
            "scorer_version": benchmark.scorer_version,
            "predictions_sha256": file_sha256(output / "predictions.jsonl"),
            "rows": rows,
            "summary": summary,
        },
    )
    return summary
