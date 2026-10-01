"""Position-matched fact-distance and whole-episode read-path diagnostics."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer
import yaml

from evaluation.bridge_data import _templates
from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.state_audit import fork_session_state, load_model, select_examples, split_episode
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import (
    TaalTraceExporter,
    TaalTraceRecorder,
    make_trace_tokens,
    trace_greedy_prediction,
)
from utils.seed import seed_everything


READ_CONDITIONS = (
    ("reads_on", 1.0, 1.0),
    ("query_off", 1.0, 0.0),
    ("episode_off", 0.0, 0.0),
)


def relocate_fact(example, tokenizer, distance: int | None) -> dict:
    """Move the sole fact within a fixed-length episode, leaving query unchanged."""
    if distance is None:
        return dict(example)
    prefix, query = split_episode(example)
    fact_prefix, fact_suffix, _, _, _ = _templates(example["record_name"])
    fact = (
        tokenizer.encode(fact_prefix, add_special_tokens=False)
        + [int(example["target_token_id"])]
        + tokenizer.encode(fact_suffix, add_special_tokens=False)
    )
    if prefix[:len(fact)] != fact:
        raise ValueError("The source episode does not begin with its expected fact")
    fact_target_offset = len(tokenizer.encode(fact_prefix, add_special_tokens=False))
    insert = len(prefix) + len(query) - distance - fact_target_offset
    body = prefix[len(fact):]
    if not 0 <= insert <= len(body):
        raise ValueError(f"Fact distance {distance} does not fit the prefix")
    moved_prefix = body[:insert] + fact + body[insert:]
    moved = dict(example)
    moved["input_ids"] = moved_prefix + query
    moved["fact_position"] = insert + fact_target_offset
    moved["final_query_position"] = len(moved_prefix)
    if len(moved["input_ids"]) != len(example["input_ids"]):
        raise AssertionError("Relocation changed the episode length")
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if settings.get("save_traces") is not True:
        parser.error("This diagnostic requires save_traces: true")
    dataset_settings = settings["dataset"]
    revision = dataset_settings["revision"]
    metadata_path = hf_hub_download(
        dataset_settings["repo_id"],
        f"{dataset_settings['variant']}/metadata.json",
        repo_type="dataset", revision=revision,
    )
    metadata = json.loads(Path(metadata_path).read_text())
    dataset = load_dataset(
        dataset_settings["repo_id"], dataset_settings["variant"],
        split=dataset_settings.get("split", "test"), revision=revision,
    )
    examples = select_examples(dataset, dataset_settings)
    checkpoint = str(args.checkpoint.resolve())
    ensure_manifest(args.output_dir / "manifest.json", {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint.resolve()),
        "config": settings,
        "dataset_revision": revision,
        "read_conditions": READ_CONDITIONS,
    })
    results = read_results(args.output_dir / "results.jsonl", id_field="evaluation_id")
    if not torch.cuda.is_available():
        raise RuntimeError("Read-path diagnostics require a CUDA GPU")
    seed_everything(settings["seed"])
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"], revision=metadata["tokenizer_revision"],
    )
    model = load_model(args.checkpoint).to(device).eval()
    layers = list(range(len(model.model.layers)))
    exporter = TaalTraceExporter(args.output_dir / "memory_traces", run_metadata={
        "checkpoint": checkpoint, "dataset_revision": revision,
        "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "layers": layers, "capture": "all_diagnostic_variants_all_layers",
    })
    block_size = settings.get("execution_block_size", model.config.working_memory_size)
    distances = settings["fact_distances"]
    if not distances or any(
        distance is not None and (type(distance) is not int or distance < 1)
        for distance in distances
    ):
        raise ValueError("fact_distances must contain positive integers or null")
    for example in examples:
        for distance in distances:
            label = "original" if distance is None else str(distance)
            moved = relocate_fact(example, tokenizer, distance)
            prefix, query = split_episode(moved)
            pending = [
                condition for condition in READ_CONDITIONS
                if f"{example['example_id']}/{label}/{condition[0]}" not in results
            ]
            if not pending:
                continue
            prefixes = {}
            for prefix_scale in sorted({condition[1] for condition in pending}):
                recorder = TaalTraceRecorder(model, layers, len(prefix))
                with recorder:
                    output = model.prefill(
                        torch.tensor([prefix], dtype=torch.long, device=device),
                        execution_block_size=block_size,
                        memory_read_scale=prefix_scale,
                        prepend_memory_tokens=True,
                    )
                prefixes[prefix_scale] = (output.state, recorder)
                del output
            for name, prefix_scale, query_scale in pending:
                base_state, prefix_trace = prefixes[prefix_scale]
                state = fork_session_state(base_state, base_state.memory_states)
                query_trace = TaalTraceRecorder(
                    model, layers, len(query), position_offset=len(prefix),
                )
                with query_trace:
                    output = model.prefill(
                        torch.tensor([query], dtype=torch.long, device=device),
                        execution_block_size=block_size, state=state,
                        memory_read_scale=query_scale,
                        prepend_memory_tokens=False,
                    )
                scores = score_logits(output.logits[0, -1].float(), moved, tokenizer)
                row = {
                    "evaluation_id": f"{example['example_id']}/{label}/{name}",
                    "example_id": example["example_id"],
                    "fact_id": int(example["fact_id"]),
                    "fact_distance": (
                        len(moved["input_ids"]) - int(moved["fact_position"])
                    ),
                    "distance_variant": label,
                    "read_condition": name,
                    "prefix_read_scale": prefix_scale,
                    "query_read_scale": query_scale,
                    "within_kv_window": (
                        len(moved["input_ids"]) - int(moved["fact_position"])
                        <= model.config.working_memory_size
                    ),
                    **scores,
                }
                prediction_trace = trace_greedy_prediction(
                    model,
                    tokenizer,
                    token_id=int(scores["vocabulary_top_token_id"]),
                    position=len(moved["input_ids"]),
                    state=output.state,
                    layers=layers,
                    memory_read_scale=query_scale,
                )
                exporter.export(TraceEpisode(
                    run_id=args.output_dir.name,
                    example_id=example["example_id"],
                    condition_id=f"{label}/{name}",
                    checkpoint=checkpoint,
                    tokenizer_id=metadata["tokenizer"],
                    tokens=[
                        *make_trace_tokens(moved["input_ids"], tokenizer),
                        prediction_trace.token,
                    ],
                    writes=(
                        prefix_trace.writes + query_trace.writes
                        + prediction_trace.writes
                    ),
                    reads=(
                        prefix_trace.reads + query_trace.reads
                        + prediction_trace.reads
                    ),
                    internal_prefixes=(
                        prefix_trace.internal_prefixes + query_trace.internal_prefixes
                        + prediction_trace.internal_prefixes
                    ),
                    outcome={
                        **row,
                        "candidate_choice_token": tokenizer.decode(
                            [int(row["candidate_choice_token_id"])]
                        ),
                    },
                    metadata={
                        "query_start_position": len(prefix),
                        "fact_position": moved["fact_position"],
                        "expected_answer": moved["answer"],
                        "expected_token_id": int(moved["target_token_id"]),
                        "prefix_read_scale": prefix_scale,
                        "query_read_scale": query_scale,
                    },
                ))
                append_result(args.output_dir / "results.jsonl", row)
                results[row["evaluation_id"]] = row
                print(f"Scored {row['evaluation_id']}", flush=True)
                del output, state
            del prefixes
    summary = grouped_score_summary(results.values(), ("distance_variant", "read_condition"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
