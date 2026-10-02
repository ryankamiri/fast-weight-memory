import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import TypedDict

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.taal.qwen.state import NeuralMemoryStates, TaalModelState
from architectures.titans.state import NeuralMemoryState
from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.prefix_store import PrefixStore, PrefixTrace, checkpoint_fingerprint
from evaluation.taal.interventions import (
    AUDIT_CONDITIONS,
    audit_conditions,
    compose_memory_states,
    donor_label_scores,
    memory_execution_controls,
    validate_component_boundary,
)
from evaluation.taal.trace_contract import TraceComparison, TraceEpisode
from evaluation.taal.trace_export import (
    TaalTraceExporter,
    TaalTraceRecorder,
    make_trace_tokens,
    trace_greedy_prediction,
)
from utils.seed import seed_everything
from training.taal_timing import TaalEvaluationTimings, TaalKernelSampler


class StateAuditExample(TypedDict):
    example_id: str
    fact_id: int
    answer: str
    condition: str
    query_variant: str
    input_ids: list[int]
    final_query_position: int
    target_token_id: int
    candidate_token_ids: list[int]


def _map_optional(value, transform):
    return None if value is None else transform(value)


def map_neural_state(state: NeuralMemoryState, transform) -> NeuralMemoryState:
    """Apply one tensor transform while preserving the recurrent state structure."""
    return NeuralMemoryState(
        weights={name: transform(value) for name, value in state.weights.items()},
        momentum={name: transform(value) for name, value in state.momentum.items()},
        pending_gradient=(
            None
            if state.pending_gradient is None
            else {
                name: transform(value)
                for name, value in state.pending_gradient.items()
            }
        ),
        pending_input_sum=_map_optional(state.pending_input_sum, transform),
        pending_count=state.pending_count,
        query_conv_history=_map_optional(state.query_conv_history, transform),
        key_conv_history=_map_optional(state.key_conv_history, transform),
        value_conv_history=_map_optional(state.value_conv_history, transform),
    )


def clone_memory_states(
    memory_states: NeuralMemoryStates,
    device: torch.device | str | None = None,
) -> NeuralMemoryStates:
    def clone(value):
        value = value.detach()
        if device is not None:
            value = value.to(device)
        return value.clone()

    return {
        layer_index: map_neural_state(state, clone)
        for layer_index, state in memory_states.items()
    }


def select_memory_session(
    memory_states: NeuralMemoryStates,
    batch_index: int,
    device: torch.device | str | None = None,
) -> NeuralMemoryStates:
    def select(value):
        value = value[batch_index:batch_index + 1].detach()
        if device is not None:
            value = value.to(device)
        return value.clone()

    return {
        layer_index: map_neural_state(state, select)
        for layer_index, state in memory_states.items()
    }


def fork_session_state(
    state: TaalModelState,
    memory_states: NeuralMemoryStates,
) -> TaalModelState:
    """Keep this episode's KV while replacing only its NeuralMemory state."""
    if state.past_key_values is None:
        raise ValueError("State audit requires a populated KV cache")
    return TaalModelState(
        past_key_values=copy.deepcopy(state.past_key_values),
        tokens_seen=state.tokens_seen,
        memory_states=clone_memory_states(memory_states),
    )


def split_episode(
    example: StateAuditExample,
) -> tuple[list[int], list[int]]:
    input_ids = list(example["input_ids"])
    final_query_position = int(example["final_query_position"])
    if not 0 < final_query_position < len(input_ids):
        raise ValueError(
            "final_query_position must split a nonempty prefix and query suffix"
        )
    return input_ids[:final_query_position], input_ids[final_query_position:]


def select_examples(dataset, settings) -> list[StateAuditExample]:
    start = settings.get("start", 0)
    end = settings.get("end")
    conditions = set(settings.get("conditions") or ())
    query_variants = set(settings.get("query_variants") or ())
    selected = [
        example
        for example in dataset
        if int(example["fact_id"]) >= start
        and (end is None or int(example["fact_id"]) < end)
        and (not conditions or example["condition"] in conditions)
        and (not query_variants or example["query_variant"] in query_variants)
    ]
    selected.sort(key=lambda example: (
        int(example["fact_id"]),
        example["condition"],
        example["query_variant"],
    ))
    ids = [example["example_id"] for example in selected]
    if not selected:
        raise ValueError("State-audit selection is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("State-audit selection contains duplicate example IDs")
    for example in selected:
        split_episode(example)

    if end is not None and conditions and query_variants:
        expected = (end - start) * len(conditions) * len(query_variants)
        if len(selected) != expected:
            raise ValueError(
                f"Expected {expected} audit examples from the configured "
                f"record range and filters, received {len(selected)}"
            )
    return selected


def swap_sources(
    examples: list[StateAuditExample], group_field: str | None
) -> dict[str, StateAuditExample]:
    """Select the other member of a matched pair, or the standard cyclic donor."""
    if group_field is None:
        return {
            example["example_id"]: examples[(index + 1) % len(examples)]
            for index, example in enumerate(examples)
        }
    if not group_field:
        raise ValueError("swap_group_field must be a nonempty field name")

    groups: dict[str, list[StateAuditExample]] = {}
    for example in examples:
        group = example.get(group_field)
        if not isinstance(group, str) or not group:
            raise ValueError(f"Every example needs a nonempty {group_field!r}")
        groups.setdefault(group, []).append(example)

    donors: dict[str, StateAuditExample] = {}
    for group, pair in groups.items():
        if len(pair) != 2:
            raise ValueError(f"Swap group {group!r} must contain exactly two examples")
        first, second = pair
        if first["target_token_id"] == second["target_token_id"]:
            raise ValueError(f"Swap group {group!r} must have opposing answers")
        donors[first["example_id"]] = second
        donors[second["example_id"]] = first
    return donors


def example_batches_by_prefix_length(examples, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("state_bank_batch_size must be a positive integer")
    groups = {}
    for example in examples:
        prefix_ids, _ = split_episode(example)
        groups.setdefault(len(prefix_ids), []).append((example, prefix_ids))
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def paired_differences(rows, conditions=AUDIT_CONDITIONS):
    by_example = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[
            row["audit_condition"]
        ] = row
    comparisons = {}
    for condition in conditions:
        if condition.name == "correct_full":
            continue
        pairs = [
            (conditions["correct_full"], conditions[condition.name])
            for conditions in by_example.values()
            if "correct_full" in conditions and condition.name in conditions
        ]
        if not pairs:
            continue
        comparisons[condition.name] = {
            "examples": len(pairs),
            "correct_full_minus_condition_candidate_accuracy": sum(
                int(correct["candidate_correct"])
                - int(intervened["candidate_correct"])
                for correct, intervened in pairs
            )
            / len(pairs),
            "correct_full_minus_condition_target_log_probability": sum(
                correct["target_log_probability"]
                - intervened["target_log_probability"]
                for correct, intervened in pairs
            )
            / len(pairs),
        }
    return comparisons


def summarize(rows, conditions=AUDIT_CONDITIONS):
    rows = list(rows)
    summary = {
        "metrics": grouped_score_summary(rows, ("audit_condition",)),
        "paired_vs_correct_full": paired_differences(rows, conditions),
    }
    swapped = [row for row in rows if row["audit_condition"] == "swapped"]
    if swapped:
        eligible = [row for row in swapped if row["swapped_target_in_candidates"]]
        summary["swapped_source"] = {
            "examples": len(swapped),
            "candidate_eligible_examples": len(eligible),
            "candidate_choice_rate_when_eligible": (
                None
                if not eligible
                else sum(
                    row["candidate_choice_token_id"]
                    == row["swapped_target_token_id"]
                    for row in eligible
                )
                / len(eligible)
            ),
            "mean_log_probability": sum(
                row["swapped_target_log_probability"] for row in swapped
            )
            / len(swapped),
            "vocabulary_top_1_rate": sum(
                row["swapped_target_vocabulary_top_1"] for row in swapped
            )
            / len(swapped),
        }
    if rows and all("own_minus_donor_label_log_probability" in row for row in rows):
        summary["label_preferences"] = {}
        by_example = {}
        for row in rows:
            by_example.setdefault(row["example_id"], {})[row["audit_condition"]] = row
        for condition in conditions:
            selected = [row for row in rows if row["audit_condition"] == condition.name]
            if not selected:
                continue
            group_shifts = {}
            for row in selected:
                baseline = by_example[row["example_id"]].get("correct_full")
                if baseline is not None:
                    group_shifts.setdefault(row["record_group"], []).append(
                        baseline["own_minus_donor_label_log_probability"]
                        - row["own_minus_donor_label_log_probability"]
                    )
            summary["label_preferences"][condition.name] = {
                "examples": len(selected),
                "mean_own_minus_donor_label_log_probability": sum(
                    row["own_minus_donor_label_log_probability"] for row in selected
                ) / len(selected),
                "donor_vocabulary_top_1_rate": sum(
                    row["donor_target_vocabulary_top_1"] for row in selected
                ) / len(selected),
                "record_group_baseline_minus_variant_label_margin": {
                    str(group): sum(shifts) / len(shifts)
                    for group, shifts in group_shifts.items()
                },
            }
    return summary


def load_model(checkpoint: Path) -> TaalQwen3ForCausalLM:
    source = str(checkpoint.resolve())
    config = TaalQwen3Config.from_pretrained(source)
    model, loading = TaalQwen3ForCausalLM.from_pretrained(
        source,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    errors = {
        name: loading.get(name)
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading.get(name)
    }
    if errors:
        raise ValueError(f"Checkpoint did not load exactly: {errors}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/configs/taal/delayed_recall_state_audit.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    conditions = audit_conditions(settings.get("audit_suite", "standard"))
    persistent_writes_enabled = settings.get("persistent_writes_enabled", True)
    query_updates_enabled = settings.get("query_updates_enabled", True)
    for name, value in (("persistent_writes_enabled", persistent_writes_enabled),
                        ("query_updates_enabled", query_updates_enabled)):
        if type(value) is not bool:
            parser.error(f"{name} must be true or false")
    save_traces = settings.get("save_traces", True)
    if type(save_traces) is not bool:
        parser.error("save_traces must be true or false")
    state_bank_batch_size = settings.get("state_bank_batch_size", 1)
    if type(state_bank_batch_size) is not int or state_bank_batch_size < 1:
        parser.error("state_bank_batch_size must be a positive integer")
    if save_traces and state_bank_batch_size != 1:
        parser.error("save_traces requires state_bank_batch_size=1")
    dataset_settings = settings["dataset"]
    local_path = dataset_settings.get("path")
    if local_path is not None:
        local_path = Path(local_path)
        metadata = json.loads(local_path.with_suffix(".metadata.json").read_text())
        revision = hashlib.sha256(local_path.read_bytes()).hexdigest()
        dataset = load_dataset("json", data_files=str(local_path), split="train")
    else:
        revision = dataset_settings.get("revision")
        if revision is None:
            revision = HfApi().dataset_info(dataset_settings["repo_id"]).sha
        metadata_file = hf_hub_download(
            dataset_settings["repo_id"],
            f"{dataset_settings['variant']}/metadata.json",
            repo_type="dataset",
            revision=revision,
        )
        metadata = json.loads(Path(metadata_file).read_text())
        dataset = load_dataset(
            dataset_settings["repo_id"],
            dataset_settings["variant"],
            split=dataset_settings.get("split", "test"),
            revision=revision,
        )
    examples = select_examples(dataset, dataset_settings)
    if len(examples) < 2:
        raise ValueError("Swapped-state evaluation requires at least two episodes")
    swap_group_field = settings.get("swap_group_field")
    if swap_group_field is not None and not isinstance(swap_group_field, str):
        parser.error("swap_group_field must be a string")
    swapped_sources = swap_sources(examples, swap_group_field)
    example_ids = {example["example_id"] for example in examples}
    expected_result_ids = {
        f"{example_id}/{condition.name}"
        for example_id in example_ids
        for condition in conditions
    }
    checkpoint = str(args.checkpoint.resolve())
    checkpoint_sha256 = checkpoint_fingerprint(args.checkpoint.resolve())
    ensure_manifest(args.output_dir / "manifest.json", {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "config": settings,
        "dataset_revision": revision,
        "dataset_metadata": metadata,
        "conditions": [
            {key: value for key, value in asdict(condition).items()
             if key != "components" or value is not None}
            for condition in conditions
        ],
        # Pilot 1A treated each record as one segment. The resumed query must
        # therefore continue that segment rather than insert TaaL tokens again.
        "query_prepends_memory_tokens": False,
        "swapped_pairing": (
            "cyclic_next_episode" if swap_group_field is None
            else f"matched_by_{swap_group_field}"
        ),
    })
    results = read_results(
        args.output_dir / "results.jsonl",
        id_field="evaluation_id",
    )
    if not set(results) <= expected_result_ids:
        raise ValueError("Saved results contain unknown state-audit IDs")
    if save_traces and results:
        trace_manifest_path = args.output_dir / "memory_traces" / "manifest.json"
        if not trace_manifest_path.exists():
            raise ValueError("Saved results have no memory trace manifest")
        trace_index = json.loads(trace_manifest_path.read_text())["episodes"]
        missing_traces = set(results) - set(trace_index)
        if missing_traces:
            raise ValueError(
                "Saved results are missing memory traces; use a fresh output "
                "directory or resume the original run without tracing"
            )
    if set(results) == expected_result_ids:
        output = summarize(results.values(), conditions)
        (args.output_dir / "summary.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        print(json.dumps(output, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("TaaL state audit expects a CUDA GPU")

    seed_everything(settings["seed"])
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"],
        revision=metadata["tokenizer_revision"],
    )
    model = load_model(args.checkpoint).to(device).eval()
    if (settings.get("audit_suite", "standard") != "standard" or not query_updates_enabled):
        if model.config.memory_chunk_size != 1:
            raise ValueError("Component/fixed-weight audits require memory_chunk_size=1")
    print(f"Evaluating on {torch.cuda.get_device_name(device)}: {len(examples)} episodes, {len(conditions)} conditions each.", flush=True)
    timings = TaalEvaluationTimings(
        model, device, os.environ.get("TAAL_TIMING_FIRST_BATCH") == "1"
    )
    trace_layers = list(range(len(model.model.layers))) if save_traces else []
    trace_exporter = (
        TaalTraceExporter(
            args.output_dir / "memory_traces",
            run_metadata={
                "checkpoint": checkpoint,
                "dataset_revision": revision,
                "tokenizer": metadata["tokenizer"],
                "tokenizer_revision": metadata["tokenizer_revision"],
                "layers": trace_layers,
                "capture": "all_examples_all_layers_compact_scalars",
                **({
                    "evaluation_config": settings,
                    "checkpoint_sha256": checkpoint_sha256,
                } if "audit_suite" in settings else {}),
            },
        )
        if save_traces else None
    )
    execution_block_size = settings.get(
        "execution_block_size",
        model.config.working_memory_size,
    )
    prefix_store = PrefixStore(args.output_dir / "prefix_states", {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": model.config.to_dict(),
        "evaluation_config": settings,
        "dataset_revision": revision,
        "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
    })

    built = 0

    def ensure_prefixes(required_examples):
        nonlocal built
        # Prepare only this episode and its swap donor. A two-hour run can
        # produce scores without first completing the whole 64-prefix bank.
        missing = [example for example in required_examples if not prefix_store.contains(example)]
        for batch in example_batches_by_prefix_length(missing, state_bank_batch_size):
            batch_ids = torch.tensor(
                [prefix_ids for _, prefix_ids in batch], dtype=torch.long, device=device,
            )
            sampler = None
            if built == 0 and timings.enabled and os.environ.get("TAAL_KERNEL_SAMPLE") == "1":
                sampler = TaalKernelSampler(model, device, args.output_dir / "timing" / "bank-forward.json")
            prefix_trace = (
                TaalTraceRecorder(model, layers=trace_layers, token_count=len(batch[0][1]))
                if save_traces else None
            )
            with (
                memory_execution_controls(model, persistent_writes_enabled=persistent_writes_enabled),
                timings.phase("state_bank_prefill", tokens=len(batch[0][1]), detailed=built == 0),
                sampler if sampler is not None else nullcontext(),
                prefix_trace if prefix_trace is not None else nullcontext(),
            ):
                output = model.prefill(
                    batch_ids,
                    execution_block_size=execution_block_size,
                    memory_read_scale=1.0,
                    prepend_memory_tokens=True,
                )
            with timings.phase("prefix_snapshot_save"):
                for batch_index, (example, _) in enumerate(batch):
                    prefix_store.save(
                        example,
                        output.state,
                        None if prefix_trace is None else PrefixTrace(
                            writes=prefix_trace.writes,
                            reads=prefix_trace.reads,
                            internal_prefixes=prefix_trace.internal_prefixes,
                        ),
                        batch_index=batch_index,
                    )
            built += len(batch)
            print(f"Saved {built} new prefix states this run", flush=True)
            del output

    for example_index, example in enumerate(examples):
        pending = [
            condition
            for condition in conditions
            if f"{example['example_id']}/{condition.name}" not in results
        ]
        if not pending:
            continue
        swapped_example = swapped_sources[example["example_id"]]
        required_examples = [example]
        if any(condition.uses_donor for condition in pending):
            required_examples.append(swapped_example)
        ensure_prefixes(required_examples)
        prefix_ids, query_ids = split_episode(example)
        with timings.phase("prefix_snapshot_load"):
            saved_prefix = prefix_store.load(example, device)
        base_state = saved_prefix.state
        prefix_trace = saved_prefix.trace
        if save_traces and prefix_trace is None:
            raise ValueError("Saved prefix has no trace; use a fresh output directory")
        with timings.phase("control_states_and_swapped_copy"):
            reset = model.model.initial_memory_states(batch_size=1)
            zeroed = model.model.zero_memory_states(batch_size=1)
            swapped = (
                prefix_store.load_memory(swapped_example, device)
                if any(condition.uses_donor for condition in pending)
                else None
            )
        sources = {
            "correct": base_state.memory_states,
            "reset": reset,
            "zeroed": zeroed,
            "swapped": swapped,
        }

        for condition in pending:
            with timings.phase("fork_kv_and_memory"):
                if condition.components is not None:
                    memory = compose_memory_states(base_state.memory_states, swapped, condition.components)
                else:
                    memory = sources[condition.memory_source]
                if not query_updates_enabled:
                    validate_component_boundary(memory)
                state = fork_session_state(base_state, memory)
            start = time.perf_counter()
            query_trace = (
                TaalTraceRecorder(
                    model,
                    layers=trace_layers,
                    token_count=len(query_ids),
                    position_offset=len(prefix_ids),
                )
                if save_traces else None
            )
            with (
                memory_execution_controls(model, updates_enabled=query_updates_enabled),
                timings.phase(f"query/{condition.name}", tokens=len(query_ids), detailed=example_index == 0),
                query_trace if query_trace is not None else nullcontext(),
            ):
                output = model.prefill(
                    torch.tensor(
                        [query_ids],
                        dtype=torch.long,
                        device=device,
                    ),
                    execution_block_size=execution_block_size,
                    state=state,
                    memory_read_scale=condition.read_scale,
                    prepend_memory_tokens=False,
                )
            with timings.phase("scoring"):
                logits = output.logits[0, -1].float()
                swapped_target = int(swapped_example["target_token_id"])
                swapped_log_probability = float(
                    (
                        logits[swapped_target]
                        - torch.logsumexp(logits, dim=0)
                    ).item()
                )
                donor_scores = donor_label_scores(logits, int(example["target_token_id"]), swapped_target)
                scores = score_logits(logits, example, tokenizer)
            result = {
                "evaluation_id": f"{example['example_id']}/{condition.name}",
                "example_id": example["example_id"],
                "fact_id": int(example["fact_id"]),
                "dataset_condition": example["condition"],
                "query_variant": example["query_variant"],
                "audit_condition": condition.name,
                "memory_source": condition.memory_source,
                "memory_read_scale": condition.read_scale,
                "persistent_writes_enabled": persistent_writes_enabled,
                "query_updates_enabled": query_updates_enabled,
                "component_sources": (
                    asdict(condition.components) if condition.components is not None else {
                        name: {"correct": "own", "swapped": "donor"}.get(
                            condition.memory_source, condition.memory_source
                        )
                        for name in ("weights", "momentum", "convolution")
                    }
                ),
                "record_group": example.get(swap_group_field) if swap_group_field else None,
                "prefix_tokens": len(prefix_ids),
                "query_tokens": len(query_ids),
                "seconds": time.perf_counter() - start,
                "swapped_from_example_id": (
                    swapped_example["example_id"]
                    if condition.uses_donor
                    else None
                ),
                "swapped_target_token_id": (
                    swapped_target
                    if condition.uses_donor
                    else None
                ),
                "swapped_target_in_candidates": (
                    swapped_target in example["candidate_token_ids"]
                    if condition.uses_donor
                    else None
                ),
                "swapped_target_log_probability": (
                    swapped_log_probability
                    if condition.uses_donor
                    else None
                ),
                "swapped_target_vocabulary_top_1": (
                    int(logits.argmax().item()) == swapped_target
                    if condition.uses_donor
                    else None
                ),
                **scores,
                **donor_scores,
            }
            if save_traces:
                assert prefix_trace is not None and query_trace is not None
                assert trace_exporter is not None
                with (
                    memory_execution_controls(model, updates_enabled=query_updates_enabled),
                    timings.phase("trace_generated_token"),
                ):
                    prediction_trace = trace_greedy_prediction(
                        model,
                        tokenizer,
                        token_id=int(scores["vocabulary_top_token_id"]),
                        position=len(example["input_ids"]),
                        state=output.state,
                        layers=trace_layers,
                        memory_read_scale=condition.read_scale,
                    )
                with timings.phase("trace_export"):
                    trace_exporter.export(TraceEpisode(
                        run_id=args.output_dir.name,
                        example_id=example["example_id"],
                        condition_id=condition.name,
                        checkpoint=checkpoint,
                        tokenizer_id=metadata["tokenizer"],
                        tokens=[
                            *make_trace_tokens(example["input_ids"], tokenizer),
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
                            **result,
                            "candidate_choice_token": tokenizer.decode(
                                [int(result["candidate_choice_token_id"])]
                            ),
                        },
                        metadata={
                            "query_start_position": len(prefix_ids),
                            "expected_answer": example.get("answer"),
                            "expected_token_id": int(example["target_token_id"]),
                            "memory_source_at_query": condition.memory_source,
                            "read_scale_at_query": condition.read_scale,
                            "persistent_writes_enabled": persistent_writes_enabled,
                            "query_updates_enabled": query_updates_enabled,
                            "component_sources": result["component_sources"],
                            "record_group": result["record_group"],
                            "swapped_from_example_id": (
                                swapped_example["example_id"]
                                if condition.uses_donor else None
                            ),
                        },
                    ))
            with timings.phase("result_append"):
                append_result(args.output_dir / "results.jsonl", result)
            results[result["evaluation_id"]] = result
            print(
                f"Scored {len(results)}/{len(expected_result_ids)}: "
                f"{result['evaluation_id']}",
                flush=True,
            )
            del output, state
        if save_traces:
            assert trace_exporter is not None
            baseline = results[f"{example['example_id']}/correct_full"]
            for condition in conditions:
                if condition.name == "correct_full":
                    continue
                variant = results[f"{example['example_id']}/{condition.name}"]
                baseline_logp = baseline["target_log_probability"]
                variant_logp = variant["target_log_probability"]
                trace_exporter.export_comparison(TraceComparison(
                    run_id=args.output_dir.name,
                    example_id=example["example_id"],
                    baseline_condition="correct_full",
                    variant_condition=condition.name,
                    intervention=(
                        "read_scale"
                        if condition.memory_source == "correct" else "memory_state"
                    ),
                    scope="whole_query",
                    identical_text_prefix=True,
                    same_starting_kv=True,
                    same_starting_memory=condition.memory_source == "correct",
                    scored_position=len(example["input_ids"]) - 1,
                    scored_token_id=int(example["target_token_id"]),
                    baseline_log_probability=baseline_logp,
                    variant_log_probability=variant_logp,
                    difference_log_probability=baseline_logp - variant_logp,
                ))
        del saved_prefix, base_state, sources, reset, zeroed, swapped

    output = summarize(results.values(), conditions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
