import argparse
import copy
from dataclasses import dataclass
import json
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
from utils.seed import seed_everything


@dataclass(frozen=True)
class AuditCondition:
    name: str
    memory_source: str
    read_scale: float


class StateAuditExample(TypedDict):
    example_id: str
    fact_id: int
    condition: str
    query_variant: str
    input_ids: list[int]
    final_query_position: int
    target_token_id: int
    candidate_token_ids: list[int]


AUDIT_CONDITIONS = (
    AuditCondition("correct_full", "correct", 1.0),
    AuditCondition("correct_half", "correct", 0.5),
    AuditCondition("reads_disabled", "correct", 0.0),
    AuditCondition("reset_initial", "reset", 1.0),
    AuditCondition("zeroed", "zeroed", 1.0),
    AuditCondition("swapped", "swapped", 1.0),
)


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


def paired_differences(rows):
    by_example = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[
            row["audit_condition"]
        ] = row
    comparisons = {}
    for condition in AUDIT_CONDITIONS:
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


def summarize(rows):
    rows = list(rows)
    summary = {
        "metrics": grouped_score_summary(rows, ("audit_condition",)),
        "paired_vs_correct_full": paired_differences(rows),
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
    dataset_settings = settings["dataset"]
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
    example_ids = {example["example_id"] for example in examples}
    expected_result_ids = {
        f"{example_id}/{condition.name}"
        for example_id in example_ids
        for condition in AUDIT_CONDITIONS
    }
    checkpoint = str(args.checkpoint.resolve())
    ensure_manifest(args.output_dir / "manifest.json", {
        "checkpoint": checkpoint,
        "config": settings,
        "dataset_revision": revision,
        "dataset_metadata": metadata,
        "conditions": [vars(condition) for condition in AUDIT_CONDITIONS],
        # Pilot 1A treated each record as one segment. The resumed query must
        # therefore continue that segment rather than insert TaaL tokens again.
        "query_prepends_memory_tokens": False,
        "swapped_pairing": "cyclic_next_episode",
    })
    results = read_results(
        args.output_dir / "results.jsonl",
        id_field="evaluation_id",
    )
    if not set(results) <= expected_result_ids:
        raise ValueError("Saved results contain unknown state-audit IDs")
    if set(results) == expected_result_ids:
        output = summarize(results.values())
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
    execution_block_size = settings.get(
        "execution_block_size",
        model.config.working_memory_size,
    )
    state_bank_batch_size = settings.get("state_bank_batch_size", 1)

    # Phase 1 retains only compact NeuralMemory states on CPU. Saving every
    # episode's full 2K KV cache would dominate memory, so phase 2 reconstructs
    # one episode's own KV immediately before forking its audit conditions.
    memory_bank = {}
    built = 0
    for batch in example_batches_by_prefix_length(
        examples,
        state_bank_batch_size,
    ):
        batch_ids = torch.tensor(
            [prefix_ids for _, prefix_ids in batch],
            dtype=torch.long,
            device=device,
        )
        output = model.prefill(
            batch_ids,
            execution_block_size=execution_block_size,
            memory_read_scale=1.0,
            prepend_memory_tokens=True,
        )
        for batch_index, (example, _) in enumerate(batch):
            memory_bank[example["example_id"]] = select_memory_session(
                output.state.memory_states,
                batch_index,
                device="cpu",
            )
        built += len(batch)
        print(
            f"Built memory states {built}/{len(examples)}",
            flush=True,
        )
        del output

    for example_index, example in enumerate(examples):
        pending = [
            condition
            for condition in AUDIT_CONDITIONS
            if f"{example['example_id']}/{condition.name}" not in results
        ]
        if not pending:
            continue
        prefix_ids, query_ids = split_episode(example)
        prefix_output = model.prefill(
            torch.tensor(
                [prefix_ids],
                dtype=torch.long,
                device=device,
            ),
            execution_block_size=execution_block_size,
            memory_read_scale=1.0,
            prepend_memory_tokens=True,
        )
        base_state = prefix_output.state
        reset = model.model.initial_memory_states(batch_size=1)
        zeroed = model.model.zero_memory_states(batch_size=1)
        swapped_example = examples[(example_index + 1) % len(examples)]
        swapped = clone_memory_states(
            memory_bank[swapped_example["example_id"]],
            device=device,
        )
        sources = {
            "correct": base_state.memory_states,
            "reset": reset,
            "zeroed": zeroed,
            "swapped": swapped,
        }

        for condition in pending:
            state = fork_session_state(base_state, sources[condition.memory_source])
            start = time.perf_counter()
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
            logits = output.logits[0, -1].float()
            swapped_target = int(swapped_example["target_token_id"])
            swapped_log_probability = float(
                (
                    logits[swapped_target]
                    - torch.logsumexp(logits, dim=0)
                ).item()
            )
            result = {
                "evaluation_id": f"{example['example_id']}/{condition.name}",
                "example_id": example["example_id"],
                "fact_id": int(example["fact_id"]),
                "dataset_condition": example["condition"],
                "query_variant": example["query_variant"],
                "audit_condition": condition.name,
                "memory_source": condition.memory_source,
                "memory_read_scale": condition.read_scale,
                "prefix_tokens": len(prefix_ids),
                "query_tokens": len(query_ids),
                "seconds": time.perf_counter() - start,
                "swapped_from_example_id": (
                    swapped_example["example_id"]
                    if condition.memory_source == "swapped"
                    else None
                ),
                "swapped_target_token_id": (
                    swapped_target
                    if condition.memory_source == "swapped"
                    else None
                ),
                "swapped_target_in_candidates": (
                    swapped_target in example["candidate_token_ids"]
                    if condition.memory_source == "swapped"
                    else None
                ),
                "swapped_target_log_probability": (
                    swapped_log_probability
                    if condition.memory_source == "swapped"
                    else None
                ),
                "swapped_target_vocabulary_top_1": (
                    int(logits.argmax().item()) == swapped_target
                    if condition.memory_source == "swapped"
                    else None
                ),
                **score_logits(logits, example, tokenizer),
            }
            append_result(args.output_dir / "results.jsonl", result)
            results[result["evaluation_id"]] = result
            print(
                f"Scored {len(results)}/{len(expected_result_ids)}: "
                f"{result['evaluation_id']}",
                flush=True,
            )
            del output, state
        del prefix_output, base_state

    output = summarize(results.values())
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
