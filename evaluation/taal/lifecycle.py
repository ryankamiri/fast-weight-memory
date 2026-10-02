"""Save acquisition/retention snapshots without putting the fact back into KV."""

from pathlib import Path

import torch

from evaluation.taal.interventions import memory_execution_controls
from evaluation.taal.prefix_store import PrefixStore, PrefixTrace
from evaluation.taal.trace_export import TaalTraceRecorder


def lifecycle_boundaries(example: dict, tokenizer) -> dict[str, int]:
    """Exclusive text-token boundaries in the declared paired conflict template."""
    fact_position = int(example["fact_position"])
    fact_start = tokenizer.encode(
        f"User: Remember that {example['record_name']}'s archive label is",
        add_special_tokens=False,
    )
    fact_end = tokenizer.encode(".\nAssistant: Understood.\n\n", add_special_tokens=False)
    expected_fact = fact_start + [int(example["target_token_id"])] + fact_end
    prefix_length = int(example["final_query_position"])
    if fact_position != len(fact_start) or example["input_ids"][:len(expected_fact)] != expected_fact:
        raise ValueError("Lifecycle audit requires the original paired conflict fact template")
    fact_boundary = len(expected_fact)
    if not 0 < fact_position + 1 < fact_boundary < prefix_length - 1:
        raise ValueError("Lifecycle boundaries need a complete fact and nonempty neutral gap")
    return {
        "label": fact_position + 1,
        "fact": fact_boundary,
        "middle": fact_boundary + (prefix_length - fact_boundary) // 2,
        "final": prefix_length,
    }


def lifecycle_stores(directory: Path, identity: dict) -> dict[str, PrefixStore]:
    return {
        stage: PrefixStore(directory / stage, {**identity, "snapshot_stage": stage})
        for stage in ("label", "fact", "middle", "gap_frozen")
    }


@torch.inference_mode()
def save_lifecycle_prefixes(
    model, tokenizer, example: dict, final_store: PrefixStore,
    stores: dict[str, PrefixStore], *, device, layers: list[int],
    execution_block_size: int, persistent_writes_enabled: bool,
    capture_gap_frozen: bool,
) -> None:
    boundaries = lifecycle_boundaries(example, tokenizer)
    ids = example["input_ids"][:boundaries["final"]]
    recorder = TaalTraceRecorder(model, layers, len(ids))
    state = None
    start = 0
    # Splitting at declared boundaries changes execution partitioning, never
    # memory_chunk_size. Prepend the persistent inputs only once per episode.
    with memory_execution_controls(model, persistent_writes_enabled=persistent_writes_enabled), recorder:
        for stage, end in boundaries.items():
            output = model.prefill(
                torch.tensor([ids[start:end]], dtype=torch.long, device=device),
                state=state, execution_block_size=execution_block_size,
                prepend_memory_tokens=start == 0,
            )
            state = output.state
            store = final_store if stage == "final" else stores[stage]
            store.save(example, state, PrefixTrace(
                writes=recorder.writes, reads=recorder.reads,
                internal_prefixes=recorder.internal_prefixes,
            ))
            start = end
    if capture_gap_frozen:
        acquired = stores["fact"].load(example, device)
        start = boundaries["fact"]
        gap_trace = TaalTraceRecorder(model, layers, len(ids) - start, position_offset=start)
        with memory_execution_controls(model, updates_enabled=False), gap_trace:
            output = model.prefill(
                torch.tensor([ids[start:]], dtype=torch.long, device=device),
                state=acquired.state, execution_block_size=execution_block_size,
                prepend_memory_tokens=False,
            )
        stores["gap_frozen"].save(example, output.state, PrefixTrace(
            writes=acquired.trace.writes + gap_trace.writes,
            reads=acquired.trace.reads + gap_trace.reads,
            internal_prefixes=acquired.trace.internal_prefixes + gap_trace.internal_prefixes,
        ))
