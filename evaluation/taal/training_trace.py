"""Compact native trajectories of every selected validation example at shutdown."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.taal.state_audit import select_examples
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import (
    TaalTraceExporter, TaalTraceRecorder, make_trace_tokens, trace_greedy_prediction,
)
from training.engine import precision_context


@torch.inference_mode()
def export_training_traces(model, config, checkpoint_directory: Path, progress) -> None:
    bounds = config.data.val
    dataset = [json.loads(line) for line in Path(config.data.dataset_id).read_text().splitlines()]
    examples = select_examples(dataset, {
        "start": bounds.start, "end": bounds.end,
        "conditions": bounds.conditions, "query_variants": bounds.query_variants,
    })
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, revision=config.model.revision)
    layers = sorted(model.model.initial_memory_states(batch_size=1))
    root = checkpoint_directory / "memory_traces"
    checkpoint = str(checkpoint_directory / "final")
    exporter = TaalTraceExporter(root, run_metadata={
        "checkpoint": checkpoint, "tokenizer": config.model.model_id,
        "tokenizer_revision": config.model.revision, "layers": layers,
        "optimizer_step": progress.step, "training_config": config.to_dict(),
        "precision": "training FP32 slow parameters with CUDA BF16 autocast",
        "capture": "Every selected validation example, native prompt plus greedy feedback",
    })
    was_training = model.training
    model.eval()
    outcomes = []
    try:
        for example in examples:
            ids = example["input_ids"]
            recorder = TaalTraceRecorder(model, layers, len(ids))
            mask = None
            if not config.data.query_fresh_writes_enabled:
                mask = torch.ones((1, len(ids)), dtype=torch.bool, device=model.device)
                mask[:, example["final_query_position"]:] = False
            with precision_context(model.device), recorder:
                output = model.prefill(
                    torch.tensor([ids], device=model.device), write_mask=mask,
                    execution_block_size=config.model.working_memory_size,
                )
            outcome = {"example_id": example["example_id"], "condition": "native",
                       **score_logits(output.logits[0, -1], example, tokenizer)}
            with precision_context(model.device):
                prediction = trace_greedy_prediction(
                    model, tokenizer, token_id=outcome["vocabulary_top_token_id"],
                    position=len(ids), state=output.state, layers=layers, memory_read_scale=1.0,
                    fresh_writes_enabled=config.data.query_fresh_writes_enabled,
                )
            exporter.export(TraceEpisode(
                checkpoint_directory.name, example["example_id"], "native", checkpoint,
                config.model.model_id, make_trace_tokens(ids, tokenizer) + [prediction.token],
                recorder.writes + prediction.writes, recorder.reads + prediction.reads,
                recorder.internal_prefixes + prediction.internal_prefixes, outcome,
                {"expected_answer": example["answer"], "expected_token_id": example["target_token_id"],
                 "query_start_position": example["final_query_position"],
                 "optimizer_step": progress.step, "training_set_control": True},
            ))
            outcomes.append(outcome)
        (checkpoint_directory / "trace_summary.json").write_text(
            json.dumps(grouped_score_summary(outcomes, ("condition",)), indent=2) + "\n"
        )
        print(f"TRAINING_TRACES_COMPLETE step={progress.step} episodes={len(outcomes)} path={root}", flush=True)
    finally:
        model.train(was_training)
