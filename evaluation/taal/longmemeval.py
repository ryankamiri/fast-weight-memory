"""TaaL generation and compact trajectories for the shared LongMemEval runner."""

import json
from pathlib import Path

import torch
from jaxtyping import Bool, Int

from evaluation.shared.generation import capture_generation, resolve_generation_settings
from evaluation.taal.trace_export import TaalTraceExporter, validate_generation_trace


def select_dataset(dataset, settings):
    start = settings.get("start", 0)
    end = settings.get("end")
    if end is None:
        end = len(dataset)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(dataset):
        raise ValueError("Dataset bounds must satisfy 0 <= start < end <= dataset length")

    return dataset.select(range(start, end))


def create_trace_exporter(output, checkpoint, model_config, metadata, settings, predictions):
    save_traces = settings.get("save_traces", True)
    if type(save_traces) is not bool:
        raise ValueError("save_traces must be boolean")

    if not save_traces:
        return None

    chat = settings.get("prompt_format", "completion") == "chat"
    exporter = TaalTraceExporter(Path(output) / "memory_traces", run_metadata={
        "checkpoint": checkpoint,
        "tokenizer": checkpoint if chat else metadata["tokenizer"],
        "tokenizer_revision": None if chat else metadata["tokenizer_revision"],
        "layers": list(range(model_config.num_hidden_layers)),
        "evaluation": "longmemeval",
        "settings": {key: value for key, value in settings.items() if key != "judge"},
        "model_config": json.loads(model_config.to_json_string(use_diff=False)),
        "capture": "Prompt and all generated tokens, including EOS; fresh state per question",
    })
    for question_id, prediction in predictions.items():
        validate_generation_trace(exporter.root, exporter.manifest, question_id, prediction)

    return exporter


@torch.inference_mode()
def generate_taal_example(model, tokenizer, example, settings, device, exporter=None):
    ids: Int[torch.Tensor, "1 S"] = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    _, S = ids.shape
    model_settings = settings.get("model", {})
    retain_prefix = model_settings.get("retain_task_prefix", False)
    persistent: Bool[torch.Tensor, "S"] | None = None
    if retain_prefix:
        if example["persistent_prefix_length"] > model.config.max_persistent_kv_tokens:
            raise ValueError("Prepared task prefix exceeds checkpoint persistent-token budget")

        persistent = torch.arange(S, device=device) < example["persistent_prefix_length"]
    generation = resolve_generation_settings(settings["generation"], model.config.working_memory_size)
    if "state" in generation:
        raise ValueError("LongMemEval questions must start with fresh state")

    generation.update({
        "persistent_mask": persistent,
        "memory_read_scale": model_settings.get("memory_read_scale", 1.0),
        "generator": torch.Generator(device=device).manual_seed(settings["seed"]),
    })
    captured = capture_generation(model, ids, generation, trace_memory=exporter is not None)
    generated_ids = captured.token_ids
    recorder = captured.recorder

    outcome = {
        "question_id": example["question_id"],
        "hypothesis": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "prompt_tokens": S,
        "generated_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "stop_reason": captured.stop_reason,
    }
    if exporter is not None:
        exporter.export_generation(
            example["question_id"], example["input_ids"], generated_ids,
            tokenizer, recorder, outcome,
            metadata={
                "expected_answer": example["answer"],
                "question": example.get("question"),
                "question_type": example.get("question_type"),
                "abstention": example.get("abstention"),
                "retain_task_prefix": retain_prefix,
                "memory_read_scale": generation["memory_read_scale"],
                "read_scale_scope": "whole_episode",
                "generation": settings["generation"],
                "judge_result": "Stored separately in judgments JSONL after grading",
                "timing_note": "A generated token's read/write events follow the prediction that chose it",
            },
        )
    return outcome
