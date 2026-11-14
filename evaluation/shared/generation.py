"""One generation/capture primitive used by shared and legacy entry points."""

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import torch
from jaxtyping import Int

from evaluation.taal.trace_export import TaalTraceRecorder


@dataclass
class GenerationCapture:
    token_ids: list[int]
    stop_reason: str
    seconds: float
    peak_allocated_bytes: int | None
    recorder: TaalTraceRecorder | None


@torch.inference_mode()
def capture_generation(
    model, ids: Int[torch.Tensor, "1 S"], settings: dict, trace_memory: bool = False
) -> GenerationCapture:
    device = ids.device
    _, S = ids.shape
    recorder = None
    if trace_memory:
        recorder = TaalTraceRecorder(
            model, list(range(model.config.num_hidden_layers)), S
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    with recorder if recorder is not None else nullcontext():
        generated = model.generate(ids, **settings)
        generated_ids = generated.token_ids[0].tolist()
        if recorder is not None:
            # Generation feeds every chosen token back through the model, including EOS.
            recorder.token_count = S + len(generated_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - started
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    return GenerationCapture(
        generated_ids, generated.stop_reason, seconds, peak, recorder
    )


def resolve_generation_settings(settings, working_memory_size):
    generation = dict(settings)
    block_size = generation.get("execution_block_size")
    if block_size is None:
        block_size = working_memory_size
    if type(block_size) is not int or not 1 <= block_size <= working_memory_size:
        raise ValueError(
            f"execution_block_size must be an integer in [1, {working_memory_size}], "
            f"or null to use the checkpoint KV window; got {block_size!r}"
        )

    generation["execution_block_size"] = block_size
    return generation
