"""Scoped inference-only changes to one memory-transition term at a time."""

from contextlib import contextmanager, ExitStack
import math
from unittest.mock import patch

import torch

from architectures.titans.neural_memory import ChunkGradientResult


INTERVENTIONS = (
    "baseline",
    "no_momentum",
    "gradient_x0.1",
    "gradient_clip1",
    "forget_floor0.01",
)


def scaled_gradient(original, arm):
    def gradient(*args, **kwargs):
        result = original(*args, **kwargs)
        B = args[1].shape[0]
        if arm == "gradient_x0.1":
            scale = torch.full((B,), 0.1, device=args[1].device)
        else:
            squared = []
            for value in result.chunk_gradients.values():
                squared.append(value.double().flatten(1).square().sum(1))
            norm = torch.stack(squared).sum(0).sqrt()
            scale = torch.min(torch.ones_like(norm), 1 / norm.clamp_min(1e-30)).float()
        chunks = {}
        tokens = {} if result.token_gradients is not None else None
        for name, value in result.chunk_gradients.items():
            chunks[name] = value * scale.reshape((B,) + (1,) * (value.ndim - 1))
        if tokens is not None:
            for name, value in result.token_gradients.items():
                tokens[name] = value * scale.reshape((B,) + (1,) * (value.ndim - 1))
        return ChunkGradientResult(chunks, tokens)

    return gradient


def floor_forgetting(original):
    def forward(inputs):
        return original(inputs).clamp_min(math.log(0.01 / 0.99))

    return forward


@contextmanager
def intervene(model, arm):
    if arm not in INTERVENTIONS or model.training:
        raise ValueError("Declared inference-only intervention required")
    with ExitStack() as stack:
        for layer in model.model.layers:
            memory = layer.taal.neural_memory
            if arm == "no_momentum":
                stack.enter_context(patch.object(memory, "momentum_enabled", False))
            elif arm in ("gradient_x0.1", "gradient_clip1"):
                stack.enter_context(
                    patch.object(
                        memory,
                        "_chunk_gradient",
                        scaled_gradient(memory._chunk_gradient, arm),
                    )
                )
            elif arm == "forget_floor0.01":
                stack.enter_context(
                    patch.object(
                        memory.forget_projection,
                        "forward",
                        floor_forgetting(memory.forget_projection.forward),
                    )
                )
        yield
