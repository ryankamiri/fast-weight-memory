"""Scoped evaluation policies and coherent, cloned partial-state interventions."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import torch

from architectures.taal.qwen.state import NeuralMemoryStates


StateSource = Literal["own", "donor"]


@dataclass(frozen=True)
class MemoryComponentSources:
    weights: StateSource = "own"
    momentum: StateSource = "own"
    convolution: StateSource = "own"


@dataclass(frozen=True)
class AuditCondition:
    name: str
    memory_source: str
    read_scale: float
    components: MemoryComponentSources | None = None

    @property
    def uses_donor(self) -> bool:
        return self.memory_source in ("swapped", "components")


AUDIT_CONDITIONS = (
    AuditCondition("correct_full", "correct", 1.0),
    AuditCondition("correct_half", "correct", 0.5),
    AuditCondition("reads_disabled", "correct", 0.0),
    AuditCondition("reset_initial", "reset", 1.0),
    AuditCondition("zeroed", "zeroed", 1.0),
    AuditCondition("swapped", "swapped", 1.0),
)


def audit_conditions(suite: str) -> tuple[AuditCondition, ...]:
    if suite == "standard":
        return AUDIT_CONDITIONS
    baseline = (AUDIT_CONDITIONS[0], AUDIT_CONDITIONS[2])
    if suite == "weights_vs_rest":
        return baseline + (
            AuditCondition(
                "weights_swapped", "components", 1.0,
                MemoryComponentSources(weights="donor"),
            ),
            AuditCondition(
                "remainder_swapped", "components", 1.0,
                MemoryComponentSources(momentum="donor", convolution="donor"),
            ),
            AUDIT_CONDITIONS[-1],
        )
    if suite == "factorial":
        conditions = list(baseline)
        names = ("weights", "momentum", "convolution")
        for sources in product(("own", "donor"), repeat=3):
            if sources == ("own", "own", "own"):
                continue
            if sources == ("donor", "donor", "donor"):
                conditions.append(AUDIT_CONDITIONS[-1])
                continue
            swapped_names = [
                name for name, source in zip(names, sources) if source == "donor"
            ]
            conditions.append(
                AuditCondition(
                    "_".join(swapped_names) + "_swapped", "components", 1.0,
                    MemoryComponentSources(*sources),
                )
            )
        return tuple(conditions)
    raise ValueError(f"Unknown audit_suite: {suite!r}")


def validate_component_boundary(states: NeuralMemoryStates) -> None:
    for layer, state in states.items():
        if (
            state.pending_count != 0
            or state.pending_gradient is not None
            or state.pending_input_sum is not None
        ):
            raise ValueError(f"Layer {layer}: component audit requires no pending writes")


def compose_memory_states(
    own: NeuralMemoryStates,
    donor: NeuralMemoryStates,
    sources: MemoryComponentSources,
) -> NeuralMemoryStates:
    """Choose each whole component at every layer; never alias either snapshot."""
    if set(own) != set(donor):
        raise ValueError("Own and donor memory layers must match")
    validate_component_boundary(own)
    validate_component_boundary(donor)
    for source in (sources.weights, sources.momentum, sources.convolution):
        if source not in ("own", "donor"):
            raise ValueError(f"Unknown component source: {source!r}")
    bank = {"own": own, "donor": donor}
    mixed = {}
    for layer, own_state in own.items():
        donor_state = donor[layer]
        for name in ("weights", "momentum"):
            left, right = getattr(own_state, name), getattr(donor_state, name)
            if set(left) != set(right) or any(
                left[key].shape != right[key].shape or left[key].dtype != right[key].dtype
                for key in left
            ):
                raise ValueError(f"Layer {layer}: incompatible {name}")
        histories = ("query_conv_history", "key_conv_history", "value_conv_history")
        for name in histories:
            left, right = getattr(own_state, name), getattr(donor_state, name)
            if (left is None) != (right is None) or (
                left is not None and (left.shape != right.shape or left.dtype != right.dtype)
            ):
                raise ValueError(f"Layer {layer}: incompatible {name}")
        conv_state = bank[sources.convolution][layer]
        history_values = {}
        for name in histories:
            history = getattr(conv_state, name)
            history_values[name] = None if history is None else history.detach().clone()
        mixed[layer] = replace(
            own_state,
            weights={
                name: value.detach().clone()
                for name, value in bank[sources.weights][layer].weights.items()
            },
            momentum={
                name: value.detach().clone()
                for name, value in bank[sources.momentum][layer].momentum.items()
            },
            **history_values,
        )
    return mixed


@contextmanager
def memory_execution_controls(
    model, *, persistent_writes_enabled: bool = True, updates_enabled: bool = True,
):
    """Restore runtime controls even on errors; never mutate checkpoint parameters."""
    if model.training:
        raise ValueError("Memory execution controls require model.eval()")
    if type(persistent_writes_enabled) is not bool or type(updates_enabled) is not bool:
        raise ValueError("Memory execution controls must be boolean")
    layers = [layer.taal for layer in model.model.layers]
    if not updates_enabled and any(layer.neural_memory.config.chunk_size != 1 for layer in layers):
        raise ValueError("Fixed memory requires memory_chunk_size=1")
    previous = [
        (layer.persistent_writes_enabled, layer.neural_memory.updates_enabled)
        for layer in layers
    ]
    try:
        for layer in layers:
            layer.persistent_writes_enabled = persistent_writes_enabled
            layer.neural_memory.updates_enabled = updates_enabled
        yield
    finally:
        for layer, (persistent, updates) in zip(layers, previous):
            layer.persistent_writes_enabled = persistent
            layer.neural_memory.updates_enabled = updates


def donor_label_scores(logits: torch.Tensor, target: int, donor_target: int) -> dict:
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite query logits in memory attribution audit")
    log_probabilities = logits.float().log_softmax(dim=0)
    return {
        "donor_target_token_id": donor_target,
        "donor_target_log_probability": float(log_probabilities[donor_target].item()),
        "own_minus_donor_label_log_probability": float(
            (log_probabilities[target] - log_probabilities[donor_target]).item()
        ),
        "donor_target_vocabulary_top_1": int(logits.argmax().item()) == donor_target,
    }
