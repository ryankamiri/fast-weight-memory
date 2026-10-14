"""Locate usable L27 information without changing the background trajectory.

Saved W is tested at one final read, with a fixed normalized q. A separate
shadow writer crosses initial neural state and pre-k/v-projection inputs.
Neither shadow state nor substituted read W enters the live recurrence.
"""

from dataclasses import dataclass
from types import MethodType

from jaxtyping import Float, Float32
import torch

from architectures.titans.neural_memory import NeuralMemory
from architectures.titans.state import NeuralMemoryState
from evaluation.taal.query_write_replay import QueryWriteReplay


@dataclass
class ShadowWriterResult:
    state: NeuralMemoryState
    writes: list[dict]


def assert_same_state(actual: NeuralMemoryState, expected: NeuralMemoryState) -> None:
    """Same-source shadow endpoints must reproduce every recurrent component."""
    if (actual.pending_count or expected.pending_count
            or actual.pending_gradient is not None or expected.pending_gradient is not None
            or actual.pending_input_sum is not None or expected.pending_input_sum is not None):
        raise ValueError("Shadow writer endpoints must be completed C=1 states")
    for component in ("weights", "momentum"):
        left, right = getattr(actual, component), getattr(expected, component)
        if left.keys() != right.keys() or any(not torch.equal(left[name], right[name]) for name in right):
            raise ValueError(f"Shadow writer native endpoint differs: {component}")
    for component in ("query_conv_history", "key_conv_history", "value_conv_history"):
        left, right = getattr(actual, component), getattr(expected, component)
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"Shadow writer native endpoint differs: {component}")
        elif not torch.equal(left, right):
            raise ValueError(f"Shadow writer native endpoint differs: {component}")


@torch.inference_mode()
def evolve_shadow_writer(
    memory: NeuralMemory,
    initial: NeuralMemoryState,
    reference: QueryWriteReplay,
    layer: int,
    *,
    position_offset: int,
) -> ShadowWriterResult:
    """Replay inputs BEFORE k/v projection and convolution, not donor k/v.

    M_t = (1-alpha_t) M_{t-1} + S_t and S_t = eta_t S_{t-1} - g_t
    are recomputed from this branch's W, momentum and convolution history. Only
    the incoming representation stream is borrowed. Shadow reads are discarded.
    Native references must process the query in one forward block, as below.
    """
    if memory.training or memory.config.chunk_size != 1 or initial.pending_count:
        raise ValueError("Shadow writes require eval, C=1 and a completed initial state")
    if memory.trace_observer is not None:
        raise ValueError("Shadow writes must run outside a live trace observer")
    events = reference.events[layer]
    if len(events) != reference.token_count or not events:
        raise ValueError("Shadow writer reference must cover the full query")
    writer_inputs: Float[torch.Tensor, "B S D"] = torch.cat([event.inputs for event in events], dim=1)
    write_mask = torch.cat([event.write_mask for event in events], dim=1)
    B, S, D = writer_inputs.shape
    if B != 1 or S != reference.token_count or not torch.isfinite(writer_inputs).all():
        raise ValueError("Invalid shadow writer inputs")
    writes = []

    def record(before, token_gradients, after, mask, strength, *, updates_enabled=True):
        index = len(writes)
        if not updates_enabled or after.pending_count or index >= S:
            raise ValueError("Shadow writer must commit exactly one update per token")
        gradient = torch.cat([value.flatten() for value in token_gradients.values()])
        previous = torch.cat([value.flatten() for value in before.weights.values()])
        current = torch.cat([value.flatten() for value in after.weights.values()])
        movement = current - previous
        metrics = torch.stack((gradient.norm(), previous.norm(), movement.norm(),
                               (movement + gradient).norm(), strength[0, 0])).float()
        if not torch.isfinite(metrics).all():
            raise ValueError("Nonfinite shadow writer metrics")
        proposed, old, net, other, rate = metrics.cpu().tolist()
        writes.append({
            "position": position_offset + index, "layer": layer,
            "proposed_write_norm": proposed, "previous_weight_norm": old,
            "net_weight_change_norm": net, "other_movement_norm": other,
            "write_strength": rate, "write_enabled": bool(mask[0, 0]),
        })

    memory.trace_observer = record
    try:
        _, state = memory(writer_inputs, state=initial, write_mask=write_mask)
    finally:
        memory.trace_observer = None
    if len(writes) != S:
        raise ValueError("Incomplete shadow writer trajectory")
    return ShadowWriterResult(state=state, writes=writes)


class FinalReadReplay:
    """Read frozen/saved W at the prediction token; keep all live state intact."""

    def __init__(
        self, model, layer: int, token_count: int,
        weights: dict[str, Float32[torch.Tensor, "B D D"]],
        query: Float[torch.Tensor, "B 1 D"],
    ):
        if (model.training or model.config.memory_chunk_size != 1
                or layer != len(model.model.layers) - 1
                or type(token_count) is not int or token_count < 1):
            raise ValueError("Final read replay requires eval, C=1 and the last layer")
        self.memory = model.model.layers[layer].taal.neural_memory
        self.token_count = token_count
        self.weights = weights
        self.query = query
        self.count = 0
        self.events: list[dict] = []
        self.original = None

    def __enter__(self):
        self.original = self.memory._process_chunk

        def process(memory, state, queries, keys, values, inputs, write_mask, output_dtype):
            B, C, D = queries.shape
            index = self.count
            if (B, C) != (1, 1) or index >= self.token_count:
                raise ValueError("Final read replay requires exactly B=C=1 per token")
            output, after = self.original(state, queries, keys, values, inputs, write_mask, output_dtype)
            self.count += 1
            if index != self.token_count - 1:
                return output, after
            if not torch.equal(queries, self.query):
                raise ValueError("Final reader query changed before the intervention")
            patched = memory.memory_mlp(self.query.float(), weights=self.weights).predicted_values.to(output_dtype)
            if not torch.isfinite(patched).all():
                raise ValueError("Nonfinite saved-weight read")
            self.events.append({
                "query_index": index,
                "natural_read_norm": float(output.float().norm()),
                "patched_read_norm": float(patched.float().norm()),
                "read_difference_norm": float((patched.float() - output.float()).norm()),
                "identical_query_verified": True,
            })
            return patched, after

        self.memory._process_chunk = MethodType(process, self.memory)
        return self

    def __exit__(self, error_type, *_):
        self.memory._process_chunk = self.original
        if error_type is None and (self.count != self.token_count or len(self.events) != 1):
            raise ValueError("Final read replay did not cover exactly one full query")
