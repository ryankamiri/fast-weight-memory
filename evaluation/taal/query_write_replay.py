"""Evaluation-only replay of native prompt-query memory writes.

Read ablations can change later layers' write inputs. Replaying native keys,
values and update-control inputs keeps W/momentum identical at each query token
without freezing them. Query addresses and downstream hidden states stay live.
"""

from dataclasses import dataclass
from types import MethodType

from jaxtyping import Bool, Float, Float32
import torch

from architectures.titans.state import NeuralMemoryState


@dataclass
class QueryWriteInputs:
    keys: Float32[torch.Tensor, "B C D"]
    values: Float32[torch.Tensor, "B C D"]
    inputs: Float[torch.Tensor, "B C D"]
    write_mask: Bool[torch.Tensor, "B C"]
    weights_after: dict[str, Float32[torch.Tensor, "B D D"]]
    momentum_after: dict[str, Float32[torch.Tensor, "B D D"]]


class QueryWriteReplay:
    def __init__(self, model, token_count: int, reference: "QueryWriteReplay | None" = None):
        if model.training or model.config.memory_chunk_size != 1:
            raise ValueError("Query write replay requires eval and memory_chunk_size=1")
        if type(token_count) is not int or token_count < 1:
            raise ValueError("Query write replay requires a positive token count")
        self.model = model
        self.token_count = token_count
        self.reference = reference
        self.events: dict[int, list[QueryWriteInputs]] = {
            layer: [] for layer in range(len(model.model.layers))
        }
        self.counts = {layer: 0 for layer in self.events}
        self.originals = {}
        if reference is not None and (
            reference.token_count != token_count
            or set(reference.events) != set(self.events)
            or any(len(events) != token_count for events in reference.events.values())
        ):
            raise ValueError("Reference query writes must cover every token and memory layer")

    def _wrap(self, layer, original):
        def process(memory, state: NeuralMemoryState, queries, keys, values, inputs, write_mask, output_dtype):
            B, C, D = queries.shape
            index = self.counts[layer]
            if (B, C) != (1, 1) or index >= self.token_count:
                raise ValueError("Query write replay requires exactly B=C=1 per recorded token")
            native = None
            if self.reference is not None:
                native = self.reference.events[layer][index]
                keys = native.keys
                values = native.values
                inputs = native.inputs
                write_mask = native.write_mask
            # Keep the live query/address and convolution histories. Only the
            # writer's post-convolution k/v and learned-control inputs replay.
            output, after = original(state, queries, keys, values, inputs, write_mask, output_dtype)
            if native is not None:
                for name, expected in native.weights_after.items():
                    if not torch.equal(after.weights[name], expected):
                        raise ValueError(f"Layer {layer}, query token {index}: replayed weights differ")
                    if not torch.equal(after.momentum[name], native.momentum_after[name]):
                        raise ValueError(f"Layer {layer}, query token {index}: replayed momentum differs")
            else:
                self.events[layer].append(QueryWriteInputs(
                    keys=keys.detach().clone(), values=values.detach().clone(),
                    inputs=inputs.detach().clone(), write_mask=write_mask.detach().clone(),
                    weights_after={name: value.detach().clone() for name, value in after.weights.items()},
                    momentum_after={name: value.detach().clone() for name, value in after.momentum.items()},
                ))
            self.counts[layer] += 1
            return output, after
        return process

    def __enter__(self):
        for layer in self.events:
            memory = self.model.model.layers[layer].taal.neural_memory
            self.originals[layer] = memory._process_chunk
            memory._process_chunk = MethodType(self._wrap(layer, memory._process_chunk), memory)
        return self

    def __exit__(self, error_type, *_):
        for layer, original in self.originals.items():
            self.model.model.layers[layer].taal.neural_memory._process_chunk = original
        if error_type is None and any(count != self.token_count for count in self.counts.values()):
            raise ValueError("Query write replay did not consume every token at every layer")
