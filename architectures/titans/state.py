from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32


@dataclass
class NeuralMemoryState:
    """Per-session fast MLP parameters, momentum, and convolution history.

    Each index along B is one independent session trajectory. Consecutive chunks
    of a session reuse that batch element's state; they are not parallelized as
    separate batch elements because later chunks depend on earlier writes.
    """

    weights: dict[str, Float32[torch.Tensor, "B D D"]]
    momentum: dict[str, Float32[torch.Tensor, "B D D"]]
    query_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    key_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    value_conv_history: Float[torch.Tensor, "B D S_history"] | None = None

    def detach(self) -> "NeuralMemoryState":
        return NeuralMemoryState(
            weights={name: value.detach() for name, value in self.weights.items()},
            momentum={name: value.detach() for name, value in self.momentum.items()},
            query_conv_history=(
                None if self.query_conv_history is None else self.query_conv_history.detach()
            ),
            key_conv_history=(
                None if self.key_conv_history is None else self.key_conv_history.detach()
            ),
            value_conv_history=(
                None if self.value_conv_history is None else self.value_conv_history.detach()
            ),
        )
