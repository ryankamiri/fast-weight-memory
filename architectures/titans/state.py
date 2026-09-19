from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32


@dataclass
class NeuralMemoryState:
    """Per-session fast MLP parameters, momentum, and convolution history.

    Each index along B is one independent session trajectory. N memory chunks
    are recurrent because each starts from the previous chunk's final state;
    the C tokens inside a chunk share the committed weights. Provisional weights
    hold the causal prefix state until that chunk is complete.
    """

    # Final weights committed by the most recently completed chunk.
    weights: dict[str, Float32[torch.Tensor, "B D D"]]
    momentum: dict[str, Float32[torch.Tensor, "B D D"]]
    # Latest weight prefix for an open chunk; committed when it reaches C tokens.
    provisional_weights: dict[str, Float32[torch.Tensor, "B D D"]] | None = None
    # Number of tokens already incorporated into provisional_weights.
    pending_count: int = 0
    query_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    key_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    value_conv_history: Float[torch.Tensor, "B D S_history"] | None = None

    @property
    def current_weights(self) -> dict[str, Float32[torch.Tensor, "B D D"]]:
        if self.provisional_weights is not None:
            return self.provisional_weights
        return self.weights

    def detach(self) -> "NeuralMemoryState":
        return NeuralMemoryState(
            weights={name: value.detach() for name, value in self.weights.items()},
            momentum={name: value.detach() for name, value in self.momentum.items()},
            provisional_weights=(
                None
                if self.provisional_weights is None
                else {
                    name: value.detach()
                    for name, value in self.provisional_weights.items()
                }
            ),
            pending_count=self.pending_count,
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
