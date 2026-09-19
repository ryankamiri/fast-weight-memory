from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32


@dataclass
class NeuralMemoryState:
    """Per-session fast MLP parameters, pending writes, and convolution history.

    Each index along B is one independent session trajectory. N memory chunks
    are recurrent because each starts from the previous chunk's final state.
    The C tokens inside a chunk share its starting weights and contribute to one
    aggregated update when the chunk completes.
    """

    # Final weights committed by the most recently completed chunk.
    weights: dict[str, Float32[torch.Tensor, "B D D"]]
    momentum: dict[str, Float32[torch.Tensor, "B D D"]]
    # Weighted surprise gradients accumulated for an incomplete chunk.
    pending_gradient: dict[str, Float32[torch.Tensor, "B D D"]] | None = None
    # Sum of its token inputs, used for chunk-level forgetting and momentum.
    pending_input_sum: Float32[torch.Tensor, "B D"] | None = None
    # Number of tokens already accumulated into the pending chunk.
    pending_count: int = 0
    query_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    key_conv_history: Float[torch.Tensor, "B D S_history"] | None = None
    value_conv_history: Float[torch.Tensor, "B D S_history"] | None = None

    def detach(self) -> "NeuralMemoryState":
        return NeuralMemoryState(
            weights={name: value.detach() for name, value in self.weights.items()},
            momentum={name: value.detach() for name, value in self.momentum.items()},
            pending_gradient=(
                None
                if self.pending_gradient is None
                else {
                    name: value.detach()
                    for name, value in self.pending_gradient.items()
                }
            ),
            pending_input_sum=(
                None
                if self.pending_input_sum is None
                else self.pending_input_sum.detach()
            ),
            pending_count=self.pending_count,
            query_conv_history=(
                None
                if self.query_conv_history is None
                else self.query_conv_history.detach()
            ),
            key_conv_history=(
                None
                if self.key_conv_history is None
                else self.key_conv_history.detach()
            ),
            value_conv_history=(
                None
                if self.value_conv_history is None
                else self.value_conv_history.detach()
            ),
        )
