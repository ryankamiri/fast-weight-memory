import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn


class CausalDepthwiseConv1d(nn.Conv1d):
    """Depthwise causal convolution with caller-owned streaming history."""

    def __init__(self, channels: int, kernel_size: int, bias: bool = False):
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(kernel_size) is not int or kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        super().__init__(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=bias,
        )

    @property
    def history_size(self) -> int:
        return self.kernel_size[0] - 1

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[torch.Tensor, "B S D"],
        # S_history = kernel_size - 1 retained inputs; the current input
        # completes the convolution's full kernel_size-position window.
        history: Float[torch.Tensor, "B D S_history"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B S D"],
        Float[torch.Tensor, "B D S_history"],
    ]:
        batch, _, channels = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, received {channels}"
            )

        channel_first = inputs.transpose(1, 2)
        if history is None:
            history = channel_first.new_zeros(batch, channels, self.history_size)
        elif history.shape != (batch, channels, self.history_size):
            raise ValueError(
                "history must have shape "
                f"{(batch, channels, self.history_size)}, received {tuple(history.shape)}"
            )

        combined = torch.cat((history, channel_first), dim=-1)
        outputs: Float[torch.Tensor, "B S D"] = super().forward(combined).transpose(1, 2)
        if self.history_size == 0:
            next_history = combined[..., :0]
        else:
            next_history = combined[..., -self.history_size:]
        return outputs, next_history
