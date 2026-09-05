from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float, Float32, jaxtyped


@jaxtyped(typechecker=beartype)
@dataclass
class FWMLPState:
    """Architecture-independent memory for one batch at one fast-weight MLP layer."""

    W_fast: Float32[torch.Tensor, "B d_model d_mlp"]
    pending_r: Float32[torch.Tensor, "B S_pending d_model"] | None = None
    pending_k: Float32[torch.Tensor, "B S_pending d_mlp"] | None = None
    # S_conv is the number of retained convolution inputs (kernel_size - 1).
    teacher_conv_state: Float[torch.Tensor, "B d_mlp S_conv"] | None = None
    student_conv_state: Float[torch.Tensor, "B d_mlp S_conv"] | None = None

    @property
    def pending_count(self) -> int:
        return 0 if self.pending_r is None else self.pending_r.shape[1]

    def detach(self):
        """Return the same memory values without their training graph."""
        return FWMLPState(
            W_fast=self.W_fast.detach(),
            pending_r=None if self.pending_r is None else self.pending_r.detach(),
            pending_k=None if self.pending_k is None else self.pending_k.detach(),
            teacher_conv_state=None if self.teacher_conv_state is None else self.teacher_conv_state.detach(),
            student_conv_state=None if self.student_conv_state is None else self.student_conv_state.detach(),
        )
