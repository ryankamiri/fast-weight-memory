"""Parameter-matched, stateless control for the recurrent TaaL branch."""

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F


class StaticResidualAdapter(nn.Module):
    def __init__(self, model_dim: int, width: int):
        super().__init__()
        self.projection_in = nn.Linear(model_dim, width, bias=False)
        self.projection_out = nn.Linear(width, model_dim, bias=False)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Float[torch.Tensor, "B S D_model"]) -> Float[torch.Tensor, "B S D_model"]:
        values: Float[torch.Tensor, "B S D_adapter"] = F.silu(self.projection_in(inputs))
        correction: Float[torch.Tensor, "B S D_model"] = self.projection_out(values)
        return inputs + self.residual_gate.tanh() * correction
