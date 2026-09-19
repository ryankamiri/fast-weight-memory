import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ModuleNotFoundError as error:
    if error.name != "liger_kernel":
        raise
    LigerFusedLinearCrossEntropyLoss = None


def causal_lm_loss(
    lm_head: nn.Linear,
    hidden_states: Float[torch.Tensor, "N D_model"],
    labels: Int[torch.Tensor, "B S"],
) -> Float[torch.Tensor, ""]:
    """Compute next-token loss without materializing full CUDA logits."""
    # N = B * (S - 1): every next-token prediction flattened across batch and sequence.
    targets: Int[torch.Tensor, "N"] = labels[:, 1:].reshape(-1).to(
        hidden_states.device
    )
    if hidden_states.is_cuda:
        if LigerFusedLinearCrossEntropyLoss is None:
            raise ImportError("CUDA loss requires Liger.")
        return LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)(
            lm_head.weight,
            hidden_states,
            targets,
        )
    return F.cross_entropy(lm_head(hidden_states).float(), targets)
