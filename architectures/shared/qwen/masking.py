import torch
from jaxtyping import Bool, Float, Int


def prepare_sliding_attention_mask(
    attention_mask: Bool[torch.Tensor, "B S_kv"] | None,
    hidden_states: Float[torch.Tensor, "B S D_model"],
    query_positions: Int[torch.Tensor, "S"],
    key_positions: Int[torch.Tensor, "S_kv"],
    window_size: int,
    key_persistent: Bool[torch.Tensor, "S_kv"] | None = None,
) -> Float[torch.Tensor, "#B 1 S S_kv"]:
    """Build an additive causal mask over retained and incoming K/V entries."""
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    B, S, _ = hidden_states.shape
    S_kv = key_positions.shape[0]
    if attention_mask is not None:
        if attention_mask.device != hidden_states.device:
            raise ValueError("attention_mask must be on the input device")
        if attention_mask.dtype != torch.bool or attention_mask.shape != (B, S_kv):
            raise ValueError(
                "attention_mask must be boolean [B, S_kv], covering retained "
                "keys plus new tokens"
            )
    if key_persistent is not None:
        if (
            key_persistent.device != hidden_states.device
            or key_persistent.dtype != torch.bool
            or key_persistent.shape != (S_kv,)
        ):
            raise ValueError("key_persistent must be boolean [S_kv] on the input device")

    causal: Bool[torch.Tensor, "1 1 S S_kv"] = (
        query_positions[:, None] >= key_positions[None, :]
    )[None, None]
    visible: Bool[torch.Tensor, "S S_kv"] = (
        key_positions[None, :] > query_positions[:, None] - window_size
    )
    if key_persistent is not None:
        visible = visible | key_persistent[None, :]
    allowed: Bool[torch.Tensor, "#B 1 S S_kv"] = causal & visible[None, None]
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :]

    mask: Float[torch.Tensor, "#B 1 S S_kv"] = torch.zeros(
        allowed.shape,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)
