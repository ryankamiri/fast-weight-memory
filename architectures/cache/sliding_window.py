import torch
from jaxtyping import Float
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer


class SlidingWindowKVLayer(DynamicSlidingWindowLayer):
    """HF sliding-cache semantics with compact storage and window-size-1 support."""

    def __init__(self, window_size: int):
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        super().__init__(sliding_window=window_size)

    def update(
        self,
        key_states: Float[torch.Tensor, "B h_kv S d_head"],
        value_states: Float[torch.Tensor, "B h_kv S d_head"],
        cache_kwargs: dict | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B h_kv S_kv d_head"],
        Float[torch.Tensor, "B h_kv S_kv d_head"],
    ]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # S_kv = retained history length + S new tokens.
        full_keys: Float[torch.Tensor, "B h_kv S_kv d_head"] = torch.cat((self.keys, key_states), dim=-2)
        full_values: Float[torch.Tensor, "B h_kv S_kv d_head"] = torch.cat((self.values, value_states), dim=-2)
        self.cumulative_length += key_states.shape[-2]

        retained_length = self.sliding_window - 1
        # A zero-length suffix needs an explicit empty slice: -0 means 0.
        retained_keys = full_keys[..., -retained_length:, :] if retained_length else full_keys[..., :0, :]
        retained_values = full_values[..., -retained_length:, :] if retained_length else full_values[..., :0, :]
        self.keys = retained_keys.clone(memory_format=torch.contiguous_format)
        self.values = retained_values.clone(memory_format=torch.contiguous_format)

        # Do not return just the retained suffix: earlier queries in a prefill
        # block need older keys than its final query does.
        return full_keys, full_values

    def reset(self) -> None:
        """Clear retained storage and positions without modifying old tensors."""
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.cumulative_length = 0


class SlidingWindowKVCache(Cache):
    """One teacher-sized sliding KV cache per decoder layer."""

    def __init__(self, num_layers: int, window_size: int):
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        super().__init__(layers=[SlidingWindowKVLayer(window_size) for _ in range(num_layers)])
        self.window_size = window_size
