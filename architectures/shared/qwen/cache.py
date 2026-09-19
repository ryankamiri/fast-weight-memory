import torch
from jaxtyping import Bool, Float, Int
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer


class SlidingWindowKVLayer(DynamicSlidingWindowLayer):
    """Recent K/V plus persistent entries, with aligned absolute positions."""

    def __init__(self, window_size: int, max_persistent_kv_tokens: int = 512):
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        super().__init__(sliding_window=window_size)
        if (
            type(max_persistent_kv_tokens) is not int
            or max_persistent_kv_tokens < 0
        ):
            raise ValueError(
                "max_persistent_kv_tokens must be a nonnegative integer"
            )
        self.max_persistent_kv_tokens = max_persistent_kv_tokens
        self.positions = None
        self.is_persistent = None

    def attention_metadata(
        self,
        positions: Int[torch.Tensor, "S"],
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Metadata for retained K/V plus incoming tokens, before eviction."""
        if persistent_mask is None:
            persistent_mask = torch.zeros_like(positions, dtype=torch.bool)
        if (persistent_mask.dtype != torch.bool or persistent_mask.shape != positions.shape
                or persistent_mask.device != positions.device):
            raise ValueError("persistent_mask must be boolean [S] on the input device")
        
        retained_count = 0 if self.is_persistent is None else int(self.is_persistent.sum())
        if (
            retained_count + int(persistent_mask.sum())
            > self.max_persistent_kv_tokens
        ):
            raise ValueError(
                "Persistent KV tokens exceed "
                f"max_persistent_kv_tokens={self.max_persistent_kv_tokens}"
            )
        if self.positions is None:
            return positions, persistent_mask
        return (torch.cat((self.positions, positions)),
                torch.cat((self.is_persistent, persistent_mask)))

    def update(
        self,
        key_states: Float[torch.Tensor, "B h_kv S d_head"],
        value_states: Float[torch.Tensor, "B h_kv S d_head"],
        cache_kwargs: dict | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B h_kv S_kv d_head"],
        Float[torch.Tensor, "B h_kv S_kv d_head"],
    ]:
        cache_kwargs = cache_kwargs or {}
        h_kv = key_states.shape[-2]
        positions = torch.arange(
            self.cumulative_length, self.cumulative_length + h_kv,
            device=key_states.device,
        )
        supplied_positions = cache_kwargs.get("cache_position")
        if supplied_positions is not None and not torch.equal(supplied_positions, positions):
            raise ValueError("Cache positions must follow the session's absolute token count")
        full_positions, full_persistent = self.attention_metadata(
            positions, cache_kwargs.get("persistent_mask"),
        )
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # S_kv = retained history length + S new tokens.
        full_keys: Float[torch.Tensor, "B h_kv S_kv d_head"] = torch.cat((self.keys, key_states), dim=-2)
        full_values: Float[torch.Tensor, "B h_kv S_kv d_head"] = torch.cat((self.values, value_states), dim=-2)
        self.cumulative_length += key_states.shape[-2]

        # Persistent entries are additional to the absolute-position window.
        keep = full_persistent | (full_positions >= self.cumulative_length - (self.sliding_window - 1))
        self.keys = full_keys[..., keep, :].contiguous()
        self.values = full_values[..., keep, :].contiguous()
        self.positions = full_positions[keep]
        self.is_persistent = full_persistent[keep]

        # Do not return just the retained suffix: earlier queries in a prefill
        # block need older keys than its final query does.
        return full_keys, full_values

    def get_mask_sizes(self, cache_position):
        if self.is_persistent is not None and self.is_persistent.any():
            raise ValueError("Persistent caches require explicit attention_metadata, not a contiguous offset")
        return super().get_mask_sizes(cache_position)

    def get_max_cache_shape(self):
        persistent_count = 0 if self.is_persistent is None else int(self.is_persistent.sum())
        return self.sliding_window + persistent_count

    def crop(self, max_length: int) -> None:
        # The parent rejects rollback after eviction, when lost K/V cannot be
        # reconstructed. Before eviction, crop metadata alongside its tensors.
        super().crop(max_length)
        if self.positions is not None:
            self.positions = self.positions[:self.cumulative_length]
            self.is_persistent = self.is_persistent[:self.cumulative_length]

    def reset(self) -> None:
        """Clear retained storage and positions without modifying old tensors."""
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.cumulative_length = 0
        self.positions = None
        self.is_persistent = None


class SlidingWindowKVCache(Cache):
    """One bounded working-memory cache per layer, plus persistent K/V."""

    def __init__(
        self,
        num_layers: int,
        window_size: int,
        max_persistent_kv_tokens: int = 512,
    ):
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        super().__init__(
            layers=[
                SlidingWindowKVLayer(window_size, max_persistent_kv_tokens)
                for _ in range(num_layers)
            ]
        )
        self.window_size = window_size
        self.max_persistent_kv_tokens = max_persistent_kv_tokens
        # Persistence is a cache policy keyed by absolute sequence position.
        # Attention modules only need to supply their ordinary cache_position.
        self.persistent_positions: Int[torch.Tensor, "P"] | None = None

    def register_persistent(
        self,
        positions: Int[torch.Tensor, "S"],
        persistent_mask: Bool[torch.Tensor, "S"] | None,
    ) -> None:
        """Idempotently mark absolute positions that must survive eviction."""
        if persistent_mask is None:
            return
        if (
            persistent_mask.dtype != torch.bool
            or persistent_mask.shape != positions.shape
            or persistent_mask.device != positions.device
        ):
            raise ValueError(
                "persistent_mask must be boolean [S] on the input device"
            )
        additions = positions[persistent_mask]
        (num_additions,) = additions.shape
        if num_additions == 0:
            return
        if self.persistent_positions is None:
            already_registered = torch.zeros_like(additions, dtype=torch.bool)
        else:
            if self.persistent_positions.device != positions.device:
                raise ValueError("Persistent positions must remain on one device")
            already_registered = (
                additions[:, None] == self.persistent_positions[None, :]
            ).any(dim=1)
        tokens_seen = self.layers[0].cumulative_length
        if ((additions < tokens_seen) & ~already_registered).any():
            raise ValueError(
                "Cannot retroactively persist positions whose K/V were already "
                "processed"
            )
        if self.persistent_positions is None:
            combined = torch.unique(additions, sorted=True)
        else:
            combined = torch.unique(
                torch.cat((self.persistent_positions, additions)),
                sorted=True,
            )
        (num_persistent_positions,) = combined.shape
        if num_persistent_positions > self.max_persistent_kv_tokens:
            raise ValueError(
                "Persistent KV tokens exceed "
                f"max_persistent_kv_tokens={self.max_persistent_kv_tokens}"
            )
        self.persistent_positions = combined

    def persistent_mask(
        self,
        positions: Int[torch.Tensor, "S"],
    ) -> Bool[torch.Tensor, "S"]:
        """Return which absolute positions are registered as persistent."""
        if self.persistent_positions is None:
            return torch.zeros_like(positions, dtype=torch.bool)
        if self.persistent_positions.device != positions.device:
            raise ValueError("Persistent positions must remain on one device")
        return (
            positions[:, None] == self.persistent_positions[None, :]
        ).any(dim=1)

    def attention_metadata(
        self,
        positions: Int[torch.Tensor, "S"],
    ) -> tuple[Int[torch.Tensor, "S_kv"], Bool[torch.Tensor, "S_kv"]]:
        """Metadata for retained K/V plus incoming registered positions."""
        return self.layers[0].attention_metadata(
            positions,
            self.persistent_mask(positions),
        )

    def update(
        self,
        key_states: Float[torch.Tensor, "B h_kv S d_head"],
        value_states: Float[torch.Tensor, "B h_kv S d_head"],
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B h_kv S_kv d_head"],
        Float[torch.Tensor, "B h_kv S_kv d_head"],
    ]:
        cache_kwargs = dict(cache_kwargs or {})
        if "persistent_mask" in cache_kwargs:
            raise ValueError(
                "Register persistent positions on SlidingWindowKVCache instead "
                "of passing persistent_mask through attention"
            )
        positions = cache_kwargs.get("cache_position")
        if positions is None:
            S = key_states.shape[-2]
            start = self.layers[layer_idx].cumulative_length
            positions = torch.arange(
                start,
                start + S,
                device=key_states.device,
            )
        cache_kwargs["persistent_mask"] = self.persistent_mask(positions)
        return super().update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self.persistent_positions = None

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        if self.persistent_positions is not None:
            length = self.layers[0].cumulative_length
            self.persistent_positions = self.persistent_positions[
                self.persistent_positions < length
            ]
