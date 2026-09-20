from dataclasses import dataclass
import math

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from torch.utils.checkpoint import checkpoint

from architectures.shared.qwen.cache import SlidingWindowKVCache
from architectures.shared.qwen.masking import prepare_sliding_attention_mask
from architectures.shared.qwen.model import StatefulQwen3Model
from architectures.titans.neural_memory import NeuralMemory

from .configuration import TaalQwen3Config
from .decoder import TaalQwen3DecoderLayer
from .state import TaalModelState


logger = logging.get_logger(__name__)


@dataclass
class TaalQwen3ModelOutput(BaseModelOutputWithPast):
    state: TaalModelState | None = None


class TaalQwen3Model(StatefulQwen3Model):
    """Qwen3 backbone with a stateful TaaL adapter before every decoder."""

    config_class = TaalQwen3Config
    _no_split_modules = ["TaalQwen3DecoderLayer"]
    # The architecture is compatible, but the current explicit additive-mask
    # path has only been implemented and verified for eager attention and SDPA.
    _supports_flash_attn = False
    _supports_flex_attn = False
    gradient_checkpointing = False

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NeuralMemory):
            # Qwen initializes every child Linear during post_init; restore the
            # configured Titans control biases after that traversal.
            module.reset_update_controls()

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func=checkpoint,
    ):
        # Checkpoint only the ordinary Qwen body. NeuralMemory uses
        # torch.func.grad, which cannot execute inside saved-tensor hooks.
        self.gradient_checkpointing = enable
        self._gradient_checkpointing_func = gradient_checkpointing_func

    def _build_decoder_layer(
        self,
        config: TaalQwen3Config,
        layer_idx: int,
    ) -> TaalQwen3DecoderLayer:
        return TaalQwen3DecoderLayer(
            config,
            layer_idx,
            taal_config=config.taal_layer_config(),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        state: TaalModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        output_hidden_states: bool = False,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool | None = None,
    ) -> TaalQwen3ModelOutput:
        past_key_values = state.past_key_values if state is not None else None
        if self.training and self.is_gradient_checkpointing:
            if past_key_values is not None:
                raise ValueError(
                    "KV caches cannot be supplied during checkpointed training; "
                    "use memory-only state"
                )
            if use_cache:
                logger.warning_once(
                    "KV caching is disabled during checkpointed training; "
                    "setting use_cache=False."
                )
            use_cache = False

        if self.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError(
                "TaalQwen3Model currently supports only eager or sdpa attention"
            )

        hidden_states: Float[torch.Tensor, "B S D_model"] = self.embed_tokens(
            input_ids
        )
        B, S, _ = hidden_states.shape
        if S == 0:
            raise ValueError("At least one input token is required")
        if write_mask is not None and write_mask.device != hidden_states.device:
            raise ValueError("write_mask must be on the input device")
        if persistent_mask is not None and persistent_mask.device != hidden_states.device:
            raise ValueError("persistent_mask must be on the input device")
        if (
            type(memory_read_scale) not in (int, float)
            or not math.isfinite(memory_read_scale)
            or memory_read_scale < 0
        ):
            raise ValueError("memory_read_scale must be finite and nonnegative")
        if prepend_memory_tokens is None:
            # A fresh state begins a segment. Cached execution blocks and decode
            # calls continue it unless the caller explicitly starts a new turn.
            prepend_memory_tokens = state is None
        elif type(prepend_memory_tokens) is not bool:
            raise ValueError("prepend_memory_tokens must be boolean")

        if past_key_values is not None:
            if not isinstance(past_key_values, SlidingWindowKVCache):
                raise ValueError("State must use SlidingWindowKVCache")
            if (
                past_key_values.window_size != self.config.working_memory_size
                or past_key_values.max_persistent_kv_tokens
                != self.config.max_persistent_kv_tokens
                or len(past_key_values.layers) != len(self.layers)
            ):
                raise ValueError(
                    "KV cache window, persistent-token limit, and layer count "
                    "must match the model"
                )
            if not use_cache:
                raise ValueError("A supplied KV cache requires use_cache=True")

        tokens_seen = state.tokens_seen if state is not None else 0
        memory_states = {} if state is None else state.memory_states
        expected_layers = set(range(len(self.layers)))
        if past_key_values is not None and any(
            layer.get_seq_length() != tokens_seen
            for layer in past_key_values.layers
        ):
            raise ValueError(
                "State tokens_seen and KV token counts disagree; old states are "
                "not snapshots"
            )
        if tokens_seen and set(memory_states) != expected_layers:
            raise ValueError(
                "Continuing a TaaL session requires every layer memory state"
            )
        if set(memory_states) - expected_layers:
            raise ValueError("Memory state contains an unknown decoder layer")

        cache_position: Int[torch.Tensor, "S"] = torch.arange(
            tokens_seen,
            tokens_seen + S,
            device=hidden_states.device,
        )
        position_ids: Int[torch.Tensor, "1 S"] = cache_position.unsqueeze(0)

        if use_cache and past_key_values is None:
            if tokens_seen:
                raise ValueError(
                    "Cannot reconstruct earlier KV history from memory-only state; "
                    "start a fresh cached session"
                )
            past_key_values = SlidingWindowKVCache(
                len(self.layers),
                self.config.working_memory_size,
                max_persistent_kv_tokens=self.config.max_persistent_kv_tokens,
            )
        if past_key_values is None:
            key_positions = cache_position
            key_persistent = persistent_mask
            if (
                persistent_mask is not None
                and int(persistent_mask.sum())
                > self.config.max_persistent_kv_tokens
            ):
                raise ValueError(
                    "Persistent KV tokens exceed "
                    "max_persistent_kv_tokens="
                    f"{self.config.max_persistent_kv_tokens}"
                )
        else:
            past_key_values.register_persistent(
                cache_position,
                persistent_mask,
            )
            key_positions, key_persistent = past_key_values.attention_metadata(
                cache_position
            )

        prepared_attention_mask = prepare_sliding_attention_mask(
            attention_mask,
            hidden_states,
            cache_position,
            key_positions,
            self.config.working_memory_size,
            key_persistent,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        next_memory_states = {}
        all_hidden_states = () if output_hidden_states else None
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states, next_memory_states[layer_idx] = (
                    decoder_layer.forward_memory(
                        hidden_states,
                        memory_state=memory_states.get(layer_idx),
                        write_mask=write_mask,
                        memory_read_scale=memory_read_scale,
                        prepend_memory_tokens=prepend_memory_tokens,
                    )
                )
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.forward_decoder,
                    hidden_states,
                    attention_mask=prepared_attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states, next_memory_states[layer_idx] = decoder_layer(
                    hidden_states,
                    attention_mask=prepared_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    memory_state=memory_states.get(layer_idx),
                    write_mask=write_mask,
                    memory_read_scale=memory_read_scale,
                    prepend_memory_tokens=prepend_memory_tokens,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_state = TaalModelState(
            past_key_values=past_key_values,
            tokens_seen=tokens_seen + S,
            memory_states=next_memory_states,
        )
        return TaalQwen3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            state=next_state,
        )
