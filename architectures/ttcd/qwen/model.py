from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from architectures.shared.qwen.cache import SlidingWindowKVCache
from architectures.shared.qwen.masking import prepare_sliding_attention_mask
from architectures.shared.qwen.model import StatefulQwen3Model

from .configuration import TTCDQwen3Config
from .decoder import TTCDQwen3DecoderLayer
from .mlp import TTCDQwen3MLP
from ..states.model_state import TTCDModelState

logger = logging.get_logger(__name__)


@dataclass
class TTCDQwen3ModelOutput(BaseModelOutputWithPast):
    state: TTCDModelState | None = None


class TTCDQwen3Model(StatefulQwen3Model):

    config_class = TTCDQwen3Config
    _no_split_modules = ["TTCDQwen3DecoderLayer"]
    _supports_flash_attn = False
    _supports_flex_attn = False

    def _build_decoder_layer(
        self,
        config: TTCDQwen3Config,
        layer_idx: int,
    ) -> TTCDQwen3DecoderLayer:
        return TTCDQwen3DecoderLayer(
            config,
            layer_idx,
            is_fast_weight_layer=layer_idx in config.fast_weight_layers,
            chunk_size=config.chunk_size,
            lr=config.lr,
            use_projection=config.use_projection,
            use_conv=config.use_conv,
            conv_kernel_size=config.conv_kernel_size,
            dynamic_beta=config.dynamic_beta,
            normalize_student_features=config.normalize_student_features,
            fast_weight_read_scale=config.fast_weight_read_scale,
        )

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, TTCDQwen3MLP) and module.is_fast_weight_layer:
            module.reset_fast_weight_parameters()

    def _prepare_masks(
        self,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None,
        hidden_states: Float[torch.Tensor, "B S d_model"],
        query_positions: Int[torch.Tensor, "S"],
        key_positions: Int[torch.Tensor, "S_kv"],
        key_persistent: Bool[torch.Tensor, "S_kv"] | None = None,
    ) -> dict[str, Float[torch.Tensor, "#B 1 S S_kv"]]:
        if attention_mask is not None:
            if self.config.fast_weight_layers and not attention_mask.all():
                raise ValueError("Padded fast-weight batches need per-example chunk state; not supported yet")

        # every layer's main attention uses the teacher window,
        # regardless of native layer_types or the input sequence length.
        masks = {
            "teacher": prepare_sliding_attention_mask(
                attention_mask,
                hidden_states,
                query_positions,
                key_positions,
                self.config.teacher_window_size,
                key_persistent,
            )
        }
        if self.config.fast_weight_layers:
            masks["student"] = prepare_sliding_attention_mask(
                attention_mask,
                hidden_states,
                query_positions,
                key_positions,
                self.config.student_window_size,
                key_persistent,
            )
        return masks

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        state: TTCDModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        output_hidden_states: bool = False,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> TTCDQwen3ModelOutput:
        # KV cache and MLP memory always enter together through session state.
        past_key_values = state.past_key_values if state is not None else None
        if self.training and self.is_gradient_checkpointing:
            if past_key_values is not None:
                raise ValueError("KV caches cannot be supplied during checkpointed training; use MLP-only state")
            if use_cache:
                logger.warning_once("KV caching is disabled during checkpointed training; setting use_cache=False.")
            use_cache = False
        
        if self.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("TTCDQwen3Model currently supports only eager or sdpa attention")
        
        # Embed
        hidden_states: Float[torch.Tensor, "B S d_model"] = self.embed_tokens(input_ids)
        B, S, d_model = hidden_states.shape
        if S == 0:
            raise ValueError("At least one input token is required")
        if persistent_mask is not None and persistent_mask.device != hidden_states.device:
            raise ValueError("persistent_mask must be on the input device")

        if past_key_values is not None:
            if not isinstance(past_key_values, SlidingWindowKVCache):
                raise ValueError("State must use SlidingWindowKVCache")
            if (past_key_values.window_size != self.config.teacher_window_size
                    or past_key_values.max_persistent_tokens != self.config.max_persistent_tokens
                    or len(past_key_values.layers) != len(self.layers)):
                raise ValueError("KV cache window, persistent-token limit, and layer count must match the model")
            if not use_cache:
                raise ValueError("A supplied KV cache requires use_cache=True")
        
        tokens_seen = state.tokens_seen if state is not None else 0
        mlp_states = {} if state is None else state.mlp_states

        if past_key_values is not None and any(
            layer.get_seq_length() != tokens_seen for layer in past_key_values.layers
        ):
            raise ValueError("State tokens_seen and KV token counts disagree; old states are not snapshots")
        if tokens_seen and self.config.fast_weight_layers and set(mlp_states) != set(self.config.fast_weight_layers):
            raise ValueError("Continuing a fast-weight session requires all its layer MLP states")
        if set(mlp_states) - set(self.config.fast_weight_layers):
            raise ValueError("MLP state contains layers not configured as fast-weight layers")
        for layer_state in mlp_states.values():
            if layer_state.W_fast.shape != (B, d_model, self.config.intermediate_size):
                raise ValueError("MLP state batch/model dimensions must match the current input")
            if layer_state.W_fast.device != hidden_states.device:
                raise ValueError("MLP state must be on the input device")

        cache_position: Int[torch.Tensor, "S"] = torch.arange(tokens_seen, tokens_seen + S, device=hidden_states.device)
        position_ids: Int[torch.Tensor, "1 S"] = cache_position.unsqueeze(0)
        
        if use_cache and past_key_values is None:
            if tokens_seen:
                raise ValueError("Cannot reconstruct earlier KV history from MLP state; start a fresh cached session")
            past_key_values = SlidingWindowKVCache(
                len(self.layers), self.config.teacher_window_size, self.config.max_persistent_tokens,
            )
        if past_key_values is None:
            # MLP-only continuation has no earlier attention keys.
            key_positions = cache_position
            key_persistent = persistent_mask
            if persistent_mask is not None and int(persistent_mask.sum()) > self.config.max_persistent_tokens:
                raise ValueError(f"Persistent tokens exceed max_persistent_tokens={self.config.max_persistent_tokens}")
        else:
            # Retained positions can have gaps. Use cache-owned metadata instead
            # of reconstructing a contiguous range from the stored tensor length.
            key_positions, key_persistent = past_key_values.layers[0].attention_metadata(
                cache_position, persistent_mask,
            )
        masks = self._prepare_masks(
            attention_mask, hidden_states, cache_position, key_positions, key_persistent,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        next_mlp_states = {}
        all_hidden_states = () if output_hidden_states else None
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            is_fast = decoder_layer.is_fast_weight_layer
            layer_output = decoder_layer(
                hidden_states,
                teacher_attention_mask=masks["teacher"],
                student_attention_mask=masks["student"] if is_fast else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                state=mlp_states.get(layer_idx),
                persistent_mask=persistent_mask,
            )
            if is_fast:
                hidden_states, next_mlp_states[layer_idx] = layer_output
            else:
                hidden_states = layer_output
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_state = TTCDModelState(
            past_key_values=past_key_values,
            mlp_states=next_mlp_states,
            tokens_seen=tokens_seen + S,
        )
        return TTCDQwen3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            state=next_state,
        )
