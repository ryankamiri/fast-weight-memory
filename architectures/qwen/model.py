from dataclasses import dataclass

import torch
from torch import nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from .configuration import FWQwen3Config
from .decoder import FWQwen3DecoderLayer
from .mlp import FWQwen3MLP
from ..cache.sliding_window import SlidingWindowKVCache
from ..states.model_state import FWModelState

logger = logging.get_logger(__name__)


@dataclass
class FWQwen3ModelOutput(BaseModelOutputWithPast):
    state: FWModelState | None = None


class FWQwen3Model(Qwen3PreTrainedModel):

    config_class = FWQwen3Config
    _no_split_modules = ["FWQwen3DecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = False
    _supports_flex_attn = False

    def __init__(self, config: FWQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            FWQwen3DecoderLayer(
                config, layer_idx,
                is_fast_weight_layer=layer_idx in config.fast_weight_layers,
                chunk_size=config.chunk_size,
                lr=config.lr,
                use_projection=config.use_projection,
                use_conv=config.use_conv,
                conv_kernel_size=config.conv_kernel_size,
                dynamic_beta=config.dynamic_beta,
                normalize_student_features=config.normalize_student_features,
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Use non-reentrant checkpointing to preserve gradients through FWMLPState."""
        checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {})
        if checkpoint_kwargs.get("use_reentrant", False) is not False:
            raise ValueError("FWQwen3Model requires use_reentrant=False for MLP state gradients")
        checkpoint_kwargs["use_reentrant"] = False
        super().gradient_checkpointing_enable(checkpoint_kwargs)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FWQwen3MLP) and module.is_fast_weight_layer:
            module.reset_fast_weight_parameters()

    def _prepare_masks(
        self,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None,
        hidden_states: Float[torch.Tensor, "B S d_model"],
        query_positions: Int[torch.Tensor, "S"],
        key_positions: Int[torch.Tensor, "S_kv"],
    ) -> dict[str, Float[torch.Tensor, "#B 1 S S_kv"]]:
        B, S, d_model = hidden_states.shape
        S_kv = key_positions.shape[0]
        # Each query gets its own diagonal window, not one global suffix slice.
        # 1, 1 are B, H_q
        causal: Bool[torch.Tensor, "1 1 S S_kv"] = (query_positions[:, None] >= key_positions[None, :])[None, None]
        if attention_mask is not None:
            if attention_mask.device != hidden_states.device:
                raise ValueError("attention_mask must be on the input device")
            if attention_mask.dtype != torch.bool or attention_mask.shape != (B, S_kv):
                raise ValueError("attention_mask must be boolean [B, S_kv], covering retained keys plus new tokens")
            if self.config.fast_weight_layers and not attention_mask.all():
                raise ValueError("Padded fast-weight batches need per-example chunk state; not supported yet")

        def make_mask(window_size: int) -> Float[torch.Tensor, "#B 1 S S_kv"]:
            left_edge = query_positions - window_size
            # Batch dimension is 1 initially, or B after applying the caller mask.
            allowed: Bool[torch.Tensor, "#B 1 S S_kv"] = causal & (key_positions[None, :] > left_edge[:, None])[None, None]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :]
            mask: Float[torch.Tensor, "#B 1 S S_kv"] = torch.zeros(allowed.shape, device=hidden_states.device, dtype=hidden_states.dtype)
            return mask.masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)

        # every layer's main attention uses the teacher window,
        # regardless of native layer_types or the input sequence length.
        masks = {"teacher": make_mask(self.config.teacher_window_size)}
        if self.config.fast_weight_layers:
            masks["student"] = make_mask(self.config.student_window_size)
        return masks

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        state: FWModelState | None = None,
        use_cache: bool = False,
        # True=valid token; columns cover retained KV history plus this call's tokens.
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        output_hidden_states: bool = False,
    ) -> FWQwen3ModelOutput:
        # KV cache and MLP memory always enter together through session state.
        past_key_values = state.past_key_values if state is not None else None
        if self.training and self.is_gradient_checkpointing:
            if past_key_values is not None:
                raise ValueError("KV caches cannot be supplied during checkpointed training; use MLP-only state")
            if use_cache:
                logger.warning_once("KV caching is disabled during checkpointed training; setting use_cache=False.")
            use_cache = False
        
        if self.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("FWQwen3Model currently supports only eager or sdpa attention")
        
        # Embed
        hidden_states: Float[torch.Tensor, "B S d_model"] = self.embed_tokens(input_ids)
        B, S, d_model = hidden_states.shape
        if S == 0:
            raise ValueError("At least one input token is required")

        if past_key_values is not None:
            if not isinstance(past_key_values, SlidingWindowKVCache):
                raise ValueError("State must use SlidingWindowKVCache")
            if (past_key_values.window_size != self.config.teacher_window_size
                    or len(past_key_values.layers) != len(self.layers)):
                raise ValueError("KV cache window and layer count must match the model")
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
        
        if past_key_values is None:
            # MLP-only continuation has no earlier attention keys.
            key_positions = cache_position
        else:
            # All layers retain the same teacher-sized history. After eviction,
            # key_offset = tokens_seen - retained_length, not zero.
            S_kv, key_offset = past_key_values.get_mask_sizes(cache_position, layer_idx=0)
            key_positions = torch.arange(key_offset, key_offset + S_kv, device=hidden_states.device)
        masks = self._prepare_masks(attention_mask, hidden_states, cache_position, key_positions)
        if use_cache and past_key_values is None:
            if tokens_seen:
                raise ValueError("Cannot reconstruct earlier KV history from MLP state; start a fresh cached session")
            past_key_values = SlidingWindowKVCache(len(self.layers), self.config.teacher_window_size)
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
            )
            if is_fast:
                hidden_states, next_mlp_states[layer_idx] = layer_output
            else:
                hidden_states = layer_output
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_state = FWModelState(
            past_key_values=past_key_values,
            mlp_states=next_mlp_states,
            tokens_seen=tokens_seen + S,
        )
        return FWQwen3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            state=next_state,
        )
