import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .attention import FWQwen3Attention
from .mlp import FWQwen3MLP
from ..states.mlp_state import FWMLPState


class FWQwen3DecoderLayer(GradientCheckpointingLayer):
    """Qwen decoder with an optional teacher/student fast-weight MLP path."""

    def __init__(
        self,
        config,
        layer_idx: int,
        is_fast_weight_layer: bool = False,
        chunk_size: int = 1024,
        lr: float = 0.3,
        use_projection: bool = True,
        use_conv: bool = True,
        conv_kernel_size: int = 5,
        dynamic_beta: bool = True,
        normalize_student_features: bool = False,
        fast_weight_read_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_fast_weight_layer = is_fast_weight_layer
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = FWQwen3Attention(
            config, layer_idx,
            is_fast_weight_layer=is_fast_weight_layer,
        )
        self.mlp = FWQwen3MLP(
            config,
            is_fast_weight_layer=is_fast_weight_layer,
            chunk_size=chunk_size,
            lr=lr,
            use_projection=use_projection,
            use_conv=use_conv,
            conv_kernel_size=conv_kernel_size,
            dynamic_beta=dynamic_beta,
            normalize_student_features=normalize_student_features,
            fast_weight_read_scale=fast_weight_read_scale,
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B S d_model"],
        position_embeddings: tuple[
            Float[torch.Tensor, "#B S d_head"],
            Float[torch.Tensor, "#B S d_head"],
        ] | None = None,
        teacher_attention_mask: Float[torch.Tensor, "#B #h_q #S S_kv"] | Bool[torch.Tensor, "#B #h_q #S S_kv"] | None = None,
        student_attention_mask: Float[torch.Tensor, "#B #h_q #S S_kv"] | Bool[torch.Tensor, "#B #h_q #S S_kv"] | None = None,
        position_ids: Int[torch.Tensor, "#B S"] | None = None,
        past_key_values: Cache | None = None,
        cache_position: Int[torch.Tensor, "S"] | None = None,
        use_cache: bool = False,
        state: FWMLPState | None = None,
        output_attentions: bool = False,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> Float[torch.Tensor, "B S d_model"] | tuple[Float[torch.Tensor, "B S d_model"], FWMLPState]:
        if position_embeddings is None:
            raise ValueError("position_embeddings must be supplied by the model's RoPE module")
        if not self.is_fast_weight_layer and state is not None:
            raise ValueError("state requires is_fast_weight_layer=True")

        residual: Float[torch.Tensor, "B S d_model"] = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            teacher_attention_mask=teacher_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
            position_ids=position_ids,
            student_attention_mask=student_attention_mask,
            persistent_mask=persistent_mask,
            fast_weight_read_scale=self.mlp.fast_weight_read_scale,
        )

        if not self.is_fast_weight_layer:
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return residual + hidden_states

        if self.mlp.fast_weight_read_scale == 0:
            teacher_residual = residual + attention_output
            mlp_output, next_state = self.mlp(
                self.post_attention_layernorm(teacher_residual), state=state,
            )
            return teacher_residual + mlp_output, next_state

        teacher_attention, student_attention = attention_output
        teacher_residual: Float[torch.Tensor, "B S d_model"] = residual + teacher_attention
        student_residual: Float[torch.Tensor, "B S d_model"] = residual + student_attention
        teacher_hidden_states: Float[torch.Tensor, "B S d_model"] = self.post_attention_layernorm(teacher_residual)
        student_hidden_states: Float[torch.Tensor, "B S d_model"] = self.post_attention_layernorm(student_residual)
        mlp_output, next_state = self.mlp(
            teacher_hidden_states,
            student_hidden_states=student_hidden_states,
            state=state,
        )
        return teacher_residual + mlp_output, next_state
