import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
)


class FWQwen3Attention(Qwen3Attention):
    """Qwen attention with optional teacher/student outputs."""

    def __init__(
        self,
        config, 
        layer_idx: int,
        is_fast_weight_layer: bool = False,
    ):
        super().__init__(config, layer_idx)
        self.is_fast_weight_layer = is_fast_weight_layer

    @jaxtyped(typechecker=beartype)
    def forward(
        self, 
        hidden_states: Float[torch.Tensor, "B S d_model"],
        position_embeddings: tuple[
            Float[torch.Tensor, "#B S d_head"],
            Float[torch.Tensor, "#B S d_head"],
        ],
        teacher_attention_mask: Float[torch.Tensor, "#B #h_q #S S_kv"] | Bool[torch.Tensor, "#B #h_q #S S_kv"] | None = None,
        student_attention_mask: Float[torch.Tensor, "#B #h_q #S S_kv"] | Bool[torch.Tensor, "#B #h_q #S S_kv"] | None = None,
        position_ids: Int[torch.Tensor, "#B S"] | None = None,
        past_key_values: Cache | None = None,
        cache_position: Int[torch.Tensor, "S"] | None = None,
        output_attentions: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "B S d_model"] | tuple[
            Float[torch.Tensor, "B S d_model"],
            Float[torch.Tensor, "B S d_model"],
        ],
        Float[torch.Tensor, "B h_q S S_kv"] | None,
    ]:
        # normal forward pass
        if not self.is_fast_weight_layer:
            return super().forward(
                hidden_states, position_embeddings, teacher_attention_mask,
                past_key_values=past_key_values, cache_position=cache_position,
                output_attentions=output_attentions, position_ids=position_ids,
            )
        if output_attentions:
            raise ValueError("Dual-window SDPA does not support output_attentions=True")
        if teacher_attention_mask is None or student_attention_mask is None:
            raise ValueError("Dual-window mode requires prepared teacher and student attention masks")

        B, S, d_model = hidden_states.shape
        h_q = self.config.num_attention_heads
        h_kv = self.config.num_key_value_heads
        d_head = self.head_dim

        query: Float[torch.Tensor, "B h_q S d_head"] = self.q_norm(self.q_proj(hidden_states).view(B, S, h_q, d_head)).transpose(1, 2)
        key: Float[torch.Tensor, "B h_kv S d_head"] = self.k_norm(self.k_proj(hidden_states).view(B, S, h_kv, d_head)).transpose(1, 2)
        value: Float[torch.Tensor, "B h_kv S d_head"] = self.v_proj(hidden_states).view(B, S, h_kv, d_head).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            # Update KV Cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs,
            )
        # Local annotations alone do not run checks. These also bind the KV
        # sequence length after caching, which can differ from the query's S.
        assert isinstance(query, Float[torch.Tensor, "B h_q S d_head"])
        assert isinstance(key, Float[torch.Tensor, "B h_kv S_kv d_head"])
        assert isinstance(value, Float[torch.Tensor, "B h_kv S_kv d_head"])
        teacher_output, _ = sdpa_attention_forward(
            self, 
            query, 
            key, 
            value, 
            teacher_attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, 
            is_causal=False,
        )
        student_output, _ = sdpa_attention_forward(
            self, 
            query, 
            key, 
            value, 
            student_attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, 
            is_causal=False,
        )
        
        teacher_output = self.o_proj(teacher_output.reshape(B, S, h_q * d_head).contiguous())
        student_output = self.o_proj(student_output.reshape(B, S, h_q * d_head).contiguous())
        return (teacher_output, student_output), None
