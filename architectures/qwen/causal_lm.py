from dataclasses import dataclass

import torch
from torch import nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from .configuration import FWQwen3Config
from .mlp import FWQwen3MLP
from .model import FWQwen3Model
from ..states.model_state import FWModelState


@dataclass
class FWQwen3CausalLMOutput(CausalLMOutputWithPast):
    state: FWModelState | None = None


class FWQwen3ForCausalLM(Qwen3PreTrainedModel):
    """State-aware LM wrapper"""

    config_class = FWQwen3Config
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["FWQwen3DecoderLayer"]
    _supports_flash_attn = False
    _supports_flex_attn = False

    def __init__(self, config: FWQwen3Config):
        super().__init__(config)
        self.model = FWQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FWQwen3MLP) and module.is_fast_weight_layer:
            module.reset_fast_weight_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        state: FWModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        labels: Int[torch.Tensor, "B S"] | None = None,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
    ) -> FWQwen3CausalLMOutput:
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if labels is not None and logits_to_keep != 0:
            raise ValueError("labels require logits_to_keep=0")

        output = self.model(
            input_ids=input_ids, state=state, use_cache=use_cache,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
        )
        # [B, S_logits, d_model] -> [B, S_logits, vocab_size]; -0 keeps all S.
        logits: Float[torch.Tensor, "B S_logits vocab_size"] = self.lm_head(output.last_hidden_state[:, -logits_to_keep:, :])
        loss = None
        if labels is not None:
            # HF shifts labels internally: logits at t predict labels at t+1.
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)
        return FWQwen3CausalLMOutput(
            loss=loss, logits=logits, state=output.state,
            past_key_values=output.past_key_values, hidden_states=output.hidden_states,
        )
