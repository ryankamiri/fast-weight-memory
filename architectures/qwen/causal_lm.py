from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ModuleNotFoundError as error:
    if error.name != "liger_kernel":
        raise
    LigerFusedLinearCrossEntropyLoss = None

from .configuration import FWQwen3Config
from .mlp import FWQwen3MLP
from .model import FWQwen3Model
from ..states.model_state import FWModelState
from inference.generation import GenerationOutput, sample_token
from inference.prefill import prefill


@dataclass
class FWQwen3CausalLMOutput(CausalLMOutputWithPast):
    state: FWModelState | None = None


class FWQwen3ForCausalLM(Qwen3PreTrainedModel):
    """State-aware LM wrapper; labeled forwards return loss without logits."""

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

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def prefill(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        execution_block_size: int | None = None,
        state: FWModelState | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> FWQwen3CausalLMOutput:
        """Prefill the backbone, then project only the final token to logits."""
        output = prefill(
            self.model, input_ids, execution_block_size=execution_block_size,
            state=state, persistent_mask=persistent_mask,
        )
        return FWQwen3CausalLMOutput(
            logits=self.lm_head(output.last_hidden_state[:, -1:, :]),
            state=output.state, past_key_values=output.past_key_values,
        )

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        input_ids: Int[torch.Tensor, "1 S"],
        state: FWModelState | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int | list[int] | None = None,
        execution_block_size: int | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationOutput:
        """Prefill new input, then decode one unpadded sequence with session state."""
        if self.training:
            raise ValueError("generate requires evaluation mode; call model.eval() first")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if do_sample:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if type(top_k) is not int or top_k < 0:
                raise ValueError("top_k must be a nonnegative integer; 0 disables filtering")
            if generator is not None and torch.device(generator.device) != input_ids.device:
                raise ValueError("generator must be on the input device")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_ids = [] if eos_token_id is None else eos_token_id
        if type(eos_ids) is int:
            eos_ids = [eos_ids]
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in eos_ids):
            raise ValueError("EOS token IDs must be valid vocabulary IDs")

        output = self.prefill(
            input_ids, execution_block_size=execution_block_size,
            state=state, persistent_mask=persistent_mask,
        )
        generated = []
        stop_reason = "max_new_tokens"
        for _ in range(max_new_tokens):
            next_token = sample_token(
                output.logits[:, -1, :], do_sample, temperature, top_k, top_p, generator,
            )
            generated.append(next_token)
            output = self(next_token, state=output.state, use_cache=True, logits_to_keep=1)
            if next_token.item() in eos_ids:
                stop_reason = "eos"
                break
        return GenerationOutput(
            token_ids=torch.cat(generated, dim=1), state=output.state,
            stop_reason=stop_reason,
        )

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
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> FWQwen3CausalLMOutput:
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if labels is not None and logits_to_keep != 0:
            raise ValueError("labels require logits_to_keep=0")

        output = self.model(
            input_ids=input_ids, state=state, use_cache=use_cache,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
            persistent_mask=persistent_mask,
        )
        logits = None
        loss = None
        if labels is not None:
            # Hidden state at t predicts label at t+1; N = B * (S - 1).
            hidden_states: Float[torch.Tensor, "N d_model"] = output.last_hidden_state[:, :-1].reshape(-1, self.config.hidden_size)
            targets: Int[torch.Tensor, "N"] = labels[:, 1:].reshape(-1).to(hidden_states.device)
            if hidden_states.is_cuda:
                if LigerFusedLinearCrossEntropyLoss is None:
                    raise ImportError("CUDA loss requires Liger.")
                # Fuse projection + loss so full [B, S, vocab_size] logits never exist.
                loss = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)(
                    self.lm_head.weight, hidden_states, targets,
                )
            else:
                # Small local CPU/MPS tests; Liger's kernels require a supported accelerator.
                loss = F.cross_entropy(self.lm_head(hidden_states).float(), targets)
        else:
            # [B, S_logits, d_model] -> [B, S_logits, vocab_size]; -0 keeps all S.
            logits: Float[torch.Tensor, "B S_logits vocab_size"] = self.lm_head(output.last_hidden_state[:, -logits_to_keep:, :])
        return FWQwen3CausalLMOutput(
            loss=loss, logits=logits, state=output.state,
            past_key_values=output.past_key_values, hidden_states=output.hidden_states,
        )
