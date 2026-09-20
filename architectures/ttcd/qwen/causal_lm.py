from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped

from architectures.shared.qwen.causal_lm import (
    StatefulQwen3BridgeMemoryOutput,
    StatefulQwen3CausalLMOutput,
    StatefulQwen3ForCausalLM,
)
from .configuration import TTCDQwen3Config
from .mlp import TTCDQwen3MLP
from .model import TTCDQwen3Model
from ..states.model_state import TTCDModelState
from inference.generation import GenerationOutput
from inference.prefill import prefill


@dataclass
class TTCDQwen3CausalLMOutput(StatefulQwen3CausalLMOutput):
    state: TTCDModelState | None = None


@dataclass
class TTCDQwen3BridgeMemoryOutput(StatefulQwen3BridgeMemoryOutput):
    state: TTCDModelState | None = None


class TTCDQwen3ForCausalLM(StatefulQwen3ForCausalLM):
    """State-aware LM wrapper with memory-efficient causal training losses."""

    config_class = TTCDQwen3Config
    _no_split_modules = ["TTCDQwen3DecoderLayer"]
    _supports_flash_attn = False
    _supports_flex_attn = False

    def _build_model(self, config: TTCDQwen3Config) -> TTCDQwen3Model:
        return TTCDQwen3Model(config)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, TTCDQwen3MLP) and module.is_fast_weight_layer:
            module.reset_fast_weight_parameters()

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def prefill(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        execution_block_size: int | None = None,
        state: TTCDModelState | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> TTCDQwen3CausalLMOutput:
        """Prefill the backbone, then project only the final token to logits."""
        output = prefill(
            self.model, input_ids, execution_block_size=execution_block_size,
            state=state, persistent_mask=persistent_mask,
        )
        return TTCDQwen3CausalLMOutput(
            logits=self.lm_head(output.last_hidden_state[:, -1:, :]),
            state=output.state, past_key_values=output.past_key_values,
        )

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        input_ids: Int[torch.Tensor, "1 S"],
        state: TTCDModelState | None = None,
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
        eos_ids = self._validate_generation(
            input_ids,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            eos_token_id,
            generator,
        )

        output = self.prefill(
            input_ids, execution_block_size=execution_block_size,
            state=state, persistent_mask=persistent_mask,
        )

        def decode_token(
            token: Int[torch.Tensor, "1 1"],
            current_state: TTCDModelState,
        ) -> TTCDQwen3CausalLMOutput:
            return self(
                token,
                state=current_state,
                use_cache=True,
                logits_to_keep=1,
            )

        return self._generate_from_prefill(
            output,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            eos_ids,
            generator,
            decode_token,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        state: TTCDModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        labels: Int[torch.Tensor, "B S"] | None = None,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> TTCDQwen3CausalLMOutput:
        self._validate_lm_inputs(input_ids, labels, logits_to_keep)

        output = self.model(
            input_ids=input_ids, state=state, use_cache=use_cache,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
            persistent_mask=persistent_mask,
        )
        loss, logits = self._causal_head(
            output.last_hidden_state,
            labels,
            logits_to_keep,
        )
        return TTCDQwen3CausalLMOutput(
            loss=loss, logits=logits, state=output.state,
            past_key_values=output.past_key_values, hidden_states=output.hidden_states,
        )

    @jaxtyped(typechecker=beartype)
    def forward_bridge_memory(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        delayed_labels: Int[torch.Tensor, "B S"],
        all_token_loss_weight: float,
        delayed_answer_loss_weight: float,
        labels: Int[torch.Tensor, "B S"] | None = None,
        state: TTCDModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
    ) -> TTCDQwen3BridgeMemoryOutput:
        """Run the bridge-memory objective without expanding the standard HF forward API."""
        self._validate_lm_inputs(input_ids, labels, logits_to_keep)
        if delayed_labels.shape != input_ids.shape:
            raise ValueError("delayed_labels must match input_ids")

        output = self.model(
            input_ids=input_ids, state=state, use_cache=use_cache,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
            persistent_mask=persistent_mask,
        )
        loss, logits, all_token_loss, delayed_answer_loss = self._bridge_head(
            output.last_hidden_state,
            delayed_labels,
            all_token_loss_weight,
            delayed_answer_loss_weight,
            labels,
            logits_to_keep,
        )
        return TTCDQwen3BridgeMemoryOutput(
            loss=loss, logits=logits, state=output.state,
            past_key_values=output.past_key_values, hidden_states=output.hidden_states,
            all_token_loss=all_token_loss,
            delayed_answer_loss=delayed_answer_loss,
        )
