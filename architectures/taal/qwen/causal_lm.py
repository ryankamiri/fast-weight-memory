from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped

from architectures.shared.qwen.causal_lm import (
    StatefulQwen3BridgeMemoryOutput,
    StatefulQwen3CausalLMOutput,
    StatefulQwen3ForCausalLM,
)
from architectures.titans.neural_memory import NeuralMemory
from inference.generation import GenerationOutput

from .configuration import TaalQwen3Config
from .model import TaalQwen3Model
from .state import TaalModelState


@dataclass
class TaalQwen3CausalLMOutput(StatefulQwen3CausalLMOutput):
    state: TaalModelState | None = None


@dataclass
class TaalQwen3BridgeMemoryOutput(StatefulQwen3BridgeMemoryOutput):
    state: TaalModelState | None = None


class TaalQwen3ForCausalLM(StatefulQwen3ForCausalLM):
    """Qwen causal LM with one recurrent Titans memory before each decoder."""

    config_class = TaalQwen3Config
    _no_split_modules = ["TaalQwen3DecoderLayer"]
    _supports_flash_attn = False
    _supports_flex_attn = False

    def _build_model(self, config: TaalQwen3Config) -> TaalQwen3Model:
        return TaalQwen3Model(config)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NeuralMemory):
            # The wrapper's post_init traverses the already-built backbone a
            # second time, so restore the configured online-update controls.
            module.reset_update_controls()

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def prefill(
        self,
        input_ids: Int[torch.Tensor, "B S_prompt"],
        execution_block_size: int | None = None,
        state: TaalModelState | None = None,
        write_mask: Bool[torch.Tensor, "B S_prompt"] | None = None,
        persistent_mask: Bool[torch.Tensor, "S_prompt"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool = True,
    ) -> TaalQwen3CausalLMOutput:
        """Process one semantic segment in bounded execution blocks."""
        if self.training:
            raise ValueError("prefill requires evaluation mode; call model.eval() first")
        if execution_block_size is None:
            execution_block_size = self.config.working_memory_size
        if type(execution_block_size) is not int or execution_block_size < 1:
            raise ValueError("execution_block_size must be a positive integer")
        if execution_block_size > self.config.working_memory_size:
            raise ValueError(
                "execution_block_size cannot exceed working_memory_size"
            )
        B, S_prompt = input_ids.shape
        if B == 0 or S_prompt == 0:
            raise ValueError("prefill requires a nonempty batch and prompt")
        if write_mask is not None and write_mask.shape != input_ids.shape:
            raise ValueError("write_mask must match input_ids")
        if persistent_mask is not None and persistent_mask.shape != (S_prompt,):
            raise ValueError("persistent_mask must have shape [S_prompt]")
        if type(prepend_memory_tokens) is not bool:
            raise ValueError("prepend_memory_tokens must be boolean")

        # Reject an oversized persistent prefix before any block mutates state.
        if persistent_mask is not None:
            cache = None if state is None else state.past_key_values
            retained = None if cache is None else cache.persistent_positions
            if retained is None:
                retained_count = 0
            else:
                (retained_count,) = retained.shape
            if (
                retained_count + int(persistent_mask.sum())
                > self.config.max_persistent_kv_tokens
            ):
                raise ValueError(
                    "Persistent KV tokens exceed "
                    "max_persistent_kv_tokens="
                    f"{self.config.max_persistent_kv_tokens}"
                )

        output = None
        for start in range(0, S_prompt, execution_block_size):
            end = min(start + execution_block_size, S_prompt)
            input_block: Int[torch.Tensor, "B S_block"] = input_ids[:, start:end]
            block_write_mask = (
                None if write_mask is None else write_mask[:, start:end]
            )
            block_persistent_mask = (
                None if persistent_mask is None else persistent_mask[start:end]
            )
            output = self.model(
                input_ids=input_block,
                state=state,
                use_cache=True,
                write_mask=block_write_mask,
                persistent_mask=block_persistent_mask,
                memory_read_scale=memory_read_scale,
                # TaaL prepends P_l at every transformer layer for each new
                # utterance. Execution blocks only split that utterance, so P_l
                # enters its first block rather than every compute block.
                prepend_memory_tokens=prepend_memory_tokens and start == 0,
            )
            state = output.state

        return TaalQwen3CausalLMOutput(
            logits=self.lm_head(output.last_hidden_state[:, -1:, :]),
            state=output.state,
            past_key_values=output.past_key_values,
        )

    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        input_ids: Int[torch.Tensor, "1 S"],
        state: TaalModelState | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int | list[int] | None = None,
        execution_block_size: int | None = None,
        write_mask: Bool[torch.Tensor, "1 S"] | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool = True,
        generator: torch.Generator | None = None,
    ) -> GenerationOutput[TaalModelState]:
        """Prefill one segment, then decode while carrying KV and memory state."""
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
            input_ids,
            execution_block_size=execution_block_size,
            state=state,
            write_mask=write_mask,
            persistent_mask=persistent_mask,
            memory_read_scale=memory_read_scale,
            prepend_memory_tokens=prepend_memory_tokens,
        )

        def decode_token(
            token: Int[torch.Tensor, "1 1"],
            current_state: TaalModelState,
        ) -> TaalQwen3CausalLMOutput:
            return self(
                token,
                state=current_state,
                use_cache=True,
                logits_to_keep=1,
                memory_read_scale=memory_read_scale,
                prepend_memory_tokens=False,
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
        state: TaalModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        labels: Int[torch.Tensor, "B S"] | None = None,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool | None = None,
    ) -> TaalQwen3CausalLMOutput:
        self._validate_lm_inputs(input_ids, labels, logits_to_keep)
        output = self.model(
            input_ids=input_ids,
            state=state,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            write_mask=write_mask,
            persistent_mask=persistent_mask,
            memory_read_scale=memory_read_scale,
            prepend_memory_tokens=prepend_memory_tokens,
        )
        loss, logits = self._causal_head(
            output.last_hidden_state,
            labels,
            logits_to_keep,
        )
        return TaalQwen3CausalLMOutput(
            loss=loss,
            logits=logits,
            state=output.state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
        )

    @jaxtyped(typechecker=beartype)
    def forward_bridge_memory(
        self,
        input_ids: Int[torch.Tensor, "B S"],
        delayed_labels: Int[torch.Tensor, "B S"],
        all_token_loss_weight: float,
        delayed_answer_loss_weight: float,
        labels: Int[torch.Tensor, "B S"] | None = None,
        state: TaalModelState | None = None,
        use_cache: bool = False,
        attention_mask: Bool[torch.Tensor, "B S_kv"] | None = None,
        logits_to_keep: int = 0,
        output_hidden_states: bool = False,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
        persistent_mask: Bool[torch.Tensor, "S"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool | None = None,
    ) -> TaalQwen3BridgeMemoryOutput:
        """Run delayed-recall training without widening the standard HF API."""
        self._validate_lm_inputs(input_ids, labels, logits_to_keep)
        if delayed_labels.shape != input_ids.shape:
            raise ValueError("delayed_labels must match input_ids")
        output = self.model(
            input_ids=input_ids,
            state=state,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            write_mask=write_mask,
            persistent_mask=persistent_mask,
            memory_read_scale=memory_read_scale,
            prepend_memory_tokens=prepend_memory_tokens,
        )
        loss, logits, all_token_loss, delayed_answer_loss = self._bridge_head(
            output.last_hidden_state,
            delayed_labels,
            all_token_loss_weight,
            delayed_answer_loss_weight,
            labels,
            logits_to_keep,
        )
        return TaalQwen3BridgeMemoryOutput(
            loss=loss,
            logits=logits,
            state=output.state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            all_token_loss=all_token_loss,
            delayed_answer_loss=delayed_answer_loss,
        )
