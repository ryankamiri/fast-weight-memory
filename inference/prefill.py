import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped

from architectures.qwen.model import FWQwen3Model, FWQwen3ModelOutput
from architectures.qwen.causal_lm import FWQwen3ForCausalLM, FWQwen3CausalLMOutput
from architectures.states.model_state import FWModelState


@torch.inference_mode()
@jaxtyped(typechecker=beartype)
def prefill(
    model: FWQwen3Model | FWQwen3ForCausalLM,
    input_ids: Int[torch.Tensor, "B S_prompt"],
    execution_block_size: int | None = None,
    state: FWModelState | None = None,
    persistent_mask: Bool[torch.Tensor, "S_prompt"] | None = None,
) -> FWQwen3ModelOutput | FWQwen3CausalLMOutput:
    """Process an unpadded prompt in blocks.

    Backbones return final-block hidden states; LMs return final-token logits.
    Both return full session state. Continue decoding under inference_mode().
    persistent_mask identifies incoming tokens whose K/V must survive eviction.
    """
    if model.training:
        raise ValueError("prefill requires evaluation mode; call model.eval() first")
    if execution_block_size is None:
        execution_block_size = model.config.chunk_size
    if type(execution_block_size) is not int or execution_block_size < 1:
        raise ValueError("execution_block_size must be a positive integer")
    B, S_prompt = input_ids.shape
    if B == 0 or S_prompt == 0:
        raise ValueError("prefill requires a nonempty batch and prompt")

    backbone = model.model if isinstance(model, FWQwen3ForCausalLM) else model
    # Reject an oversized prompt before any execution block mutates the session.
    if persistent_mask is not None:
        cache = None if state is None else state.past_key_values
        retained = None if cache is None else cache.layers[0].is_persistent
        retained_count = 0 if retained is None else int(retained.sum())
        if retained_count + int(persistent_mask.sum()) > model.config.max_persistent_tokens:
            raise ValueError(f"Persistent tokens exceed max_persistent_tokens={model.config.max_persistent_tokens}")
    for start in range(0, S_prompt, execution_block_size):
        input_block: Int[torch.Tensor, "B S_block"] = input_ids[:, start : start + execution_block_size]
        block_persistent = None if persistent_mask is None else persistent_mask[start : start + execution_block_size]
        output = backbone(
            input_ids=input_block, state=state, use_cache=True,
            persistent_mask=block_persistent,
        )
        state = output.state

    if isinstance(model, FWQwen3ForCausalLM):
        return FWQwen3CausalLMOutput(
            logits=model.lm_head(output.last_hidden_state[:, -1:, :]),
            state=state, past_key_values=output.past_key_values,
        )
    return output
