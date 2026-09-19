import torch
from beartype import beartype
from jaxtyping import Bool, Int, jaxtyped

from architectures.ttcd.qwen.model import TTCDQwen3Model, TTCDQwen3ModelOutput
from architectures.ttcd.states.model_state import TTCDModelState


@torch.inference_mode()
@jaxtyped(typechecker=beartype)
def prefill(
    model: TTCDQwen3Model,
    input_ids: Int[torch.Tensor, "B S_prompt"],
    execution_block_size: int | None = None,
    state: TTCDModelState | None = None,
    persistent_mask: Bool[torch.Tensor, "S_prompt"] | None = None,
) -> TTCDQwen3ModelOutput:
    """Process an unpadded prompt in blocks."""
    if model.training:
        raise ValueError("prefill requires evaluation mode; call model.eval() first")
    if execution_block_size is None:
        execution_block_size = model.config.chunk_size
    if type(execution_block_size) is not int or execution_block_size < 1:
        raise ValueError("execution_block_size must be a positive integer")
    B, S_prompt = input_ids.shape
    if B == 0 or S_prompt == 0:
        raise ValueError("prefill requires a nonempty batch and prompt")

    # Reject an oversized prompt before any execution block mutates the session.
    if persistent_mask is not None:
        cache = None if state is None else state.past_key_values
        retained = None if cache is None else cache.layers[0].is_persistent
        retained_count = 0 if retained is None else int(retained.sum())
        if (
            retained_count + int(persistent_mask.sum())
            > model.config.max_persistent_kv_tokens
        ):
            raise ValueError(
                "Persistent KV tokens exceed "
                "max_persistent_kv_tokens="
                f"{model.config.max_persistent_kv_tokens}"
            )
    for start in range(0, S_prompt, execution_block_size):
        input_block: Int[torch.Tensor, "B S_block"] = input_ids[:, start : start + execution_block_size]
        block_persistent = None if persistent_mask is None else persistent_mask[start : start + execution_block_size]
        output = model(
            input_ids=input_block, state=state, use_cache=True,
            persistent_mask=block_persistent,
        )
        state = output.state

    return output
