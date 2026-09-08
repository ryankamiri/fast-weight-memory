"""Generation results and next-token sampling for stateful inference."""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int

from architectures.states.model_state import FWModelState


@dataclass
class GenerationOutput:
    # New tokens only, including EOS when emitted. The prompt is not repeated.
    token_ids: Int[torch.Tensor, "1 S_generated"]
    state: FWModelState
    stop_reason: Literal["eos", "max_new_tokens"]


def sample_token(
    logits: Float[torch.Tensor, "1 vocab_size"],
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None,
) -> Int[torch.Tensor, "1 1"]:
    if not torch.isfinite(logits).all():
        raise ValueError("Cannot generate from non-finite logits")
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    if top_k > 0:
        cutoff = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_scores, indices = scores.sort(dim=-1, descending=True)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        # Keep the first token that crosses the threshold, and at least one.
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = torch.zeros_like(remove).scatter(1, indices, remove)
        scores = scores.masked_fill(remove, -torch.inf)
    return torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)
