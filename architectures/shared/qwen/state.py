from dataclasses import dataclass

from transformers.cache_utils import Cache


@dataclass
class QwenSessionState:
    """Architecture-neutral state shared by one batch of Qwen sessions."""

    past_key_values: Cache | None = None
    tokens_seen: int = 0
