from dataclasses import dataclass, field

from transformers.cache_utils import Cache

from .mlp_state import TTCDMLPState


@dataclass
class TTCDModelState:
    """Caller-owned state for one batch/session, never checkpoint parameters."""

    past_key_values: Cache | None = None
    mlp_states: dict[int, TTCDMLPState] = field(default_factory=dict)
    tokens_seen: int = 0
