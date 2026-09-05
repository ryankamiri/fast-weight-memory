from dataclasses import dataclass, field

from transformers.cache_utils import Cache

from .mlp_state import FWMLPState


@dataclass
class FWModelState:
    """Caller-owned state for one batch/session, never checkpoint parameters."""

    past_key_values: Cache | None = None
    mlp_states: dict[int, FWMLPState] = field(default_factory=dict)
    tokens_seen: int = 0
