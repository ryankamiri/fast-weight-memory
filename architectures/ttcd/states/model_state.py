from dataclasses import dataclass, field

from architectures.shared.qwen.state import QwenSessionState
from .mlp_state import TTCDMLPState


@dataclass
class TTCDModelState(QwenSessionState):
    """Caller-owned state for one batch/session, never checkpoint parameters."""

    mlp_states: dict[int, TTCDMLPState] = field(default_factory=dict)
