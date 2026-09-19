from dataclasses import dataclass, field

from architectures.shared.qwen.state import QwenSessionState
from architectures.titans.state import NeuralMemoryState


@dataclass
class TaalModelState(QwenSessionState):
    """Bounded K/V and one independent NeuralMemory state per Qwen layer."""

    memory_states: dict[int, NeuralMemoryState] = field(default_factory=dict)
