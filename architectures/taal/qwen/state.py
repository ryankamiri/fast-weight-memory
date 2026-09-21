from dataclasses import dataclass, field
from typing import TypeAlias

from architectures.shared.qwen.state import QwenSessionState
from architectures.titans.state import NeuralMemoryState


NeuralMemoryStates: TypeAlias = dict[int, NeuralMemoryState]


@dataclass
class TaalModelState(QwenSessionState):
    """Bounded K/V and one independent NeuralMemory state per Qwen layer."""

    memory_states: NeuralMemoryStates = field(default_factory=dict)
