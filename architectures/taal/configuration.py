from dataclasses import dataclass
import math

from architectures.titans.configuration import NeuralMemoryConfig


@dataclass(frozen=True)
class TaalLayerConfig:
    """Configuration for one Titans-as-a-Layer residual adapter."""

    model_dim: int
    memory: NeuralMemoryConfig
    num_persistent_tokens: int = 8
    persistent_init_std: float = 0.02

    def __post_init__(self):
        if type(self.model_dim) is not int or self.model_dim < 1:
            raise ValueError("model_dim must be a positive integer")
        if not isinstance(self.memory, NeuralMemoryConfig):
            raise TypeError("memory must be a NeuralMemoryConfig")
        if (
            type(self.num_persistent_tokens) is not int
            or self.num_persistent_tokens < 1
        ):
            raise ValueError("num_persistent_tokens must be a positive integer")
        if (
            type(self.persistent_init_std) not in (int, float)
            or not math.isfinite(self.persistent_init_std)
            or self.persistent_init_std <= 0
        ):
            raise ValueError("persistent_init_std must be finite and positive")
