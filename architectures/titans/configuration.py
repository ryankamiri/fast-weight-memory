from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NeuralMemoryConfig:
    """Architecture and stable initial update settings for Titans NeuralMemory."""

    dim: int
    depth: int = 2
    conv_kernel_size: int = 4
    initial_forget: float = 0.01
    initial_momentum: float = 0.9
    initial_write_strength: float = 0.1

    def __post_init__(self):
        for name in ("dim", "depth", "conv_kernel_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initial_forget", "initial_momentum", "initial_write_strength"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and strictly between 0 and 1")
