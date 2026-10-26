from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NeuralMemoryConfig:
    """Architecture and stable initial update settings for Titans NeuralMemory."""

    dim: int
    depth: int = 2
    conv_kernel_size: int = 4
    chunk_size: int = 1
    initial_forget: float = 0.01
    initial_momentum: float = 0.9
    initial_write_strength: float = 0.1
    pre_rmsnorm: bool = False
    mlp_output_layernorm: bool = False
    read_layernorm: bool = False
    norm_learned_scale: bool = False
    norm_residual: bool = False
    normalization_eps: float = 1e-6

    def __post_init__(self):
        for name in ("pre_rmsnorm", "mlp_output_layernorm", "read_layernorm", "norm_learned_scale", "norm_residual"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

        if (type(self.normalization_eps) not in (int, float)
                or not math.isfinite(self.normalization_eps) or self.normalization_eps <= 0):
            raise ValueError("normalization_eps must be finite and positive")

        if (self.norm_learned_scale or self.norm_residual) and not self.mlp_output_layernorm:
            raise ValueError("Memory normalization scale/residual requires normalized writes and reads")

        for name in ("dim", "depth", "conv_kernel_size", "chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initial_forget", "initial_momentum", "initial_write_strength"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and strictly between 0 and 1")
