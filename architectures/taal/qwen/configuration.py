from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from architectures.taal.configuration import TaalLayerConfig
from architectures.titans.configuration import NeuralMemoryConfig


class TaalQwen3Config(Qwen3Config):
    """Serializable Qwen and Titans-as-a-Layer architecture settings."""

    model_type = "taal_qwen3"

    def __init__(
        self,
        working_memory_size: int = 4096,
        max_persistent_kv_tokens: int = 512,
        memory_dim: int = 256,
        memory_depth: int = 2,
        memory_conv_kernel_size: int = 4,
        memory_chunk_size: int = 1,
        memory_initial_forget: float = 0.01,
        memory_initial_momentum: float = 0.9,
        memory_initial_write_strength: float = 0.1,
        num_persistent_tokens: int = 8,
        persistent_init_std: float = 0.02,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        if type(working_memory_size) is not int or working_memory_size < 1:
            raise ValueError("working_memory_size must be a positive integer")
        if (
            type(max_persistent_kv_tokens) is not int
            or max_persistent_kv_tokens < 0
        ):
            raise ValueError(
                "max_persistent_kv_tokens must be a nonnegative integer"
            )

        memory = NeuralMemoryConfig(
            dim=memory_dim,
            depth=memory_depth,
            conv_kernel_size=memory_conv_kernel_size,
            chunk_size=memory_chunk_size,
            initial_forget=memory_initial_forget,
            initial_momentum=memory_initial_momentum,
            initial_write_strength=memory_initial_write_strength,
        )
        taal = TaalLayerConfig(
            model_dim=self.hidden_size,
            memory=memory,
            num_persistent_tokens=num_persistent_tokens,
            persistent_init_std=persistent_init_std,
        )

        self.working_memory_size = working_memory_size
        # Qwen attention K/V retained outside the rolling working-memory
        # window. This is distinct from TaaL's internal learned prefix tokens.
        self.max_persistent_kv_tokens = max_persistent_kv_tokens
        self.memory_dim = memory.dim
        self.memory_depth = memory.depth
        self.memory_conv_kernel_size = memory.conv_kernel_size
        self.memory_chunk_size = memory.chunk_size
        self.memory_initial_forget = memory.initial_forget
        self.memory_initial_momentum = memory.initial_momentum
        self.memory_initial_write_strength = memory.initial_write_strength
        self.num_persistent_tokens = taal.num_persistent_tokens
        self.persistent_init_std = taal.persistent_init_std

    def taal_layer_config(self) -> TaalLayerConfig:
        return TaalLayerConfig(
            model_dim=self.hidden_size,
            memory=NeuralMemoryConfig(
                dim=self.memory_dim,
                depth=self.memory_depth,
                conv_kernel_size=self.memory_conv_kernel_size,
                chunk_size=self.memory_chunk_size,
                initial_forget=self.memory_initial_forget,
                initial_momentum=self.memory_initial_momentum,
                initial_write_strength=self.memory_initial_write_strength,
            ),
            num_persistent_tokens=self.num_persistent_tokens,
            persistent_init_std=self.persistent_init_std,
        )
