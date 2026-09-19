from abc import ABC, abstractmethod

from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


class StatefulQwen3Model(Qwen3PreTrainedModel, ABC):
    """Shared host-model modules for Qwen architectures with recurrent state."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            self._build_decoder_layer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.post_init()

    @abstractmethod
    def _build_decoder_layer(self, config: Qwen3Config, layer_idx: int) -> nn.Module:
        """Construct one architecture-specific decoder layer."""

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Stateful updates require non-reentrant checkpointing."""
        checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {})
        if checkpoint_kwargs.get("use_reentrant", False) is not False:
            raise ValueError(
                f"{type(self).__name__} requires use_reentrant=False for state gradients"
            )
        checkpoint_kwargs["use_reentrant"] = False
        super().gradient_checkpointing_enable(checkpoint_kwargs)
