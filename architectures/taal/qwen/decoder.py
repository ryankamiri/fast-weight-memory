import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from typing_extensions import Unpack
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.utils.generic import TransformersKwargs

from architectures.taal.configuration import TaalLayerConfig
from architectures.taal.layer import TaalLayer
from architectures.titans.state import NeuralMemoryState


class TaalQwen3DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder preceded by one stateful Titans-as-a-Layer adapter."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        taal_config: TaalLayerConfig,
    ):
        if taal_config.model_dim != config.hidden_size:
            raise ValueError(
                "taal_config.model_dim must equal the Qwen hidden size"
            )
        super().__init__(config, layer_idx)
        self.taal = TaalLayer(taal_config)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B S D_model"],
        attention_mask: (
            Float[torch.Tensor, "#B #H S S_kv"]
            | Bool[torch.Tensor, "#B #H S S_kv"]
            | None
        ) = None,
        position_ids: Int[torch.Tensor, "#B S"] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: Int[torch.Tensor, "S"] | None = None,
        position_embeddings: tuple[
            Float[torch.Tensor, "#B S D_head"],
            Float[torch.Tensor, "#B S D_head"],
        ]
        | None = None,
        memory_state: NeuralMemoryState | None = None,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Float[torch.Tensor, "B S D_model"], NeuralMemoryState]:
        # Memory changes the representation entering both the ordinary Qwen
        # residual path and attention. Its persistent tokens stay inside TaaL.
        hidden_states, next_memory_state = self.taal(
            hidden_states,
            state=memory_state,
            write_mask=write_mask,
        )
        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return hidden_states, next_memory_state
