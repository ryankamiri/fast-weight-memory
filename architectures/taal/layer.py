import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import nn

from architectures.titans.neural_memory import NeuralMemory
from architectures.titans.state import NeuralMemoryState

from .configuration import TaalLayerConfig


class TaalLayer(nn.Module):
    """Inject a stateful Titans memory read into a model residual stream."""

    def __init__(self, config: TaalLayerConfig):
        super().__init__()
        self.config = config
        memory_dim = config.memory.dim

        self.memory_projection_in = nn.Linear(
            config.model_dim,
            memory_dim,
            bias=False,
        )
        self.persistent_tokens = nn.Parameter(
            torch.empty(config.num_persistent_tokens, memory_dim)
        )
        nn.init.normal_(
            self.persistent_tokens,
            mean=0.0,
            std=config.persistent_init_std,
        )
        self.neural_memory = NeuralMemory(config.memory)
        self.memory_projection_out = nn.Linear(
            memory_dim,
            config.model_dim,
            bias=False,
        )
        # The adapter initially preserves the host model exactly.
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B S D_model"],
        state: NeuralMemoryState | None = None,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
        memory_read_scale: float = 1.0,
        prepend_memory_tokens: bool = True,
    ) -> tuple[
        Float[torch.Tensor, "B S D_model"],
        NeuralMemoryState,
    ]:
        B, S, _ = hidden_states.shape
        projected: Float[torch.Tensor, "B S D_memory"] = (
            self.memory_projection_in(hidden_states)
        )
        if prepend_memory_tokens:
            persistent: Float[torch.Tensor, "B N_persistent D_memory"] = (
                self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)
            )
            memory_inputs: Float[
                torch.Tensor, "B N_memory D_memory"
            ] = torch.cat((persistent, projected), dim=1)
        else:
            memory_inputs = projected

        memory_write_mask: Bool[torch.Tensor, "B N_memory"] | None = None
        if write_mask is not None and prepend_memory_tokens:
            persistent_write_mask: Bool[
                torch.Tensor, "B N_persistent"
            ] = torch.ones(
                B,
                self.config.num_persistent_tokens,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            memory_write_mask = torch.cat(
                (persistent_write_mask, write_mask),
                dim=1,
            )
        elif write_mask is not None:
            memory_write_mask = write_mask

        memory_output: Float[torch.Tensor, "B N_memory D_memory"]
        memory_output, next_state = self.neural_memory(
            memory_inputs,
            state=state,
            write_mask=memory_write_mask,
        )
        # Persistent tokens exist only inside the memory branch.
        if prepend_memory_tokens:
            # Learned persistent tokens are memory-call-only context. They are
            # inserted once per semantic segment, then removed before Qwen.
            memory_output = memory_output[:, self.config.num_persistent_tokens :]
        correction: Float[torch.Tensor, "B S D_model"] = (
            self.memory_projection_out(memory_output)
        )
        output: Float[torch.Tensor, "B S D_model"] = (
            hidden_states
            + memory_read_scale * torch.tanh(self.residual_gate) * correction
        )
        return output, next_state
