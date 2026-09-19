from dataclasses import replace

import torch
from beartype import beartype
from jaxtyping import Float, Float32, jaxtyped
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, grad, vmap

from architectures.shared.causal_conv import CausalDepthwiseConv1d

from .configuration import NeuralMemoryConfig
from .state import NeuralMemoryState


class MemoryMLP(nn.Module):
    """Same-width, bias-free MLP whose parameters form the fast memory state."""

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(depth)
        )
        self.activation = nn.SiLU()

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[torch.Tensor, "D"],
    ) -> Float[torch.Tensor, "D"]:
        # This is intentionally one session vector, not B x D: vmap owns the
        # outer batch dimension because every session has different fast weights.
        for index, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if index + 1 != len(self.layers):
                inputs = self.activation(inputs)
        return inputs


class NeuralMemory(nn.Module):
    """Titans neural memory with exact sequential online updates."""

    def __init__(self, config: NeuralMemoryConfig):
        super().__init__()
        self.config = config
        dim = config.dim

        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.query_conv = CausalDepthwiseConv1d(dim, config.conv_kernel_size)
        self.key_conv = CausalDepthwiseConv1d(dim, config.conv_kernel_size)
        self.value_conv = CausalDepthwiseConv1d(dim, config.conv_kernel_size)
        self.activation = nn.SiLU()
        self.memory_mlp = MemoryMLP(dim, config.depth)
        self.forget_projection = nn.Linear(dim, 1)
        self.momentum_projection = nn.Linear(dim, 1)
        self.write_strength_projection = nn.Linear(dim, 1)
        with torch.no_grad():
            for projection, initial_value in (
                (self.forget_projection, config.initial_forget),
                (self.momentum_projection, config.initial_momentum),
                (self.write_strength_projection, config.initial_write_strength),
            ):
                projection.weight.zero_()
                projection.bias.fill_(torch.logit(torch.tensor(initial_value)))

    def initial_state(self, batch_size: int) -> NeuralMemoryState:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        # Disable an enclosing inference_mode so future online writes can request
        # gradients with respect to these fast parameter copies.
        with torch.inference_mode(False):
            weights: dict[str, Float32[torch.Tensor, "B D D"]] = {
                name: parameter.float().unsqueeze(0).expand(
                    batch_size, *parameter.shape
                ).clone()
                for name, parameter in self.memory_mlp.named_parameters()
            }
            momentum: dict[str, Float32[torch.Tensor, "B D D"]] = {
                name: torch.zeros_like(parameter)
                for name, parameter in weights.items()
            }
        return NeuralMemoryState(weights=weights, momentum=momentum)

    def _validate_state(self, state: NeuralMemoryState, batch_size: int) -> None:
        parameter_names = set(dict(self.memory_mlp.named_parameters()))
        if set(state.weights) != parameter_names:
            raise ValueError("state.weights do not match the memory MLP parameters")
        if set(state.momentum) != parameter_names:
            raise ValueError("state.momentum does not match the memory MLP parameters")
        for collection_name, collection in (
            ("weights", state.weights),
            ("momentum", state.momentum),
        ):
            for name, value in collection.items():
                expected = dict(self.memory_mlp.named_parameters())[name].shape
                if value.shape != (batch_size, *expected):
                    raise ValueError(
                        f"state.{collection_name}[{name!r}] must have shape "
                        f"{(batch_size, *expected)}, received {tuple(value.shape)}"
                    )
                if value.dtype != torch.float32:
                    raise ValueError(f"state.{collection_name}[{name!r}] must be float32")

    def _surprise_gradients(
        self,
        weights: dict[str, Float32[torch.Tensor, "B D D"]],
        keys: Float32[torch.Tensor, "B D"],
        values: Float32[torch.Tensor, "B D"],
    ) -> dict[str, Float32[torch.Tensor, "B D D"]]:
        # Online writes still require a local gradient during no-grad/inference
        # generation. Only training retains the higher-order graph used by the
        # outer delayed-answer objective.
        with torch.inference_mode(False), torch.enable_grad():
            if not self.training:
                weights = {
                    name: value.detach().clone()
                    for name, value in weights.items()
                }
                keys = keys.detach().clone()
                values = values.detach().clone()

            def memory_loss(
                sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
                key: Float32[torch.Tensor, "D"],
                target: Float32[torch.Tensor, "D"],
            ) -> Float[torch.Tensor, ""]:
                # Run the normal MLP forward with this sample's current fast
                # weights instead of the module's shared template parameters.
                prediction: Float[torch.Tensor, "D"] = functional_call(
                    self.memory_mlp, sample_weights, (key,)
                )
                # L_memory = (1 / D) * ||M_W(key) - value||^2_2
                return F.mse_loss(prediction, target)

            per_example_grad = vmap(
                grad(memory_loss), in_dims=(0, 0, 0)
            )
            return per_example_grad(weights, keys, values)

    def _update(
        self,
        state: NeuralMemoryState,
        key: Float32[torch.Tensor, "B D"],
        value: Float32[torch.Tensor, "B D"],
        alpha: Float[torch.Tensor, "B 1"],
        eta: Float[torch.Tensor, "B 1"],
        theta: Float[torch.Tensor, "B 1"],
    ) -> NeuralMemoryState:
        gradients: dict[str, Float32[torch.Tensor, "B D D"]] = (
            self._surprise_gradients(state.weights, key, value)
        )
        next_weights: dict[str, Float32[torch.Tensor, "B D D"]] = {}
        next_momentum: dict[str, Float32[torch.Tensor, "B D D"]] = {}

        B = key.shape[0]
        # One update control per session broadcasts across its D x D weights.
        forget: Float32[torch.Tensor, "B 1 1"] = alpha.reshape(B, 1, 1).float()
        momentum_retention: Float32[torch.Tensor, "B 1 1"] = eta.reshape(
            B, 1, 1
        ).float()
        write_strength: Float32[torch.Tensor, "B 1 1"] = theta.reshape(
            B, 1, 1
        ).float()

        for name, weight in state.weights.items():
            surprise: Float32[torch.Tensor, "B D D"] = (
                momentum_retention * state.momentum[name]
                - write_strength * gradients[name]
            )
            next_momentum[name] = surprise
            next_weights[name] = (1.0 - forget) * weight + surprise

        if not self.training:
            next_weights = {name: value.detach() for name, value in next_weights.items()}
            next_momentum = {name: value.detach() for name, value in next_momentum.items()}

        return NeuralMemoryState(
            weights=next_weights,
            momentum=next_momentum,
            query_conv_history=state.query_conv_history,
            key_conv_history=state.key_conv_history,
            value_conv_history=state.value_conv_history,
        )

    def _read(
        self,
        weights: dict[str, Float32[torch.Tensor, "B D D"]],
        queries: Float[torch.Tensor, "B D"],
    ) -> Float[torch.Tensor, "B D"]:
        def read_one(
            sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
            query: Float32[torch.Tensor, "D"],
        ) -> Float[torch.Tensor, "D"]:
            return functional_call(
                self.memory_mlp,
                sample_weights,
                (query,),
            )

        # Each session owns independent fast weights, so reads can be mapped
        # across B even though memory updates remain sequential across tokens.
        batched_read = vmap(read_one, in_dims=(0, 0))
        return batched_read(weights, queries.float())

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[torch.Tensor, "B S D"],
        state: NeuralMemoryState | None = None,
    ) -> tuple[Float[torch.Tensor, "B S D"], NeuralMemoryState]:
        batch_size, sequence_length, dim = inputs.shape
        if dim != self.config.dim:
            raise ValueError(f"Expected dimension {self.config.dim}, received {dim}")
        if state is None:
            state = self.initial_state(batch_size)
        self._validate_state(state, batch_size)
        if sequence_length == 0:
            return inputs.new_empty(batch_size, 0, dim), state

        queries: Float[torch.Tensor, "B S D"]
        query_history: Float[torch.Tensor, "B D S_history"]
        queries, query_history = self.query_conv(
            self.query_projection(inputs), state.query_conv_history
        )
        keys: Float[torch.Tensor, "B S D"]
        key_history: Float[torch.Tensor, "B D S_history"]
        keys, key_history = self.key_conv(
            self.key_projection(inputs), state.key_conv_history
        )
        values: Float[torch.Tensor, "B S D"]
        value_history: Float[torch.Tensor, "B D S_history"]
        values, value_history = self.value_conv(
            self.value_projection(inputs), state.value_conv_history
        )
        queries = F.normalize(self.activation(queries), dim=-1, eps=1e-6)
        keys = F.normalize(self.activation(keys), dim=-1, eps=1e-6)
        values = self.activation(values)
        alpha: Float[torch.Tensor, "B S 1"] = self.forget_projection(
            inputs
        ).sigmoid()
        eta: Float[torch.Tensor, "B S 1"] = self.momentum_projection(
            inputs
        ).sigmoid()
        theta: Float[torch.Tensor, "B S 1"] = self.write_strength_projection(
            inputs
        ).sigmoid()

        outputs: list[Float[torch.Tensor, "B D"]] = []
        for token_index in range(sequence_length):
            state = self._update(
                state,
                keys[:, token_index].float(),
                values[:, token_index].float(),
                alpha[:, token_index],
                eta[:, token_index],
                theta[:, token_index],
            )
            outputs.append(self._read(
                state.weights, queries[:, token_index]
            ).to(inputs.dtype))

        state = replace(
            state,
            query_conv_history=query_history,
            key_conv_history=key_history,
            value_conv_history=value_history,
        )
        return torch.stack(outputs, dim=1), state
