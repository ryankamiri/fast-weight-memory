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
    """Titans neural memory with configurable chunkwise online updates."""

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
        if not 0 <= state.pending_count < self.config.chunk_size:
            raise ValueError(
                "state.pending_count must be between 0 and chunk_size - 1"
            )
        if (state.provisional_weights is None) != (state.pending_count == 0):
            raise ValueError(
                "state.provisional_weights must exist exactly when a chunk is pending"
            )

        collections = [
            ("weights", state.weights),
            ("momentum", state.momentum),
        ]
        if state.provisional_weights is not None:
            if set(state.provisional_weights) != parameter_names:
                raise ValueError(
                    "state.provisional_weights do not match the memory MLP parameters"
                )
            collections.append(("provisional_weights", state.provisional_weights))
        for collection_name, collection in collections:
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
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
    ) -> dict[str, Float32[torch.Tensor, "B C D D"]]:
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

            # For one session, share its weights while mapping over the C keys
            # and values in the chunk.
            per_token_grad = vmap(grad(memory_loss), in_dims=(None, 0, 0))
            # Then map that per-session computation over the B independent
            # memory trajectories in the batch.
            batched_grad = vmap(per_token_grad, in_dims=(0, 0, 0))
            return batched_grad(weights, keys, values)

    def _update(
        self,
        state: NeuralMemoryState,
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
        alpha: Float[torch.Tensor, "B C 1"],
        eta: Float[torch.Tensor, "B C 1"],
        theta: Float[torch.Tensor, "B C 1"],
    ) -> tuple[
        NeuralMemoryState,
        dict[str, Float32[torch.Tensor, "B C D D"]],
    ]:
        # All C gradients are deliberately stale with respect to the provisional
        # prefix states: they share the same committed chunk-start weights so
        # they can run in parallel. The next chunk refreshes its gradients from
        # the newly committed final weights of this chunk.
        gradients: dict[str, Float32[torch.Tensor, "B C D D"]] = (
            self._surprise_gradients(state.weights, keys, values)
        )

        B, C, _ = keys.shape
        forget: Float32[torch.Tensor, "B C 1 1"] = alpha.reshape(
            B, C, 1, 1
        ).float()
        momentum_retention: Float32[torch.Tensor, "B C 1 1"] = eta.reshape(
            B, C, 1, 1
        ).float()
        write_strength: Float32[torch.Tensor, "B C 1 1"] = theta.reshape(
            B, C, 1, 1
        ).float()

        weights = state.current_weights
        momentum = state.momentum
        weights_by_token: dict[
            str, list[Float32[torch.Tensor, "B D D"]]
        ] = {name: [] for name in weights}

        # Gradients are parallel across C and share the chunk-start weights.
        # This inexpensive recurrence produces the state visible at each token.
        for token_index in range(C):
            next_weights: dict[str, Float32[torch.Tensor, "B D D"]] = {}
            next_momentum: dict[str, Float32[torch.Tensor, "B D D"]] = {}
            for name, weight in weights.items():
                surprise: Float32[torch.Tensor, "B D D"] = (
                    momentum_retention[:, token_index] * momentum[name]
                    - write_strength[:, token_index] * gradients[name][:, token_index]
                )
                next_momentum[name] = surprise
                next_weights[name] = (
                    1.0 - forget[:, token_index]
                ) * weight + surprise
                weights_by_token[name].append(next_weights[name])
            weights = next_weights
            momentum = next_momentum

        stacked_weights: dict[str, Float32[torch.Tensor, "B C D D"]] = {
            name: torch.stack(token_weights, dim=1)
            for name, token_weights in weights_by_token.items()
        }

        # forward splits work at chunk boundaries: crossing one here would be
        # invalid because the next chunk must refresh gradients from the newly
        # committed weights.
        pending_count = state.pending_count + C
        if pending_count == self.config.chunk_size:
            committed_weights = weights
            provisional_weights = None
            pending_count = 0
        else:
            committed_weights = state.weights
            provisional_weights = weights

        if not self.training:
            committed_weights = {
                name: value.detach() for name, value in committed_weights.items()
            }
            momentum = {name: value.detach() for name, value in momentum.items()}
            stacked_weights = {
                name: value.detach() for name, value in stacked_weights.items()
            }
            if provisional_weights is not None:
                provisional_weights = {
                    name: value.detach()
                    for name, value in provisional_weights.items()
                }

        return (
            NeuralMemoryState(
                weights=committed_weights,
                momentum=momentum,
                provisional_weights=provisional_weights,
                pending_count=pending_count,
                query_conv_history=state.query_conv_history,
                key_conv_history=state.key_conv_history,
                value_conv_history=state.value_conv_history,
            ),
            stacked_weights,
        )

    def _read(
        self,
        weights: dict[str, Float32[torch.Tensor, "B C D D"]],
        queries: Float[torch.Tensor, "B C D"],
    ) -> Float[torch.Tensor, "B C D"]:
        def read_one(
            sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
            query: Float32[torch.Tensor, "D"],
        ) -> Float[torch.Tensor, "D"]:
            return functional_call(
                self.memory_mlp,
                sample_weights,
                (query,),
            )

        per_session_read = vmap(read_one, in_dims=(0, 0))
        batched_read = vmap(per_session_read, in_dims=(0, 0))
        return batched_read(weights, queries.float())

    def _process_chunk(
        self,
        state: NeuralMemoryState,
        queries: Float[torch.Tensor, "B C D"],
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
        alpha: Float[torch.Tensor, "B C 1"],
        eta: Float[torch.Tensor, "B C 1"],
        theta: Float[torch.Tensor, "B C 1"],
        output_dtype: torch.dtype,
    ) -> tuple[Float[torch.Tensor, "B C D"], NeuralMemoryState]:
        state, weights_by_token = self._update(
            state,
            keys,
            values,
            alpha,
            eta,
            theta,
        )
        output: Float[torch.Tensor, "B C D"] = self._read(
            weights_by_token, queries
        ).to(output_dtype)
        return output, state

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

        C = self.config.chunk_size
        outputs: list[Float[torch.Tensor, "B C D"]] = []
        start = 0

        # Finish a memory chunk carried over from an earlier forward call.
        if state.pending_count > 0:
            end = min(sequence_length, C - state.pending_count)
            output, state = self._process_chunk(
                state,
                queries[:, start:end],
                keys[:, start:end].float(),
                values[:, start:end].float(),
                alpha[:, start:end],
                eta[:, start:end],
                theta[:, start:end],
                inputs.dtype,
            )
            outputs.append(output)
            start = end

        # [B, S, D] -> [B, N, C, D]. N chunks are recurrent; the C tokens
        # within each chunk share its starting weights and are tensorized.
        full_length = ((sequence_length - start) // C) * C
        # Prefill may contain full chunks; token-by-token decoding usually does not.
        if full_length > 0:
            end = start + full_length
            N = full_length // C
            query_chunks: Float[torch.Tensor, "B N C D"] = queries[
                :, start:end
            ].reshape(batch_size, N, C, dim)
            key_chunks: Float32[torch.Tensor, "B N C D"] = keys[
                :, start:end
            ].float().reshape(batch_size, N, C, dim)
            value_chunks: Float32[torch.Tensor, "B N C D"] = values[
                :, start:end
            ].float().reshape(batch_size, N, C, dim)
            alpha_chunks: Float[torch.Tensor, "B N C 1"] = alpha[
                :, start:end
            ].reshape(batch_size, N, C, 1)
            eta_chunks: Float[torch.Tensor, "B N C 1"] = eta[
                :, start:end
            ].reshape(batch_size, N, C, 1)
            theta_chunks: Float[torch.Tensor, "B N C 1"] = theta[
                :, start:end
            ].reshape(batch_size, N, C, 1)

            for chunk_index in range(N):
                output, state = self._process_chunk(
                    state,
                    query_chunks[:, chunk_index],
                    key_chunks[:, chunk_index],
                    value_chunks[:, chunk_index],
                    alpha_chunks[:, chunk_index],
                    eta_chunks[:, chunk_index],
                    theta_chunks[:, chunk_index],
                    inputs.dtype,
                )
                outputs.append(output)
            start = end

        # A final partial chunk remains open in state for the next call.
        if start < sequence_length:
            output, state = self._process_chunk(
                state,
                queries[:, start:],
                keys[:, start:].float(),
                values[:, start:].float(),
                alpha[:, start:],
                eta[:, start:],
                theta[:, start:],
                inputs.dtype,
            )
            outputs.append(output)

        state = replace(
            state,
            query_conv_history=query_history,
            key_conv_history=key_history,
            value_conv_history=value_history,
        )
        return torch.cat(outputs, dim=1), state
