from dataclasses import replace
from time import perf_counter

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Float32, jaxtyped
from torch import nn
from torch.nn import functional as F
from torch.func import grad, vmap

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
        self.trace_observer = None
        # Set only for an explicitly timed microbatch; no training-state effect.
        self.timing_observer = None
        self.reset_update_controls()

    @torch.no_grad()
    def reset_update_controls(self) -> None:
        """Restore configured online-update controls after host-model init."""
        for projection, initial_value in (
            (self.forget_projection, self.config.initial_forget),
            (self.momentum_projection, self.config.initial_momentum),
            (self.write_strength_projection, self.config.initial_write_strength),
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

    def zero_state(self, batch_size: int) -> NeuralMemoryState:
        """Create the explicit all-zero control state for a batch of sessions."""
        state = self.initial_state(batch_size)
        return NeuralMemoryState(
            weights={
                name: torch.zeros_like(value)
                for name, value in state.weights.items()
            },
            momentum=state.momentum,
        )

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
        has_pending_state = (
            state.pending_gradient is not None and state.pending_input_sum is not None
        )
        if has_pending_state != (state.pending_count > 0):
            raise ValueError(
                "pending gradient and input sum must exist exactly when a chunk is pending"
            )
        if (state.pending_gradient is None) != (state.pending_input_sum is None):
            raise ValueError(
                "state.pending_gradient and state.pending_input_sum must coexist"
            )

        collections = [
            ("weights", state.weights),
            ("momentum", state.momentum),
        ]
        if state.pending_gradient is not None:
            if set(state.pending_gradient) != parameter_names:
                raise ValueError(
                    "state.pending_gradient does not match the memory MLP parameters"
                )
            collections.append(("pending_gradient", state.pending_gradient))
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
        if state.pending_input_sum is not None:
            if state.pending_input_sum.shape != (batch_size, self.config.dim):
                raise ValueError(
                    "state.pending_input_sum must have shape "
                    f"{(batch_size, self.config.dim)}"
                )
            if state.pending_input_sum.dtype != torch.float32:
                raise ValueError("state.pending_input_sum must be float32")

    def _predict_memory(
        self,
        sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
        key: Float32[torch.Tensor, "D"],
    ) -> Float[torch.Tensor, "D"]:
        # Execute with the session's fast weights directly, avoiding module
        # parameter substitution. The template still supplies init/checkpoint weights.
        prediction = key
        for layer_index in range(self.config.depth):
            weight = sample_weights[f"layers.{layer_index}.weight"]
            prediction = F.linear(prediction, weight)
            if layer_index + 1 < self.config.depth:
                prediction = F.silu(prediction)
        return prediction

    def _memory_loss(
        self,
        sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
        chunk_keys: Float32[torch.Tensor, "C D"],
        chunk_values: Float32[torch.Tensor, "C D"],
        chunk_write_strength: Float32[torch.Tensor, "C"],
    ) -> Float[torch.Tensor, ""]:
        # Define the loss once rather than rebuilding annotated functions per write.
        # Each call still evaluates fresh inputs and the current session weights.
        C, _ = chunk_keys.shape

        # With one token, vmap adds overhead without any batching benefit.
        # Restore C afterward so the loss always receives C x D predictions.
        predictions: Float[torch.Tensor, "C D"]
        if C == 1:
            key = chunk_keys.squeeze(0)
            prediction = self._predict_memory(sample_weights, key)
            predictions = prediction.unsqueeze(0)
        else:
            predict_tokens = vmap(self._predict_memory, in_dims=(None, 0))
            predictions = predict_tokens(sample_weights, chunk_keys)

        per_token_loss: Float32[torch.Tensor, "C"] = F.mse_loss(
            predictions,
            chunk_values,
            reduction="none",
        ).mean(dim=-1)
        # L_chunk = sum_i theta_i * ||M_W(k_i) - v_i||^2 / D
        return (chunk_write_strength * per_token_loss).sum()

    def _chunk_gradient(
        self,
        weights: dict[str, Float32[torch.Tensor, "B D D"]],
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
        write_strength: Float32[torch.Tensor, "B C"],
    ) -> dict[str, Float32[torch.Tensor, "B D D"]]:
        B, _, _ = keys.shape
        # Online writes still require a local gradient during no-grad/inference
        # generation. Only training retains the higher-order graph used by the
        # outer delayed-answer objective.
        with torch.inference_mode(False), torch.enable_grad():
            timer = self.timing_observer
            tic = perf_counter() if timer is not None else 0.0
            if not self.training:
                weights = {
                    name: value.detach().clone()
                    for name, value in weights.items()
                }
                keys = keys.detach().clone()
                values = values.detach().clone()
                write_strength = write_strength.detach().clone()
            if timer is not None:
                timer("gradient_input_setup", perf_counter() - tic)

            # One gradient per independent session, taken from the aggregate
            # weighted loss of all C tokens at the shared chunk-start weights.
            tic = perf_counter() if timer is not None else 0.0
            if B > 1:
                # autograd.grad cannot run inside the session vmap transform.
                per_session_grad = grad(self._memory_loss)
                batched_grad = vmap(per_session_grad, in_dims=(0, 0, 0, 0))
            if timer is not None:
                timer("gradient_transform_setup", perf_counter() - tic)
            tic = perf_counter() if timer is not None else 0.0
            if B == 1:
                # Direct autograd avoids torch.func transform overhead for B=1;
                # CPU benchmarks halved gradient-call time without changing writes.
                sample_weights = {}
                for name, weight in weights.items():
                    sample_weight = weight.squeeze(0)
                    if not sample_weight.requires_grad:
                        # Inference or frozen initial weights need a local leaf.
                        # Never detach weights already connected to earlier writes.
                        sample_weight = sample_weight.detach().requires_grad_(True)
                    sample_weights[name] = sample_weight
                sample_keys = keys.squeeze(0)
                sample_values = values.squeeze(0)
                sample_write_strength = write_strength.squeeze(0)

                loss = self._memory_loss(
                    sample_weights,
                    sample_keys,
                    sample_values,
                    sample_write_strength,
                )
                # The outer answer loss must differentiate through this gradient
                # to teach earlier writes. Keep the same graph policy as func.grad.
                sample_gradients = torch.autograd.grad(
                    loss,
                    tuple(sample_weights.values()),
                    create_graph=True,
                )

                result = {}
                for name, gradient in zip(sample_weights, sample_gradients):
                    result[name] = gradient.unsqueeze(0)
            else:
                result = batched_grad(weights, keys, values, write_strength)
            if timer is not None:
                timer("gradient_execution", perf_counter() - tic)
            return result

    def _update(
        self,
        state: NeuralMemoryState,
        inputs: Float[torch.Tensor, "B C D"],
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
        write_mask: Bool[torch.Tensor, "B C"],
    ) -> NeuralMemoryState:
        B, C, _ = keys.shape
        timer = self.timing_observer
        tic = perf_counter() if timer is not None else 0.0
        pending_count = state.pending_count + C
        if pending_count > self.config.chunk_size:
            raise ValueError("_update cannot cross a memory chunk boundary")

        # The C tokens contribute one weighted loss and therefore one gradient
        # at the committed chunk-start weights, matching the reference code.
        write_strength: Float32[torch.Tensor, "B C"] = (
            self.write_strength_projection(inputs)
            .sigmoid()
            .reshape(B, C)
            .float()
        )
        # A masked token can still read memory, but contributes no write loss.
        write_strength = torch.where(write_mask, write_strength, 0.0)
        if timer is not None:
            timer("write_strength_and_mask", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        chunk_gradient: dict[str, Float32[torch.Tensor, "B D D"]] = (
            self._chunk_gradient(
                state.weights,
                keys,
                values,
                write_strength,
            )
        )
        if timer is not None:
            timer("gradient_call_total", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        input_sum: Float32[torch.Tensor, "B D"] = inputs.float().sum(dim=1)

        if state.pending_gradient is not None:
            chunk_gradient = {
                name: state.pending_gradient[name] + gradient
                for name, gradient in chunk_gradient.items()
            }
            input_sum = state.pending_input_sum + input_sum
        if timer is not None:
            timer("input_and_gradient_accumulation", perf_counter() - tic)

        if pending_count < self.config.chunk_size:
            tic = perf_counter() if timer is not None else 0.0
            if not self.training:
                chunk_gradient = {
                    name: value.detach() for name, value in chunk_gradient.items()
                }
                input_sum = input_sum.detach()
            result = replace(
                state,
                pending_gradient=chunk_gradient,
                pending_input_sum=input_sum,
                pending_count=pending_count,
            )
            if timer is not None:
                timer("pending_state_construction", perf_counter() - tic)
            return result

        # Summarize the completed chunk only to choose its forget/momentum controls;
        # the keys, values, and accumulated gradient determine what memory stores.
        tic = perf_counter() if timer is not None else 0.0
        chunk_input: Float[torch.Tensor, "B D"] = (
            input_sum / self.config.chunk_size
        ).to(self.forget_projection.weight.dtype)
        forget: Float32[torch.Tensor, "B 1 1"] = (
            self.forget_projection(chunk_input).sigmoid().reshape(B, 1, 1).float()
        )
        momentum_retention: Float32[torch.Tensor, "B 1 1"] = (
            self.momentum_projection(chunk_input).sigmoid().reshape(B, 1, 1).float()
        )
        if timer is not None:
            timer("forget_and_momentum_controls", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        next_momentum: dict[str, Float32[torch.Tensor, "B D D"]] = {
            name: momentum_retention * state.momentum[name] - gradient
            for name, gradient in chunk_gradient.items()
        }
        next_weights: dict[str, Float32[torch.Tensor, "B D D"]] = {
            name: (1.0 - forget) * weight + next_momentum[name]
            for name, weight in state.weights.items()
        }
        if timer is not None:
            timer("momentum_and_weight_update", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0

        if not self.training:
            next_weights = {
                name: value.detach() for name, value in next_weights.items()
            }
            next_momentum = {
                name: value.detach() for name, value in next_momentum.items()
            }

        next_state = NeuralMemoryState(
            weights=next_weights,
            momentum=next_momentum,
            query_conv_history=state.query_conv_history,
            key_conv_history=state.key_conv_history,
            value_conv_history=state.value_conv_history,
        )
        if timer is not None:
            timer("committed_state_construction", perf_counter() - tic)
        if self.trace_observer is not None:
            tic = perf_counter() if timer is not None else 0.0
            self.trace_observer(
                state,
                chunk_gradient,
                next_state,
                write_mask,
                write_strength,
            )
            if timer is not None:
                timer("trace_write_observer", perf_counter() - tic)
        return next_state

    def _read(
        self,
        weights: dict[str, Float32[torch.Tensor, "B D D"]],
        queries: Float[torch.Tensor, "B C D"],
    ) -> Float[torch.Tensor, "B C D"]:
        B, C, _ = queries.shape
        timer = self.timing_observer
        tic = perf_counter() if timer is not None else 0.0
        def read_one(
            sample_weights: dict[str, Float32[torch.Tensor, "D D"]],
            query: Float32[torch.Tensor, "D"],
        ) -> Float[torch.Tensor, "D"]:
            return self._predict_memory(sample_weights, query)

        # A single query can be read directly, avoiding token-vmap overhead.
        if C == 1:
            def per_session_read(sample_weights, sample_queries):
                query = sample_queries.squeeze(0)
                read = read_one(sample_weights, query)
                return read.unsqueeze(0)
        else:
            per_session_read = vmap(read_one, in_dims=(None, 0))

        # Session batching is independent of token batching: only use it when B > 1.
        if B == 1:
            batched_read = per_session_read
        else:
            batched_read = vmap(per_session_read, in_dims=(0, 0))
        if timer is not None:
            timer("read_transform_setup", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        if B == 1:
            # Remove B for the direct call and restore it on the returned reads.
            sample_weights = {}
            for name, weight in weights.items():
                sample_weights[name] = weight.squeeze(0)
            sample_queries = queries.squeeze(0).float()

            sample_reads = batched_read(sample_weights, sample_queries)
            result = sample_reads.unsqueeze(0)
        else:
            result = batched_read(weights, queries.float())
        if timer is not None:
            timer("read_execution", perf_counter() - tic)
        return result

    def _process_chunk(
        self,
        state: NeuralMemoryState,
        queries: Float[torch.Tensor, "B C D"],
        keys: Float32[torch.Tensor, "B C D"],
        values: Float32[torch.Tensor, "B C D"],
        inputs: Float[torch.Tensor, "B C D"],
        write_mask: Bool[torch.Tensor, "B C"],
        output_dtype: torch.dtype,
    ) -> tuple[Float[torch.Tensor, "B C D"], NeuralMemoryState]:
        _, C, _ = queries.shape
        # _update returns a new state, so this remains the pre-update snapshot.
        previous_weights = state.weights
        reaches_chunk_boundary = (
            state.pending_count + C == self.config.chunk_size
        )
        state = self._update(state, inputs, keys, values, write_mask)

        outputs: list[Float[torch.Tensor, "B C D"]] = []
        if reaches_chunk_boundary:
            # Earlier tokens read the previous completed chunk state. The token
            # at this chunk boundary can read the newly completed update.
            if C > 1:
                outputs.append(self._read(previous_weights, queries[:, :-1]))
            outputs.append(self._read(state.weights, queries[:, -1:]))
        else:
            outputs.append(self._read(previous_weights, queries))
        timer = self.timing_observer
        tic = perf_counter() if timer is not None else 0.0
        output = torch.cat(outputs, dim=1).to(output_dtype)
        if timer is not None:
            timer("per_chunk_output_assembly", perf_counter() - tic)
        return output, state

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inputs: Float[torch.Tensor, "B S D"],
        state: NeuralMemoryState | None = None,
        write_mask: Bool[torch.Tensor, "B S"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B S D"], NeuralMemoryState]:
        batch_size, sequence_length, dim = inputs.shape
        timer = self.timing_observer
        tic = perf_counter() if timer is not None else 0.0
        if dim != self.config.dim:
            raise ValueError(f"Expected dimension {self.config.dim}, received {dim}")
        if state is None:
            state = self.initial_state(batch_size)
        self._validate_state(state, batch_size)
        if write_mask is None:
            write_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=inputs.device,
            )
        if sequence_length == 0:
            return inputs.new_empty(batch_size, 0, dim), state
        if timer is not None:
            timer("state_and_mask_setup", perf_counter() - tic)

        tic = perf_counter() if timer is not None else 0.0
        queries: Float[torch.Tensor, "B S D"]
        query_history: Float[torch.Tensor, "B D S_history"]
        queries, query_history = self.query_conv(
            self.query_projection(inputs), state.query_conv_history
        )
        if timer is not None:
            timer("query_projection_and_conv", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        keys: Float[torch.Tensor, "B S D"]
        key_history: Float[torch.Tensor, "B D S_history"]
        keys, key_history = self.key_conv(
            self.key_projection(inputs), state.key_conv_history
        )
        if timer is not None:
            timer("key_projection_and_conv", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        values: Float[torch.Tensor, "B S D"]
        value_history: Float[torch.Tensor, "B D S_history"]
        values, value_history = self.value_conv(
            self.value_projection(inputs), state.value_conv_history
        )
        if timer is not None:
            timer("value_projection_and_conv", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        queries = F.normalize(self.activation(queries), dim=-1, eps=1e-6)
        keys = F.normalize(self.activation(keys), dim=-1, eps=1e-6)
        values = self.activation(values)
        if timer is not None:
            timer("activation_and_normalization", perf_counter() - tic)
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
                inputs[:, start:end],
                write_mask[:, start:end],
                inputs.dtype,
            )
            outputs.append(output)
            start = end

        # [B, S, D] -> [B, N, C, D]. N chunks are recurrent; the C tokens
        # within each chunk share its starting weights and are tensorized.
        full_length = ((sequence_length - start) // C) * C
        # Prefill may contain full chunks; token-by-token decoding usually does not.
        if full_length > 0:
            tic = perf_counter() if timer is not None else 0.0
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
            input_chunks: Float[torch.Tensor, "B N C D"] = inputs[
                :, start:end
            ].reshape(batch_size, N, C, dim)
            write_mask_chunks: Bool[torch.Tensor, "B N C"] = write_mask[
                :, start:end
            ].reshape(batch_size, N, C)
            if timer is not None:
                timer("chunk_views_and_casts", perf_counter() - tic)

            for chunk_index in range(N):
                tic = perf_counter() if timer is not None else 0.0
                output, state = self._process_chunk(
                    state,
                    query_chunks[:, chunk_index],
                    key_chunks[:, chunk_index],
                    value_chunks[:, chunk_index],
                    input_chunks[:, chunk_index],
                    write_mask_chunks[:, chunk_index],
                    inputs.dtype,
                )
                if timer is not None:
                    timer("token_loop_dispatch_total", perf_counter() - tic)
                outputs.append(output)
            start = end

        # A final partial chunk remains open in state for the next call.
        if start < sequence_length:
            output, state = self._process_chunk(
                state,
                queries[:, start:],
                keys[:, start:].float(),
                values[:, start:].float(),
                inputs[:, start:],
                write_mask[:, start:],
                inputs.dtype,
            )
            outputs.append(output)

        tic = perf_counter() if timer is not None else 0.0
        state = replace(
            state,
            query_conv_history=query_history,
            key_conv_history=key_history,
            value_conv_history=value_history,
        )
        if timer is not None:
            timer("conv_history_state_construction", perf_counter() - tic)
        tic = perf_counter() if timer is not None else 0.0
        result = torch.cat(outputs, dim=1)
        if timer is not None:
            timer("sequence_output_concatenation", perf_counter() - tic)
        return result, state
