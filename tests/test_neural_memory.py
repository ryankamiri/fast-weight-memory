import copy
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F

from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import MemoryMLPResult, NeuralMemory


def always_vmap_gradient(memory, weights, keys, values, write_strength):
    """Pre-fast-path implementation, retained as a numerical reference."""
    with torch.inference_mode(False), torch.enable_grad():
        if not memory.training:
            weights = {name: value.detach().clone() for name, value in weights.items()}
            keys, values, write_strength = (
                value.detach().clone() for value in (keys, values, write_strength)
            )

        def chunk_loss(sample_weights, chunk_keys, chunk_values, strength):
            def predict(key):
                return functional_call(memory.memory_mlp, sample_weights, (key,)).predicted_values

            predictions = vmap(predict)(chunk_keys)
            loss = F.mse_loss(predictions, chunk_values, reduction="none").mean(-1)
            return (strength * loss).sum()

        return vmap(grad(chunk_loss), in_dims=(0, 0, 0, 0))(
            weights, keys, values, write_strength
        )


def always_vmap_read(memory, weights, queries):
    def read_one(sample_weights, query):
        return functional_call(memory.memory_mlp, sample_weights, (query,)).predicted_values

    per_session_read = vmap(read_one, in_dims=(None, 0))
    return vmap(per_session_read, in_dims=(0, 0))(weights, queries.float())


def indexed_chunk_parts(chunks, dim=0):
    """Reproduce the old per-chunk indexing for split-once parity tests."""
    if dim != 1:
        raise ValueError("The memory loop splits only its chunk dimension")
    _, N, *remaining_dimensions = chunks.shape
    parts = []
    for chunk_index in range(N):
        parts.append(chunks[:, chunk_index])
    return tuple(parts)


class NeuralMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = NeuralMemoryConfig(
            dim=4,
            depth=2,
            conv_kernel_size=3,
            chunk_size=3,
            initial_forget=0.1,
            initial_momentum=0.8,
            initial_write_strength=0.2,
        )
        self.inputs = torch.randn(2, 6, 4)

    def _assert_states_close(self, actual, expected):
        for field_name, expected_value in vars(expected).items():
            actual_value = getattr(actual, field_name)
            if isinstance(expected_value, dict):
                self.assertEqual(actual_value.keys(), expected_value.keys())
                for name, value in expected_value.items():
                    torch.testing.assert_close(actual_value[name], value)
            elif isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                self.assertEqual(actual_value, expected_value)

    def test_split_once_matches_indexing_outputs_states_and_delayed_gradients(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            for frozen_initial_weights in (False, True):
                with self.subTest(B=B, C=C, frozen=frozen_initial_weights):
                    config = NeuralMemoryConfig(dim=4, chunk_size=C, conv_kernel_size=2)
                    memory = NeuralMemory(config).train()
                    if frozen_initial_weights:
                        memory.memory_mlp.requires_grad_(False)
                    reference = copy.deepcopy(memory)
                    inputs = torch.randn(B, 11, 4, requires_grad=True)
                    reference_inputs = inputs.detach().clone().requires_grad_(True)
                    write_mask = torch.ones(B, 11, dtype=torch.bool)
                    write_mask[:, 1::3] = False

                    output, state = memory(inputs, write_mask=write_mask)
                    with patch.object(torch.Tensor, "unbind", indexed_chunk_parts):
                        expected_output, expected_state = reference(
                            reference_inputs, write_mask=write_mask
                        )
                    torch.testing.assert_close(output, expected_output)
                    self._assert_states_close(state, expected_state)

                    output[:, -1].square().mean().backward()
                    expected_output[:, -1].square().mean().backward()
                    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                    self.assertGreater(inputs.grad[:, :2].abs().sum().item(), 0)
                    for (name, parameter), (_, expected_parameter) in zip(
                        memory.named_parameters(), reference.named_parameters()
                    ):
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, name)
                            torch.testing.assert_close(parameter.grad, expected_parameter.grad)
                        else:
                            self.assertIsNone(parameter.grad)
                            self.assertIsNone(expected_parameter.grad)

    def test_chunked_writes_match_autograd_and_tokenwise_execution(self):
        for B in (1, 2):
            for C in ((4, 8, 64) if B == 1 else (4, 8)):
                with self.subTest(B=B, C=C):
                    memory = NeuralMemory(NeuralMemoryConfig(
                        dim=4, depth=2, conv_kernel_size=3, chunk_size=C,
                    )).train()
                    memory.memory_mlp.requires_grad_(False)
                    reference = copy.deepcopy(memory)
                    reference._chunk_gradient = MethodType(always_vmap_gradient, reference)
                    S = 2 * C + 3  # Includes a trailing incomplete chunk.
                    inputs = torch.randn(B, S, 4, requires_grad=True)
                    reference_inputs = inputs.detach().clone().requires_grad_(True)
                    mask = torch.ones(B, S, dtype=torch.bool)
                    mask[:, 1::3] = False
                    output, state = memory(inputs, write_mask=mask)
                    expected_outputs = []
                    expected_state = None
                    # The independent autograd oracle receives one token per
                    # call. Pending gradients must still commit at C, not per call.
                    for position in range(S):
                        token_output, expected_state = reference(
                            reference_inputs[:, position:position + 1],
                            state=expected_state,
                            write_mask=mask[:, position:position + 1],
                        )
                        expected_outputs.append(token_output)
                    expected_output = torch.cat(expected_outputs, dim=1)
                    torch.testing.assert_close(output, expected_output)
                    self._assert_states_close(state, expected_state)
                    self.assertEqual(state.pending_count, 3)
                    output[:, -1].square().mean().backward()
                    expected_output[:, -1].square().mean().backward()
                    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                    self.assertGreater(inputs.grad[:, :C].abs().sum().item(), 0)
                    for (name, parameter), (_, expected) in zip(
                        memory.named_parameters(), reference.named_parameters()
                    ):
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, name)
                            torch.testing.assert_close(parameter.grad, expected.grad)
                        else:
                            self.assertIsNone(parameter.grad)

    def test_chunked_reads_do_not_use_future_writes(self):
        for C in (4, 8, 64):
            with self.subTest(C=C), torch.inference_mode():
                memory = NeuralMemory(NeuralMemoryConfig(
                    dim=4, depth=2, conv_kernel_size=3, chunk_size=C,
                )).eval()
                inputs = torch.randn(1, 2 * C + 3, 4)
                expected, _ = memory(inputs)
                changed = inputs.clone()
                # Change a token within the first chunk, before its commit.
                changed[:, C - 2] += 5
                actual, _ = memory(changed)
                torch.testing.assert_close(actual[:, :C - 2], expected[:, :C - 2])

    def test_split_once_matches_indexing_across_training_call_boundaries(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            with self.subTest(B=B, C=C):
                memory = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=C)).train()
                reference = copy.deepcopy(memory)
                inputs = torch.randn(B, 11, 4, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_(True)
                write_mask = torch.ones(B, 11, dtype=torch.bool)
                write_mask[:, 2::3] = False
                state = None
                expected_state = None
                outputs = []
                expected_outputs = []
                # C=3 carries an incomplete chunk into the second call, then
                # executes full chunks and retains another incomplete chunk.
                for start, end in ((0, 2), (2, 11)):
                    output, state = memory(
                        inputs[:, start:end], state, write_mask[:, start:end]
                    )
                    with patch.object(torch.Tensor, "unbind", indexed_chunk_parts):
                        expected_output, expected_state = reference(
                            reference_inputs[:, start:end],
                            expected_state,
                            write_mask[:, start:end],
                        )
                    outputs.append(output)
                    expected_outputs.append(expected_output)
                    self._assert_states_close(state, expected_state)

                output = torch.cat(outputs, dim=1)
                expected_output = torch.cat(expected_outputs, dim=1)
                torch.testing.assert_close(output, expected_output)
                output[:, -1].square().mean().backward()
                expected_output[:, -1].square().mean().backward()
                torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                for (name, parameter), (_, expected_parameter) in zip(
                    memory.named_parameters(), reference.named_parameters()
                ):
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_split_once_matches_indexing_in_inference_modes(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            for context in (torch.no_grad, torch.inference_mode):
                with self.subTest(B=B, C=C, context=context.__name__):
                    memory = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=C)).eval()
                    reference = copy.deepcopy(memory)
                    inputs = torch.randn(B, 11, 4)
                    write_mask = torch.ones(B, 11, dtype=torch.bool)
                    write_mask[:, 1::3] = False
                    with context():
                        output, state = memory(inputs, write_mask=write_mask)
                        with patch.object(torch.Tensor, "unbind", indexed_chunk_parts):
                            expected_output, expected_state = reference(
                                inputs, write_mask=write_mask
                            )
                    torch.testing.assert_close(output, expected_output)
                    self._assert_states_close(state, expected_state)

    def test_batched_writes_and_reads_skip_vmap(self):
        memory = NeuralMemory(self.config).train()
        D = self.config.dim
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            with self.subTest(B=B, C=C):
                weights = {
                    name: torch.randn(B, D, D, requires_grad=True)
                    for name in dict(memory.memory_mlp.named_parameters())
                }
                keys = torch.randn(B, C, D, requires_grad=True)
                values = torch.randn(B, C, D, requires_grad=True)
                strength = torch.rand(B, C, requires_grad=True)
                with patch("torch.func.vmap", wraps=vmap) as mapped:
                    actual = memory._chunk_gradient(weights, keys, values, strength)
                self.assertEqual(mapped.call_count, 0)
                reference = always_vmap_gradient(memory, weights, keys, values, strength)
                for name in weights:
                    self.assertEqual(actual[name].shape, (B, D, D))
                    torch.testing.assert_close(actual[name], reference[name])

                leaves = (*weights.values(), keys, values, strength)
                actual_outer = torch.autograd.grad(
                    sum(value.square().sum() for value in actual.values()), leaves
                )
                reference_outer = torch.autograd.grad(
                    sum(value.square().sum() for value in reference.values()), leaves
                )
                for observed, expected in zip(actual_outer, reference_outer):
                    torch.testing.assert_close(observed, expected)

                with patch("torch.func.vmap", wraps=vmap) as mapped:
                    actual_read = memory.memory_mlp(keys.float(), weights=weights).predicted_values
                self.assertEqual(mapped.call_count, 0)
                reference_read = always_vmap_read(memory, weights, keys)
                self.assertEqual(actual_read.shape, (B, C, D))
                torch.testing.assert_close(actual_read, reference_read)
                read_leaves = (*weights.values(), keys)
                actual_outer = torch.autograd.grad(actual_read.square().sum(), read_leaves)
                reference_outer = torch.autograd.grad(reference_read.square().sum(), read_leaves)
                for observed, expected in zip(actual_outer, reference_outer):
                    torch.testing.assert_close(observed, expected)

    def test_singleton_paths_preserve_recurrent_delayed_loss_gradients(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            with self.subTest(B=B, C=C):
                config = NeuralMemoryConfig(dim=4, chunk_size=C, conv_kernel_size=2)
                memory = NeuralMemory(config).train()
                reference = copy.deepcopy(memory)
                reference._chunk_gradient = MethodType(always_vmap_gradient, reference)
                inputs = torch.randn(B, 7, 4, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_(True)
                write_mask = torch.ones(B, 7, dtype=torch.bool)
                write_mask[:, 1] = False

                output, state = memory(inputs, write_mask=write_mask)
                expected_output, expected_state = reference(reference_inputs, write_mask=write_mask)
                torch.testing.assert_close(output, expected_output)
                self.assertEqual(state.pending_count, expected_state.pending_count)
                for collection in ("weights", "momentum", "pending_gradient"):
                    observed, expected = getattr(state, collection), getattr(expected_state, collection)
                    if expected is None:
                        self.assertIsNone(observed)
                    else:
                        for name in expected:
                            torch.testing.assert_close(observed[name], expected[name])
                for name in ("query_conv_history", "key_conv_history", "value_conv_history", "pending_input_sum"):
                    observed, expected = getattr(state, name), getattr(expected_state, name)
                    if expected is None:
                        self.assertIsNone(observed)
                    else:
                        torch.testing.assert_close(observed, expected)

                output[:, -1].square().mean().backward()
                expected_output[:, -1].square().mean().backward()
                torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                for (name, parameter), (_, expected_parameter) in zip(
                    memory.named_parameters(), reference.named_parameters()
                ):
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad)

    def test_explicit_writes_do_not_invoke_inner_autograd_or_vmap(self):
        memory = NeuralMemory(NeuralMemoryConfig(dim=4))
        for B in (1, 2):
            with self.subTest(B=B):
                weights = memory.initial_state(B).weights
                keys = torch.randn(B, 1, 4)
                values = torch.randn(B, 1, 4)
                strength = torch.rand(B, 1)
                with (
                    patch("torch.autograd.grad", side_effect=AssertionError("No inner autograd")),
                    patch("torch.func.grad", side_effect=AssertionError("No inner func.grad")),
                    patch("torch.func.vmap", side_effect=AssertionError("No write vmap")),
                ):
                    gradients = memory._chunk_gradient(weights, keys, values, strength)
                for gradient in gradients.values():
                    self.assertEqual(gradient.shape, (B, 4, 4))

    def test_manual_write_derivative_matches_autograd_with_tight_tolerance(self):
        generator = torch.Generator().manual_seed(304)
        # Equivalent FP32 operations can round differently. Bound the absolute
        # error by twice FP32 epsilon, rather than expecting bitwise equality.
        absolute_tolerance = 2 * torch.finfo(torch.float32).eps
        for D in (4, 128):
            for depth in (1, 2, 3, 4):
                memory = NeuralMemory(NeuralMemoryConfig(dim=D, depth=depth)).train()
                for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
                    weights = {
                        f"layers.{layer_index}.weight": (
                            torch.randn(B, D, D, generator=generator) / D**0.5
                        ).requires_grad_(True)
                        for layer_index in range(depth)
                    }
                    keys = F.normalize(torch.randn(B, C, D, generator=generator), dim=-1)
                    values = torch.randn(B, C, D, generator=generator)
                    strength = torch.rand(B, C, generator=generator)

                    for write_pattern in ("unmasked", "partially_masked", "fully_masked"):
                        with self.subTest(D=D, depth=depth, B=B, C=C, writes=write_pattern):
                            write_strength = strength.clone()
                            if write_pattern == "partially_masked":
                                write_strength[:, 0] = 0
                            elif write_pattern == "fully_masked":
                                write_strength.zero_()

                            actual = memory._chunk_gradient(weights, keys, values, write_strength)

                            # Independent reference: ordinary Linear + SiLU and
                            # autograd, with neither the manual derivative nor vmap.
                            predictions_by_session = []
                            for session_index in range(B):
                                predicted_values = keys[session_index]
                                for layer_index in range(depth):
                                    name = f"layers.{layer_index}.weight"
                                    predicted_values = F.linear(
                                        predicted_values, weights[name][session_index]
                                    )
                                    if layer_index + 1 < depth:
                                        predicted_values = F.silu(predicted_values)
                                predictions_by_session.append(predicted_values)
                            predictions = torch.stack(predictions_by_session)

                            # Same write objective: sum_t theta_t ||M(k_t)-v_t||^2 / D.
                            # Summing sessions preserves independent writes, not a B average.
                            per_token_loss = F.mse_loss(
                                predictions, values, reduction="none"
                            ).mean(dim=-1)
                            write_loss = (write_strength * per_token_loss).sum()
                            expected_gradients = torch.autograd.grad(
                                write_loss, tuple(weights.values())
                            )

                            for name, expected in zip(weights, expected_gradients):
                                self.assertEqual(actual[name].shape, (B, D, D))
                                torch.testing.assert_close(
                                    actual[name], expected, rtol=1e-6, atol=absolute_tolerance
                                )
                                # Require a tiny absolute difference as well, including
                                # zero gradients from fully masked writes.
                                max_error = (actual[name] - expected).abs().max().item()
                                self.assertLessEqual(max_error, absolute_tolerance, name)
                                if write_pattern == "fully_masked":
                                    self.assertTrue(torch.equal(actual[name], torch.zeros_like(expected)))

    def test_explicit_preserves_delayed_gradients_with_frozen_initial_weights(self):
        cases = ((depth, B, C) for depth in (1, 2, 3, 4) for B in (1, 2) for C in (1, 3))
        for depth, B, C in cases:
            with self.subTest(depth=depth, B=B, C=C):
                memory = NeuralMemory(NeuralMemoryConfig(dim=4, depth=depth, chunk_size=C)).train()
                memory.memory_mlp.requires_grad_(False)
                reference = copy.deepcopy(memory)
                reference._chunk_gradient = MethodType(always_vmap_gradient, reference)
                inputs = torch.randn(B, 7, 4, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_(True)
                initial_weights = memory.initial_state(B).weights
                self.assertTrue(all(not weight.requires_grad for weight in initial_weights.values()))

                outputs = []
                states = []
                for model, model_inputs in ((memory, inputs), (reference, reference_inputs)):
                    _, state = model(model_inputs[:, :3])
                    output, state = model(model_inputs[:, 3:], state=state)
                    output[:, -1].square().mean().backward()
                    outputs.append(output)
                    states.append(state)

                torch.testing.assert_close(outputs[0], outputs[1])
                torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                self.assertGreater(inputs.grad[:, :2].abs().sum().item(), 0)
                for name in states[0].weights:
                    torch.testing.assert_close(states[0].weights[name], states[1].weights[name])
                for (name, parameter), (_, expected) in zip(memory.named_parameters(), reference.named_parameters()):
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        torch.testing.assert_close(parameter.grad, expected.grad)
                    else:
                        self.assertIsNone(parameter.grad)

    def test_explicit_eval_writes_under_no_grad(self):
        memory = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=1)).eval()
        reference = copy.deepcopy(memory)
        reference._chunk_gradient = MethodType(always_vmap_gradient, reference)
        inputs = torch.randn(1, 7, 4)
        with torch.no_grad():
            initial_state = memory.initial_state(1)
            initial_requires_grad = {
                name: weight.requires_grad
                for name, weight in initial_state.weights.items()
            }
            output, state = memory(inputs, state=initial_state)
            expected_output, expected_state = reference(inputs)
        torch.testing.assert_close(output, expected_output)
        for name in state.weights:
            torch.testing.assert_close(state.weights[name], expected_state.weights[name])
            self.assertEqual(initial_state.weights[name].requires_grad, initial_requires_grad[name])

    def test_explicit_eval_gradients_do_not_retain_a_graph(self):
        for depth in (1, 2, 3, 4):
            for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
                with self.subTest(depth=depth, B=B, C=C):
                    memory = NeuralMemory(NeuralMemoryConfig(dim=4, depth=depth)).eval()
                    weights = memory.initial_state(B).weights
                    keys = torch.randn(B, C, 4, requires_grad=True)
                    values = torch.randn(B, C, 4, requires_grad=True)
                    strength = torch.rand(B, C, requires_grad=True)
                    expected = always_vmap_gradient(memory, weights, keys, values, strength)
                    # Eval must not retain an inner write graph even if its caller
                    # has gradient recording enabled.
                    actual = memory._chunk_gradient(weights, keys, values, strength)
                    for name in weights:
                        torch.testing.assert_close(actual[name], expected[name])
                        self.assertFalse(actual[name].requires_grad)
                        self.assertIsNone(actual[name].grad_fn)

    def test_explicit_gradients_support_configurable_memory_depth(self):
        D = 4
        for depth in (1, 2, 3, 4):
            for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
                with self.subTest(depth=depth, B=B, C=C):
                    memory = NeuralMemory(NeuralMemoryConfig(dim=D, depth=depth))
                    self.assertEqual(len(memory.memory_mlp.layers), depth)
                    # Online weights differ from the template: neither path may read it.
                    weights = {
                        name: 0.1 * torch.randn_like(weight, requires_grad=True)
                        for name, weight in memory.initial_state(B).weights.items()
                    }
                    keys = torch.randn(B, C, D, requires_grad=True)
                    values = torch.randn(B, C, D, requires_grad=True)
                    strength = torch.rand(B, C, requires_grad=True)

                    expected = always_vmap_gradient(memory, weights, keys, values, strength)
                    expected_reads = always_vmap_read(memory, weights, keys)
                    with patch.object(
                        memory.memory_mlp,
                        "forward",
                        wraps=memory.memory_mlp.forward,
                    ) as mlp_forward:
                        actual = memory._chunk_gradient(weights, keys, values, strength)
                        actual_reads = memory.memory_mlp(keys.float(), weights=weights).predicted_values
                    self.assertEqual(mlp_forward.call_count, 2)
                    write_call, read_call = mlp_forward.call_args_list
                    self.assertTrue(write_call.kwargs["return_intermediates"])
                    self.assertNotIn("return_intermediates", read_call.kwargs)
                    self.assertIsNotNone(write_call.kwargs["weights"])
                    self.assertIsNotNone(read_call.kwargs["weights"])
                    for call in (write_call, read_call):
                        self.assertEqual(call.args[0].shape, (B, C, D))
                        for weight in call.kwargs["weights"].values():
                            self.assertEqual(weight.shape, (B, D, D))
                    for name in weights:
                        torch.testing.assert_close(actual[name], expected[name])
                    torch.testing.assert_close(actual_reads, expected_reads)

                    leaves = (*weights.values(), keys, values, strength)
                    actual_outer = torch.autograd.grad(
                        sum(value.square().sum() for value in actual.values()), leaves
                    )
                    expected_outer = torch.autograd.grad(
                        sum(value.square().sum() for value in expected.values()), leaves
                    )
                    for observed, reference in zip(actual_outer, expected_outer):
                        torch.testing.assert_close(observed, reference)

                    read_leaves = (*weights.values(), keys)
                    actual_outer = torch.autograd.grad(actual_reads.square().sum(), read_leaves)
                    expected_outer = torch.autograd.grad(expected_reads.square().sum(), read_leaves)
                    for observed, reference in zip(actual_outer, expected_outer):
                        torch.testing.assert_close(observed, reference)

    def test_memory_mlp_forward_returns_optional_connected_intermediates(self):
        for depth in (1, 2, 3, 4):
            for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
                with self.subTest(depth=depth, B=B, C=C):
                    memory = NeuralMemory(NeuralMemoryConfig(dim=4, depth=depth))
                    weights = {
                        name: torch.randn_like(weight, requires_grad=True)
                        for name, weight in memory.initial_state(B).weights.items()
                    }
                    keys = torch.randn(B, C, 4, requires_grad=True)
                    result = memory.memory_mlp(
                        keys, weights=weights, return_intermediates=True
                    )
                    reads = memory.memory_mlp(keys, weights=weights)
                    self.assertIsInstance(result, MemoryMLPResult)
                    self.assertIsInstance(reads, MemoryMLPResult)
                    self.assertIsNone(reads.keys_by_layer)
                    self.assertIsNone(reads.values_before_silu)
                    torch.testing.assert_close(result.predicted_values, reads.predicted_values)
                    torch.testing.assert_close(result.predicted_values, always_vmap_read(memory, weights, keys))
                    self.assertIs(result.keys_by_layer[0], keys)
                    self.assertEqual(len(result.keys_by_layer), depth)
                    self.assertEqual(len(result.values_before_silu), depth)
                    self.assertIs(result.predicted_values, result.values_before_silu[-1])
                    for tensor in result.values_before_silu:
                        self.assertTrue(tensor.requires_grad)
                        self.assertIsNotNone(tensor.grad_fn)
                    # The returned intermediates remain usable in the outer graph.
                    loss = sum(tensor.square().mean() for tensor in result.values_before_silu)
                    gradients = torch.autograd.grad(loss, (keys, *weights.values()))
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_singleton_paths_preserve_inference_writes(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            with self.subTest(B=B, C=C):
                memory = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=C)).eval()
                reference = copy.deepcopy(memory)
                reference._chunk_gradient = MethodType(always_vmap_gradient, reference)
                inputs = torch.randn(B, 7, 4)
                with torch.inference_mode():
                    output, state = memory(inputs)
                    expected_output, expected_state = reference(inputs)
                torch.testing.assert_close(output, expected_output)
                for name in state.weights:
                    torch.testing.assert_close(state.weights[name], expected_state.weights[name])
                    torch.testing.assert_close(state.momentum[name], expected_state.momentum[name])
                    self.assertFalse(state.weights[name].requires_grad)

    def test_state_is_named_fp32_and_not_modified_in_place(self):
        memory = NeuralMemory(self.config)
        initial = memory.initial_state(batch_size=2)
        before = {name: value.clone() for name, value in initial.weights.items()}
        _, updated = memory(self.inputs[:, :2], initial)

        self.assertEqual(set(initial.weights), set(dict(memory.memory_mlp.named_parameters())))
        self.assertEqual(set(initial.weights), set(initial.momentum))
        for name in initial.weights:
            self.assertEqual(initial.weights[name].dtype, torch.float32)
            self.assertEqual(initial.momentum[name].dtype, torch.float32)
            torch.testing.assert_close(initial.weights[name], before[name])
            torch.testing.assert_close(updated.weights[name], initial.weights[name])
            self.assertIsNot(updated.pending_gradient[name], initial.weights[name])
        self.assertEqual(updated.pending_count, 2)
        self.assertIsNotNone(updated.pending_input_sum)

    def test_zero_state_is_distinct_from_learned_initial_state(self):
        memory = NeuralMemory(self.config)
        initial = memory.initial_state(batch_size=2)
        zeroed = memory.zero_state(batch_size=2)

        self.assertTrue(any(
            value.count_nonzero() > 0 for value in initial.weights.values()
        ))
        self.assertTrue(all(
            value.count_nonzero() == 0 for value in zeroed.weights.values()
        ))
        self.assertTrue(all(
            value.count_nonzero() == 0 for value in zeroed.momentum.values()
        ))
        self.assertEqual(set(zeroed.weights), set(initial.weights))

    def test_split_calls_match_concatenated_call(self):
        full_memory = NeuralMemory(self.config).eval()
        split_memory = copy.deepcopy(full_memory).eval()
        write_mask = torch.tensor(
            [
                [True, False, True, True, False],
                [True, True, False, True, True],
            ]
        )

        with torch.no_grad():
            full_output, full_state = full_memory(
                self.inputs[:, :5],
                write_mask=write_mask,
            )
            state = None
            pieces = []
            for start, end in ((0, 1), (1, 3), (3, 4), (4, 5)):
                output, state = split_memory(
                    self.inputs[:, start:end],
                    state,
                    write_mask[:, start:end],
                )
                pieces.append(output)

        torch.testing.assert_close(torch.cat(pieces, dim=1), full_output)
        for name in full_state.weights:
            torch.testing.assert_close(state.weights[name], full_state.weights[name])
            torch.testing.assert_close(state.momentum[name], full_state.momentum[name])
        torch.testing.assert_close(state.query_conv_history, full_state.query_conv_history)
        torch.testing.assert_close(state.key_conv_history, full_state.key_conv_history)
        torch.testing.assert_close(state.value_conv_history, full_state.value_conv_history)
        self.assertEqual(state.pending_count, full_state.pending_count)
        for name in full_state.pending_gradient:
            torch.testing.assert_close(
                state.pending_gradient[name],
                full_state.pending_gradient[name],
            )
        torch.testing.assert_close(
            state.pending_input_sum,
            full_state.pending_input_sum,
        )

    def test_one_step_matches_explicit_titans_update(self):
        config = NeuralMemoryConfig(
            dim=3,
            depth=1,
            conv_kernel_size=1,
            initial_forget=0.2,
            initial_momentum=0.3,
            initial_write_strength=0.4,
        )
        memory = NeuralMemory(config)
        with torch.no_grad():
            for projection in (
                memory.query_projection,
                memory.key_projection,
                memory.value_projection,
            ):
                projection.weight.copy_(torch.eye(3))
            for conv in (memory.query_conv, memory.key_conv, memory.value_conv):
                conv.weight.fill_(1)
            memory.memory_mlp.layers[0].weight.zero_()

        inputs = torch.tensor([[[1.0, -0.5, 0.25]]])
        initial = memory.initial_state(1)
        query = torch.nn.functional.normalize(torch.nn.functional.silu(inputs[0, 0]), dim=-1)
        key = query
        value = torch.nn.functional.silu(inputs[0, 0])
        weight = initial.weights["layers.0.weight"][0].detach().clone().requires_grad_(True)
        loss = torch.nn.functional.mse_loss(weight @ key, value)
        gradient, = torch.autograd.grad(loss, weight)
        expected_momentum = -config.initial_write_strength * gradient
        expected_weight = (
            (1.0 - config.initial_forget) * weight.detach() + expected_momentum
        )
        expected_output = expected_weight @ query

        output, state = memory(inputs, initial)
        torch.testing.assert_close(
            state.momentum["layers.0.weight"][0], expected_momentum
        )
        torch.testing.assert_close(
            state.weights["layers.0.weight"][0], expected_weight
        )
        torch.testing.assert_close(output[0, 0], expected_output)

    def test_chunk_aggregates_token_gradients_into_one_update(self):
        config = NeuralMemoryConfig(
            dim=1,
            depth=1,
            conv_kernel_size=1,
            chunk_size=2,
            initial_forget=0.2,
            initial_momentum=0.3,
            initial_write_strength=0.4,
        )
        memory = NeuralMemory(config)
        with torch.no_grad():
            memory.memory_mlp.layers[0].weight.zero_()

        state = memory.initial_state(1)
        keys = torch.tensor([[[1.0], [2.0]]])
        values = torch.tensor([[[0.5], [-0.25]]])
        inputs = torch.tensor([[[0.1], [-0.2]]])

        chunk_start = state.weights["layers.0.weight"][0].detach().clone()
        gradients = []
        for token_index in range(2):
            weight = chunk_start.clone().requires_grad_(True)
            prediction = weight @ keys[0, token_index]
            loss = torch.nn.functional.mse_loss(
                prediction, values[0, token_index]
            )
            gradient, = torch.autograd.grad(loss, weight)
            gradients.append(gradient)

        expected_momentum = -config.initial_write_strength * sum(gradients)
        expected_weight = (
            (1.0 - config.initial_forget) * chunk_start + expected_momentum
        )

        write_mask = torch.ones(1, 2, dtype=torch.bool)
        updated = memory._update(state, inputs, keys, values, write_mask)
        torch.testing.assert_close(
            updated.weights["layers.0.weight"][0], expected_weight
        )
        torch.testing.assert_close(
            updated.momentum["layers.0.weight"][0], expected_momentum
        )
        self.assertEqual(updated.pending_count, 0)
        self.assertIsNone(updated.pending_gradient)
        self.assertIsNone(updated.pending_input_sum)

    def test_chunk_reads_previous_weight_until_boundary(self):
        config = NeuralMemoryConfig(
            dim=1,
            depth=1,
            conv_kernel_size=1,
            chunk_size=2,
            initial_write_strength=0.25,
        )
        memory = NeuralMemory(config)
        with torch.no_grad():
            memory.memory_mlp.layers[0].weight.zero_()

        initial = memory.initial_state(1)
        queries = torch.ones(1, 2, 1)
        keys = torch.ones(1, 2, 1)
        values = torch.tensor([[[1.0], [2.0]]])
        inputs = torch.ones(1, 2, 1)

        output, _ = memory._process_chunk(
            initial,
            queries,
            keys,
            values,
            inputs,
            torch.ones(1, 2, dtype=torch.bool),
            torch.float32,
        )

        # W0 = 0 and the aggregated chunk update produces W_chunk = 1.5.
        # The first token reads W0; the boundary token reads W_chunk.
        expected_reads = torch.tensor([[[0.0], [1.5]]])
        token_update_reads = torch.tensor([[[0.5], [1.5]]])
        torch.testing.assert_close(output, expected_reads)
        self.assertFalse(torch.equal(output, token_update_reads))

    def test_write_mask_prevents_masked_token_from_updating_memory(self):
        config = NeuralMemoryConfig(
            dim=1,
            depth=1,
            conv_kernel_size=1,
            chunk_size=2,
            initial_write_strength=0.25,
        )
        memory = NeuralMemory(config)
        with torch.no_grad():
            memory.memory_mlp.layers[0].weight.zero_()

        initial = memory.initial_state(1)
        queries = torch.ones(1, 2, 1)
        keys = torch.ones(1, 2, 1)
        values = torch.tensor([[[1.0], [2.0]]])
        inputs = torch.ones(1, 2, 1)
        write_mask = torch.tensor([[True, False]])

        output, updated = memory._process_chunk(
            initial,
            queries,
            keys,
            values,
            inputs,
            write_mask,
            torch.float32,
        )

        # Only the first token contributes: -theta * grad = -0.25 * -2 = 0.5.
        expected_weight = torch.tensor([[0.5]])
        torch.testing.assert_close(
            updated.weights["layers.0.weight"][0], expected_weight
        )
        # The mask controls writes only; the boundary token still reads memory.
        torch.testing.assert_close(output, torch.tensor([[[0.0], [0.5]]]))

    def test_outer_loss_backpropagates_through_online_writes(self):
        memory = NeuralMemory(self.config).train()
        output, state = memory(self.inputs)
        (output[:, -1].square().mean()).backward()

        for name, parameter in memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(any(value.grad_fn is not None for value in state.weights.values()))

    def test_inference_mode_still_performs_online_writes(self):
        memory = NeuralMemory(self.config).eval()
        with torch.inference_mode():
            output, state = memory(self.inputs)
        self.assertEqual(output.shape, self.inputs.shape)
        self.assertTrue(all(not value.requires_grad for value in state.weights.values()))
        self.assertTrue(any(value.abs().sum() > 0 for value in state.momentum.values()))

    def test_state_validation(self):
        memory = NeuralMemory(self.config)
        state = memory.initial_state(2)
        state.weights.pop(next(iter(state.weights)))
        with self.assertRaises(ValueError):
            memory(self.inputs, state)

if __name__ == "__main__":
    unittest.main()
