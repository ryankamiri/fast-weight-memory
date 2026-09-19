import copy
import unittest

import torch

from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import NeuralMemory


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

    def test_split_calls_match_concatenated_call(self):
        full_memory = NeuralMemory(self.config).eval()
        split_memory = copy.deepcopy(full_memory).eval()

        with torch.no_grad():
            full_output, full_state = full_memory(self.inputs[:, :5])
            state = None
            pieces = []
            for start, end in ((0, 1), (1, 3), (3, 4), (4, 5)):
                output, state = split_memory(self.inputs[:, start:end], state)
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

        updated = memory._update(state, inputs, keys, values)
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
            torch.float32,
        )

        # W0 = 0 and the aggregated chunk update produces W_chunk = 1.5.
        # The first token reads W0; the boundary token reads W_chunk.
        expected_reads = torch.tensor([[[0.0], [1.5]]])
        token_update_reads = torch.tensor([[[0.5], [1.5]]])
        torch.testing.assert_close(output, expected_reads)
        self.assertFalse(torch.equal(output, token_update_reads))

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
