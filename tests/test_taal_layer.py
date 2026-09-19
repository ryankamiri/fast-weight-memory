import unittest

import torch
from torch import nn

from architectures.taal.configuration import TaalLayerConfig
from architectures.taal.layer import TaalLayer
from architectures.titans.configuration import NeuralMemoryConfig


class PassthroughMemory(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.inputs = None
        self.write_mask = None

    def forward(self, inputs, state=None, write_mask=None):
        self.inputs = inputs
        self.write_mask = write_mask
        return inputs, self.state if state is None else state


class TaalLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = TaalLayerConfig(
            model_dim=4,
            memory=NeuralMemoryConfig(
                dim=2,
                depth=2,
                conv_kernel_size=1,
                chunk_size=1,
            ),
            num_persistent_tokens=2,
        )

    def test_zero_gate_preserves_host_states_while_updating_memory(self):
        layer = TaalLayer(self.config)
        hidden_states = torch.randn(2, 3, 4)

        output, state = layer(hidden_states)

        torch.testing.assert_close(output, hidden_states, atol=0, rtol=0)
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(any(value.abs().sum() > 0 for value in state.momentum.values()))

    def test_persistent_outputs_are_removed_before_residual_addition(self):
        config = TaalLayerConfig(
            model_dim=2,
            memory=NeuralMemoryConfig(
                dim=2,
                depth=1,
                conv_kernel_size=1,
                chunk_size=1,
            ),
            num_persistent_tokens=2,
        )
        layer = TaalLayer(config)
        state = layer.neural_memory.initial_state(batch_size=1)
        passthrough = PassthroughMemory(state)
        layer.neural_memory = passthrough
        with torch.no_grad():
            layer.memory_projection_in.weight.copy_(torch.eye(2))
            layer.memory_projection_out.weight.copy_(torch.eye(2))
            layer.persistent_tokens.copy_(torch.tensor([[7.0, 8.0], [9.0, 10.0]]))
            layer.residual_gate.copy_(torch.atanh(torch.tensor(0.5)))

        hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        write_mask = torch.tensor([[True, False]])
        output, returned_state = layer(
            hidden_states,
            state=state,
            write_mask=write_mask,
        )

        expected_memory_inputs = torch.tensor(
            [[[7.0, 8.0], [9.0, 10.0], [1.0, 2.0], [3.0, 4.0]]]
        )
        torch.testing.assert_close(passthrough.inputs, expected_memory_inputs)
        torch.testing.assert_close(output, 1.5 * hidden_states)
        torch.testing.assert_close(
            passthrough.write_mask,
            torch.tensor([[True, True, True, False]]),
        )
        self.assertIs(returned_state, state)

    def test_passes_existing_state_to_neural_memory(self):
        layer = TaalLayer(self.config)
        hidden_states = torch.randn(1, 2, 4)
        _, first_state = layer(hidden_states)
        _, second_state = layer(hidden_states, state=first_state)

        self.assertIsNot(second_state, first_state)
        self.assertTrue(
            any(
                not torch.equal(second_state.weights[name], first_state.weights[name])
                for name in first_state.weights
            )
        )

    def test_configuration_validation(self):
        memory = NeuralMemoryConfig(dim=2)
        with self.assertRaises(ValueError):
            TaalLayerConfig(model_dim=0, memory=memory)
        with self.assertRaises(ValueError):
            TaalLayerConfig(model_dim=2, memory=memory, num_persistent_tokens=0)
        with self.assertRaises(ValueError):
            TaalLayerConfig(model_dim=2, memory=memory, persistent_init_std=0)


if __name__ == "__main__":
    unittest.main()
