import unittest

import torch
from jaxtyping import TypeCheckError

from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.model import FWQwen3Model
from inference.prefill import prefill


class PrefillTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = FWQwen3Model(FWQwen3Config(
            vocab_size=40, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4,
            fast_weight_layers=[0], teacher_window_size=5,
            student_window_size=2, chunk_size=4, conv_kernel_size=3,
            attention_dropout=0.0,
        )).eval()
        # Non-identity filters exercise history carried across execution blocks.
        with torch.no_grad():
            mlp = self.model.layers[0].mlp
            mlp.teacher_conv.weight.normal_(std=0.2)
            mlp.student_conv.weight.normal_(std=0.2)
            mlp.W_proj.normal_(std=0.2)
            mlp.beta_proj.normal_(std=0.2)
        self.ids = torch.randint(0, 40, (2, 25))

    def assert_states_close(self, actual, expected):
        self.assertEqual(actual.tokens_seen, expected.tokens_seen)
        self.assertEqual(set(actual.mlp_states), set(expected.mlp_states))
        for idx, state in actual.mlp_states.items():
            for name, tensor in vars(state).items():
                torch.testing.assert_close(
                    tensor, getattr(expected.mlp_states[idx], name), atol=2e-6, rtol=2e-5,
                )
        for actual_layer, expected_layer in zip(
            actual.past_key_values.layers, expected.past_key_values.layers,
        ):
            self.assertEqual(actual_layer.get_seq_length(), expected.tokens_seen)
            self.assertEqual(actual_layer.keys.shape[-2], min(expected.tokens_seen, 4))
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_short_long_and_partial_prompts_match_single_call(self):
        for length in (1, 4, 5, 13, 20):
            expected = self.model(self.ids[:, :length], use_cache=True)
            for block_size in (None, 1, 3, 7, 32):
                with self.subTest(length=length, block_size=block_size):
                    blocks = []
                    def capture(module, args, kwargs, output):
                        blocks.append(output.last_hidden_state)
                        self.assertTrue(torch.is_inference_mode_enabled())
                    handle = self.model.register_forward_hook(capture, with_kwargs=True)
                    try:
                        actual = prefill(self.model, self.ids[:, :length], block_size)
                    finally:
                        handle.remove()
                    E = block_size if block_size is not None else 4
                    self.assertEqual(len(blocks), (length + E - 1) // E)
                    self.assertTrue(all(block.shape[1] <= E for block in blocks))
                    self.assertIs(actual.last_hidden_state, blocks[-1])
                    torch.testing.assert_close(
                        torch.cat(blocks, dim=1), expected.last_hidden_state, atol=2e-6, rtol=2e-5,
                    )
                    self.assert_states_close(actual.state, expected.state)
                    self.assertEqual(actual.state.mlp_states[0].pending_count, length % 4)

    @torch.inference_mode()
    def test_incoming_partial_chunk_and_decode_handoff(self):
        expected = self.model(self.ids[:, :19], use_cache=True)
        first = prefill(self.model, self.ids[:, :3])
        self.assertEqual(first.state.mlp_states[0].pending_count, 3)
        actual = prefill(self.model, self.ids[:, 3:19], execution_block_size=3, state=first.state)
        self.assertIs(actual.state.past_key_values, first.state.past_key_values)
        self.assert_states_close(actual.state, expected.state)
        for position in range(19, 25):
            actual = self.model(self.ids[:, position:position + 1], state=actual.state, use_cache=True)
            expected = self.model(self.ids[:, position:position + 1], state=expected.state, use_cache=True)
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-6, rtol=2e-5)
            self.assert_states_close(actual.state, expected.state)

    def test_inference_mode_and_validation(self):
        actual = prefill(self.model, self.ids[:, :3])
        self.assertFalse(actual.last_hidden_state.requires_grad)
        self.assertFalse(self.model.training)
        self.assertIsNone(self.model.embed_tokens.weight.grad)
        for invalid in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                prefill(self.model, self.ids, invalid)
        with self.assertRaises(TypeCheckError):
            prefill(self.model, self.ids, 1.5)
        for empty in (self.ids[:, :0], self.ids[:0]):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                prefill(self.model, empty)
        with self.assertRaises(TypeCheckError):
            prefill(self.model, self.ids[0])
        self.model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            prefill(self.model, self.ids)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
