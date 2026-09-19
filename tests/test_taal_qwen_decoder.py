import unittest

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from architectures.taal.configuration import TaalLayerConfig
from architectures.taal.qwen.decoder import TaalQwen3DecoderLayer
from architectures.titans.configuration import NeuralMemoryConfig


class TaalQwen3DecoderLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.qwen_config = Qwen3Config(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        self.qwen_config._attn_implementation = "sdpa"
        self.taal_config = TaalLayerConfig(
            model_dim=16,
            memory=NeuralMemoryConfig(
                dim=8,
                depth=2,
                conv_kernel_size=1,
                chunk_size=1,
            ),
            num_persistent_tokens=2,
        )

    def test_zero_gate_matches_standard_qwen_decoder_exactly(self):
        base = Qwen3DecoderLayer(self.qwen_config, layer_idx=0).eval()
        layer = TaalQwen3DecoderLayer(
            self.qwen_config,
            layer_idx=0,
            taal_config=self.taal_config,
        ).eval()
        missing, unexpected = layer.load_state_dict(
            base.state_dict(),
            strict=False,
        )
        self.assertTrue(all(name.startswith("taal.") for name in missing))
        self.assertEqual(unexpected, [])

        hidden_states = torch.randn(2, 3, 16)
        position_embeddings = (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        )
        with torch.no_grad():
            expected = base(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            actual, memory_state = layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(memory_state.weights["layers.0.weight"].shape, (2, 8, 8))

    def test_reuses_memory_state_and_forwards_write_mask(self):
        layer = TaalQwen3DecoderLayer(
            self.qwen_config,
            layer_idx=0,
            taal_config=self.taal_config,
        ).eval()
        hidden_states = torch.randn(1, 2, 16)
        position_embeddings = (
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 4),
        )
        write_mask = torch.tensor([[True, False]])

        with torch.no_grad():
            _, first_state = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                write_mask=write_mask,
            )
            _, second_state = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                memory_state=first_state,
                write_mask=write_mask,
            )

        self.assertIsNot(second_state, first_state)
        self.assertEqual(second_state.weights["layers.0.weight"].shape, (1, 8, 8))

    def test_rejects_mismatched_model_dimension(self):
        taal_config = TaalLayerConfig(
            model_dim=8,
            memory=NeuralMemoryConfig(dim=4),
        )
        with self.assertRaisesRegex(ValueError, "Qwen hidden size"):
            TaalQwen3DecoderLayer(
                self.qwen_config,
                layer_idx=0,
                taal_config=taal_config,
            )


if __name__ == "__main__":
    unittest.main()
