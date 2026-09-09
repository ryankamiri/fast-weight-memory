import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from architectures.qwen.decoder import FWQwen3DecoderLayer
from architectures.qwen.attention import sdpa_attention_forward


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = Qwen3Config(
            hidden_size=32, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.0,
        )
        self.config._attn_implementation = "sdpa"
        self.rope = Qwen3RotaryEmbedding(self.config)
        self.x = torch.randn(2, 9, 32)

    def fast_layer(self):
        return FWQwen3DecoderLayer(
            self.config, 0, is_fast_weight_layer=True,
            chunk_size=4, conv_kernel_size=3,
        ).eval()

    def inputs(self, start=0, end=9, student_window=2):
        positions = torch.arange(start, end)
        distance = positions[:, None] - torch.arange(end)[None, :]
        return dict(
            position_embeddings=self.rope(self.x[:, start:end], positions[None]),
            cache_position=positions,
            teacher_attention_mask=((distance >= 0) & (distance < 5))[None, None],
            student_attention_mask=((distance >= 0) & (distance < student_window))[None, None],
        )

    def test_normal_matches_qwen_and_checkpoint_names(self):
        base = Qwen3DecoderLayer(self.config, 0).eval()
        layer = FWQwen3DecoderLayer(self.config, 0).eval()
        layer.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(set(layer.state_dict()), set(base.state_dict()))
        kwargs = self.inputs()
        kwargs.pop("student_attention_mask")
        actual = layer(self.x, **kwargs)
        kwargs["attention_mask"] = kwargs.pop("teacher_attention_mask")
        torch.testing.assert_close(actual, base(self.x, **kwargs), rtol=0, atol=0)

    def test_no_reads_skips_student_attention_and_writes(self):
        layer = self.fast_layer()
        kwargs = self.inputs()
        (teacher, _), _ = layer.self_attn(layer.input_layernorm(self.x), **kwargs)
        residual = self.x + teacher
        features = layer.post_attention_layernorm(residual)
        expected = residual + layer.mlp.down_proj(
            layer.mlp.act_fn(layer.mlp.gate_proj(features)) * layer.mlp.up_proj(features)
        )
        weights = {name: value.clone() for name, value in layer.state_dict().items()}
        layer.mlp.fast_weight_read_scale = 0.0
        kwargs.pop("student_attention_mask")
        with patch("architectures.qwen.attention.sdpa_attention_forward", wraps=sdpa_attention_forward) as attention, \
             patch.object(layer.mlp, "_convolve", side_effect=AssertionError("FW convolution ran")):
            actual, state = layer(self.x, **kwargs)
        self.assertEqual(attention.call_count, 1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state.W_fast, torch.zeros_like(state.W_fast))
        self.assertEqual(state.pending_count, 0)
        for name, value in layer.state_dict().items():
            torch.testing.assert_close(value, weights[name])
        layer.mlp.fast_weight_read_scale = 0.5
        with patch("architectures.qwen.attention.sdpa_attention_forward", wraps=sdpa_attention_forward) as attention:
            layer(self.x, **self.inputs())
        self.assertEqual(attention.call_count, 2)

    def test_dual_residuals_match_manual_composition_and_gradients(self):
        layer = self.fast_layer()
        # Exercise active writes rather than the zero-write initialization.
        with torch.no_grad():
            layer.mlp.student_conv.weight[..., -1] = 1
        kwargs = self.inputs()
        (teacher, student), _ = layer.self_attn(layer.input_layernorm(self.x), **kwargs)
        teacher_residual = self.x + teacher
        mlp_output, expected_state = layer.mlp(
            layer.post_attention_layernorm(teacher_residual),
            layer.post_attention_layernorm(self.x + student),
        )
        actual, state = layer(self.x, **kwargs)
        torch.testing.assert_close(actual, teacher_residual + mlp_output)
        torch.testing.assert_close(state.W_fast, expected_state.W_fast)
        self.assertGreater(state.W_fast.abs().sum().item(), 0)
        actual.square().mean().backward()
        for name, param in layer.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_equal_windows_match_qwen(self):
        layer = self.fast_layer()
        base = Qwen3DecoderLayer(self.config, 0).eval()
        base.load_state_dict({k: v for k, v in layer.state_dict().items() if k in base.state_dict()})
        kwargs = self.inputs(student_window=5)
        actual, state = layer(self.x, **kwargs)
        kwargs.pop("student_attention_mask")
        kwargs["attention_mask"] = kwargs.pop("teacher_attention_mask")
        torch.testing.assert_close(actual, base(self.x, **kwargs))
        torch.testing.assert_close(state.W_fast, torch.zeros_like(state.W_fast))

    def test_prefill_and_decode_match_full_sequence(self):
        layer = self.fast_layer()
        # Non-identity filters ensure convolution state also crosses calls.
        with torch.no_grad():
            layer.mlp.teacher_conv.weight.normal_(std=0.2)
            layer.mlp.student_conv.weight.normal_(std=0.2)
            expected, expected_state = layer(self.x, **self.inputs())
            cache = DynamicCache(config=self.config)
            state = None
            outputs = []
            for start, end in ((0, 3), (3, 7), (7, 8), (8, 9)):
                output, state = layer(
                    self.x[:, start:end], **self.inputs(start, end),
                    past_key_values=cache, use_cache=True, state=state,
                )
                outputs.append(output)
                self.assertEqual(cache.get_seq_length(), end)
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=2e-6, rtol=2e-5)
            for name in vars(state):
                torch.testing.assert_close(getattr(state, name), getattr(expected_state, name), atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
