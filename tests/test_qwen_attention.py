import unittest

import torch
from jaxtyping import TypeCheckError
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding

from architectures.qwen.attention import FWQwen3Attention


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.config = Qwen3Config(hidden_size=32, intermediate_size=48,
                                  num_hidden_layers=1, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8,
                                  attention_dropout=0.0)
        self.config._attn_implementation = "eager"
        self.base = Qwen3Attention(self.config, 0).eval()
        self.rope = Qwen3RotaryEmbedding(self.config)
        self.x = torch.randn(2, 9, 32)

    def wrapped(self):
        wrapped = FWQwen3Attention(
            self.config, 0, is_fast_weight_layer=True,
        )
        wrapped.load_state_dict(self.base.state_dict())
        return wrapped.eval()

    def embeddings(self, x, start=0):
        return self.rope(x, torch.arange(start, start + x.shape[1])[None])

    def masks(self, start=0, end=9, teacher=5, student=3):
        distance = torch.arange(start, end)[:, None] - torch.arange(end)[None, :]
        return tuple(((distance >= 0) & (distance < window))[None, None]
                     for window in (teacher, student))

    def run_dual(self, wrapped, x, start=0, teacher=5, student=3, **kwargs):
        teacher, student = self.masks(start, start + x.shape[1], teacher, student)
        return wrapped(x, self.embeddings(x, start), teacher_attention_mask=teacher,
                       student_attention_mask=student, **kwargs)

    def reference(self, x, window, padding=None):
        # Independent explicit additive mask, exercised through original eager GQA.
        mask = torch.full((x.shape[0], 1, x.shape[1], x.shape[1]), float("-inf"))
        for i in range(x.shape[1]):
            mask[:, :, i, max(0, i - window + 1):i + 1] = 0
        if padding is not None:
            mask.masked_fill_(~padding[:, None, None, :].bool(), float("-inf"))
        return self.base(x, self.embeddings(x), mask)[0]

    def test_normal_fallback_and_checkpoint_loading(self):
        wrapped = FWQwen3Attention(self.config, 0).eval()
        wrapped.load_state_dict(self.base.state_dict())
        self.assertEqual(set(wrapped.state_dict()), set(self.base.state_dict()))
        mask = torch.ones(9, 9).tril().log()[None, None]
        expected = self.base(self.x, self.embeddings(self.x), mask)
        actual = wrapped(self.x, self.embeddings(self.x), mask)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_two_windows_and_gradients(self):
        x = self.x.clone().requires_grad_()
        wrapped = self.wrapped()
        (teacher, student), weights = self.run_dual(wrapped, x)
        self.assertIsNone(weights)
        expected_t = self.reference(x, 5)
        expected_s = self.reference(x, 3)
        torch.testing.assert_close(teacher, expected_t)
        torch.testing.assert_close(student, expected_s)
        actual_grad = torch.autograd.grad(teacher.square().sum() + student.square().sum(), (x, *wrapped.parameters()))
        expected_grad = torch.autograd.grad(expected_t.square().sum() + expected_s.square().sum(), (x, *self.base.parameters()))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_equal_windows(self):
        (teacher, student), _ = self.run_dual(self.wrapped(), self.x, teacher=4, student=4)
        torch.testing.assert_close(teacher, student, rtol=0, atol=0)

    def test_padding_and_4d_masks(self):
        padding = torch.ones(2, 9, dtype=torch.bool)
        padding[0, 2] = False
        expected = self.reference(self.x, 5, padding)
        masks = tuple(mask & padding[:, None, None, :] for mask in self.masks())
        for teacher_mask, student_mask in (masks, tuple(torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf) for mask in masks)):
            (teacher, _), _ = self.wrapped()(self.x, self.embeddings(self.x), teacher_mask,
                                            student_attention_mask=student_mask)
            torch.testing.assert_close(teacher, expected)

    def test_cached_blocks_and_decode(self):
        wrapped = self.wrapped()
        full, _ = self.run_dual(wrapped, self.x)
        cache = DynamicCache()
        pieces = [[], []]
        for start, end in ((0, 4), (4, 8), (8, 9)):
            x = self.x[:, start:end]
            outputs, _ = self.run_dual(wrapped, x, start, past_key_values=cache,
                                      cache_position=torch.arange(start, end))
            self.assertEqual(cache.get_seq_length(), end)
            for parts, output in zip(pieces, outputs):
                parts.append(output)
        for expected, parts in zip(full, pieces):
            torch.testing.assert_close(torch.cat(parts, dim=1), expected)

    def test_future_tokens_do_not_affect_past(self):
        wrapped = self.wrapped()
        original, _ = self.run_dual(wrapped, self.x)
        changed = self.x.clone()
        changed[:, 5:] += 100
        outputs, _ = self.run_dual(wrapped, changed)
        for a, b in zip(original, outputs):
            torch.testing.assert_close(a[:, :5], b[:, :5])

    def test_runtime_shape_checks(self):
        wrapped = self.wrapped()
        teacher_mask, student_mask = self.masks()
        with self.assertRaises(TypeCheckError):
            wrapped(self.x[0], self.embeddings(self.x), teacher_mask,
                    student_attention_mask=student_mask)
        with self.assertRaises(TypeCheckError):
            wrapped(self.x, self.embeddings(self.x), teacher_mask.expand(3, -1, -1, -1),
                    student_attention_mask=student_mask)
        with self.assertRaises(TypeCheckError):
            wrapped(self.x, self.embeddings(self.x), teacher_mask,
                    student_attention_mask=student_mask[..., :-1])


if __name__ == "__main__":
    unittest.main()
