import copy
import unittest

import torch
from jaxtyping import TypeCheckError
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from architectures.qwen.mlp import FWQwen3MLP
from architectures.states.mlp_state import FWMLPState


class MLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config = Qwen3Config(hidden_size=8, intermediate_size=12)
        self.teacher = torch.randn(2, 7, 8)
        self.student = torch.randn(2, 7, 8)

    def model(self, **kwargs):
        return FWQwen3MLP(self.config, is_fast_weight_layer=True, chunk_size=3, **kwargs)

    def test_normal_qwen_parity(self):
        base = Qwen3MLP(self.config)
        model = FWQwen3MLP(self.config)
        model.load_state_dict(base.state_dict(), strict=True)
        self.assertIs(model.W_base, model.down_proj.weight)
        torch.testing.assert_close(model(self.teacher), base(self.teacher), rtol=0, atol=0)
        self.assertFalse(hasattr(model, "W_fast"))

    def test_independent_chunk_recurrence(self):
        model = self.model(use_conv=False)
        with torch.no_grad():
            model.W_proj.normal_()
            model.beta_proj.normal_()
        initial = torch.randn(2, 8, 12) * 0.01
        initial_copy = initial.clone()
        expected_state = initial.clone()
        teacher = model.act_fn(model.gate_proj(self.teacher)) * model.up_proj(self.teacher)
        student = model.act_fn(model.gate_proj(self.student)) * model.up_proj(self.student)
        beta = (self.student @ model.beta_proj).sigmoid()
        expected_outputs = []
        for start in (0, 3, 6):
            end = min(start + 3, 7)
            effective = model.W_base[None] + expected_state
            expected_outputs.append(torch.bmm(teacher[:, start:end], effective.transpose(1, 2)))
            if end - start == 3:
                for t in range(start, end):
                    correction = (teacher[:, t] - student[:, t]) @ model.W_base.T @ model.W_proj
                    key = F.normalize(student[:, t], dim=-1, eps=1e-6)
                    expected_state = expected_state + model.lr * beta[:, t, None, None] * correction[:, :, None] * key[:, None, :]
        actual, state = model(self.teacher, self.student, state=FWMLPState(initial))
        torch.testing.assert_close(actual, torch.cat(expected_outputs, dim=1))
        torch.testing.assert_close(state.W_fast, expected_state)
        torch.testing.assert_close(initial, initial_copy)
        self.assertEqual(state.pending_count, 1)

    def test_split_calls_match_prefill_with_nontrivial_convs(self):
        full_model = self.model()
        self.assertIsNot(full_model.teacher_conv.weight, full_model.student_conv.weight)
        with torch.no_grad():
            full_model.teacher_conv.weight.normal_()
            full_model.student_conv.weight.normal_()
        split_model = copy.deepcopy(full_model)
        full, full_state = full_model(self.teacher, self.student)
        pieces = []
        split_state = None
        for start, end in ((0, 2), (2, 4), (4, 5), (5, 7)):
            output, split_state = split_model(self.teacher[:, start:end], self.student[:, start:end], state=split_state)
            pieces.append(output)
        torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-5)
        for name in ("W_fast", "pending_r", "pending_k", "teacher_conv_state", "student_conv_state"):
            torch.testing.assert_close(getattr(full_state, name), getattr(split_state, name), atol=1e-6, rtol=1e-5)

    def test_read_uses_raw_teacher_but_writes_use_convolution(self):
        model = self.model()
        teacher = self.teacher[:, :3]
        student = self.student[:, :3]
        initial = torch.randn(2, 8, 12) * 0.01
        z_teacher = model.act_fn(model.gate_proj(teacher)) * model.up_proj(teacher)
        expected = model.down_proj(z_teacher) + torch.bmm(z_teacher, initial.transpose(1, 2))
        baseline, baseline_state = model(teacher, student, state=FWMLPState(initial))
        with torch.no_grad():
            model.teacher_conv.weight.mul_(2)
        actual, changed_state = model(teacher, student, state=FWMLPState(initial))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, baseline)
        self.assertFalse(torch.allclose(changed_state.W_fast, baseline_state.W_fast))

    def test_zero_default_equal_streams_and_reset(self):
        model = self.model()
        result, state = model(self.teacher, self.teacher)
        expected = model.down_proj(model.act_fn(model.gate_proj(self.teacher)) * model.up_proj(self.teacher))
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(state.W_fast, torch.zeros_like(state.W_fast))
        fresh, _ = model(self.teacher, self.teacher, state=None)
        torch.testing.assert_close(fresh, result)
        self.assertNotIn("W_fast", model.state_dict())
        self.assertNotIn("W_base", model.state_dict())

    def test_chunk_update_is_delayed(self):
        model = self.model(use_conv=False)
        first, state = model(self.teacher[:, :2], self.student[:, :2])
        torch.testing.assert_close(state.W_fast, torch.zeros_like(state.W_fast))
        _, next_state = model(self.teacher[:, 2:3], self.student[:, 2:3], state=state)
        self.assertGreater(next_state.W_fast.abs().sum().item(), 0)
        self.assertEqual(next_state.pending_count, 0)
        self.assertEqual(state.pending_count, 2)
        torch.testing.assert_close(state.W_fast, torch.zeros_like(state.W_fast))
        expected = model.down_proj(model.act_fn(model.gate_proj(self.teacher[:, :2])) * model.up_proj(self.teacher[:, :2]))
        torch.testing.assert_close(first, expected)

    def test_gradients_and_detach(self):
        model = self.model()
        teacher = self.teacher.clone().requires_grad_()
        student = self.student.clone().requires_grad_()
        output, state = model(teacher, student)
        output.square().sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)
        detached = state.detach()
        self.assertIsNone(detached.W_fast.grad_fn)
        self.assertIsNone(detached.pending_r.grad_fn)
        self.assertIsNotNone(state.W_fast.grad_fn)

    def test_validation(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model(self.teacher)
        with self.assertRaises(RuntimeError):
            model(self.teacher, self.student, state=FWMLPState(torch.zeros(2, 8, 13)))

    def test_pending_count_is_computed(self):
        state = FWMLPState(torch.zeros(2, 8, 12))
        self.assertEqual(state.pending_count, 0)
        state.pending_r = torch.zeros(2, 2, 8)
        state.pending_k = torch.zeros(2, 2, 12)
        self.assertEqual(state.pending_count, 2)
        with self.assertRaises(AttributeError):
            state.pending_count = 1
        state.pending_k = torch.zeros(2, 1, 12)
        with self.assertRaises(TypeCheckError):
            self.model()(self.teacher, self.student, state=state)

    def test_state_shape_annotations(self):
        with self.assertRaises(TypeCheckError):
            FWMLPState(torch.zeros(2, 8, 12), pending_r=torch.zeros(3, 1, 8))
        with self.assertRaises(TypeCheckError):
            FWMLPState(torch.zeros(2, 8, 12), student_conv_state=torch.zeros(2, 13, 4))
        with self.assertRaises(TypeCheckError):
            FWMLPState(torch.zeros(2, 8, 12, dtype=torch.int64))

    def test_interleaved_sessions(self):
        model = self.model()
        expected, _ = model(self.teacher, self.student)
        first, state = model(self.teacher[:, :2], self.student[:, :2])
        before = {name: tensor.clone() for name, tensor in vars(state).items() if tensor is not None}
        model(self.teacher * 2, self.student * 3)
        second, _ = model(self.teacher[:, 2:], self.student[:, 2:], state=state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
        for name, tensor in before.items():
            torch.testing.assert_close(getattr(state, name), tensor)

    def test_state_uses_fp32_master_weights(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.model(use_conv=False).to(dtype=dtype)
                teacher = self.teacher.to(dtype)
                student = self.student.to(dtype)
                _, state = model(teacher[:, :2], student[:, :2])
                self.assertEqual(state.W_fast.dtype, torch.float32)
                self.assertEqual(state.pending_r.dtype, torch.float32)
                self.assertEqual(state.pending_k.dtype, torch.float32)
                _, state = model(teacher[:, 2:3], student[:, 2:3], state=state)
                self.assertEqual(state.W_fast.dtype, torch.float32)
                self.assertTrue(torch.isfinite(state.W_fast).all())
                # Overrides already satisfy the FP32 state contract.
                initial = FWMLPState(torch.zeros(2, 8, 12, dtype=torch.float32))
                _, state = model(teacher[:, :3], student[:, :3], state=initial)
                self.assertEqual(state.W_fast.dtype, torch.float32)
                self.assertEqual(initial.W_fast.dtype, torch.float32)

    def test_state_requires_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeCheckError):
                    FWMLPState(torch.zeros(2, 8, 12, dtype=dtype))
                with self.assertRaises(TypeCheckError):
                    FWMLPState(torch.zeros(2, 8, 12), pending_r=torch.zeros(2, 1, 8, dtype=dtype))
                with self.assertRaises(TypeCheckError):
                    FWMLPState(torch.zeros(2, 8, 12), pending_k=torch.zeros(2, 1, 12, dtype=dtype))

    def test_update_respects_caller_autocast(self):
        model = self.model(use_conv=False)
        initial = FWMLPState(torch.ones(2, 8, 12))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, pending = model(self.teacher[:, :2], self.student[:, :2], state=initial)
            expected = pending.W_fast + model.lr * (pending.pending_r.transpose(1, 2) @ pending.pending_k)
        # Complete the chunk with zero features (no additional write).
        zero = torch.zeros(2, 1, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, committed = model(zero, zero, state=pending)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(committed.W_fast.dtype, torch.float32)
        torch.testing.assert_close(committed.W_fast, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(committed.W_fast, committed.W_fast.bfloat16().float()))
        torch.testing.assert_close(initial.W_fast, torch.ones_like(initial.W_fast))


if __name__ == "__main__":
    unittest.main()
