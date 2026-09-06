import tempfile
import unittest
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from inference.prefill import prefill


class CausalLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.ids = torch.randint(0, 40, (2, 13))

    def config(self, **kwargs):
        return FWQwen3Config(
            vocab_size=40, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, teacher_window_size=32, student_window_size=2,
            chunk_size=4, conv_kernel_size=3, **kwargs,
        )

    def test_logits_shifted_loss_and_gradients(self):
        model = FWQwen3ForCausalLM(self.config()).train()
        labels = self.ids.clone()
        labels[:, 3] = -100
        result = model(self.ids, labels=labels, output_hidden_states=True)
        reference_logits = model.lm_head(result.hidden_states[-1])
        expected = F.cross_entropy(
            reference_logits[:, :-1].float().reshape(-1, 40), labels[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(result.loss, expected)
        self.assertIsNone(result.logits)
        self.assertEqual(result.state.tokens_seen, 13)
        self.assertIsNone(result.past_key_values)
        result.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad():
            all_logits = model(self.ids).logits
            for keep in (1, 3, 99):
                torch.testing.assert_close(model(self.ids, logits_to_keep=keep).logits, all_logits[:, -keep:])
        for keep in (-1, True):
            with self.assertRaises(ValueError):
                model(self.ids, logits_to_keep=keep)
        with self.assertRaisesRegex(ValueError, "labels require"):
            model(self.ids, labels=labels, logits_to_keep=1)

    def test_base_checkpoint_loading_and_native_parity(self):
        for tied in (False, True):
            config = self.config(fast_weight_layers=[], tie_word_embeddings=tied)
            native = Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict())).eval()
            with tempfile.TemporaryDirectory() as folder:
                native.save_pretrained(folder)
                loaded, info = FWQwen3ForCausalLM.from_pretrained(
                    folder, config=config, output_loading_info=True,
                )
            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["unexpected_keys"], [])
            self.assertEqual(loaded.lm_head.weight is loaded.model.embed_tokens.weight, tied)
            with torch.no_grad():
                actual = loaded(self.ids, labels=self.ids)
                expected = native(self.ids, labels=self.ids, use_cache=False)
            self.assertIsNone(actual.logits)
            with torch.no_grad():
                torch.testing.assert_close(loaded(self.ids).logits, expected.logits)
            torch.testing.assert_close(actual.loss, expected.loss)

    def test_missing_fast_weights_and_save_reload(self):
        for tied in (False, True):
            config = self.config(tie_word_embeddings=tied)
            native = Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict()))
            with tempfile.TemporaryDirectory() as folder:
                native.save_pretrained(folder)
                loaded, info = FWQwen3ForCausalLM.from_pretrained(folder, config=config, output_loading_info=True)
                self.assertEqual(info["unexpected_keys"], [])
                self.assertTrue(info["missing_keys"])
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
                mlp = loaded.model.layers[0].mlp
                torch.testing.assert_close(mlp.W_proj, mlp.W_proj.diagonal().diag())
                self.assertGreater(mlp.W_proj.abs().sum().item(), 0)
                self.assertGreater(mlp.beta_proj.abs().sum().item(), 0)
                torch.testing.assert_close(mlp.student_conv.weight, torch.zeros_like(mlp.student_conv.weight))
                for conv in (mlp.teacher_conv,):
                    torch.testing.assert_close(conv.weight[..., -1], torch.ones(24, 1))
                    torch.testing.assert_close(conv.weight[..., :-1], torch.zeros(24, 1, 2))
                with torch.no_grad():
                    mlp.W_proj.normal_()
                    mlp.student_conv.weight.normal_()
                loaded.save_pretrained(folder)
                restored = FWQwen3ForCausalLM.from_pretrained(folder)
            self.assertEqual(restored.lm_head.weight is restored.model.embed_tokens.weight, tied)
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            self.assertFalse(any("W_fast" in key for key in restored.state_dict()))

    @torch.inference_mode()
    def test_prefill_projects_only_final_token_and_decodes(self):
        config = self.config()
        config.teacher_window_size = 5
        model = FWQwen3ForCausalLM(config).eval()
        expected = model(self.ids, use_cache=True, logits_to_keep=1)
        calls = []
        handle = model.lm_head.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
        try:
            first = prefill(model, self.ids[:, :3])
            calls.clear()
            actual = prefill(model, self.ids[:, 3:], execution_block_size=3, state=first.state)
        finally:
            handle.remove()
        self.assertEqual(calls, [torch.Size([2, 1, 16])])
        torch.testing.assert_close(actual.logits, expected.logits)
        self.assertIs(actual.past_key_values, actual.state.past_key_values)
        self.assertEqual(actual.state.tokens_seen, 13)
        for idx, state in actual.state.mlp_states.items():
            for name, tensor in vars(state).items():
                torch.testing.assert_close(tensor, getattr(expected.state.mlp_states[idx], name))
        for _ in range(4):
            token = expected.logits[:, -1].argmax(-1, keepdim=True)
            actual = model(token, state=actual.state, use_cache=True, logits_to_keep=1)
            expected = model(token, state=expected.state, use_cache=True, logits_to_keep=1)
            torch.testing.assert_close(actual.logits, expected.logits)

    def test_checkpointing_contract(self):
        model = FWQwen3ForCausalLM(self.config()).train()
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            model.gradient_checkpointing_enable({"use_reentrant": True})
        model.gradient_checkpointing_enable()
        self.assertTrue(model.is_gradient_checkpointing)
        model(self.ids, labels=self.ids).loss.backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        self.assertIsNotNone(model.model.layers[0].mlp.W_proj.grad)
        model.gradient_checkpointing_disable()
        self.assertFalse(model.is_gradient_checkpointing)

    @unittest.skipUnless(torch.cuda.is_available(), "Liger parity requires CUDA")
    def test_fused_loss_and_gradients_match_cross_entropy(self):
        for mixed_precision in (False, True):
            with self.subTest(mixed_precision=mixed_precision):
                fused = FWQwen3ForCausalLM(self.config(tie_word_embeddings=True)).cuda().train()
                reference = FWQwen3ForCausalLM(self.config(tie_word_embeddings=True)).cuda().train()
                reference.load_state_dict(fused.state_dict())
                fused.gradient_checkpointing_enable()
                ids = self.ids.cuda()
                labels = ids.clone()
                labels[:, 3] = -100
                context = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
                head_calls = []
                handle = fused.lm_head.register_forward_pre_hook(lambda *args: head_calls.append(True))
                try:
                    with context:
                        result = fused(ids, labels=labels)
                        logits = reference(ids).logits
                        expected = F.cross_entropy(logits[:, :-1].float().reshape(-1, 40), labels[:, 1:].reshape(-1))
                    self.assertIsNone(result.logits)
                    self.assertEqual(head_calls, [])
                    torch.testing.assert_close(result.loss, expected, atol=2e-3, rtol=2e-3)
                    result.loss.backward()
                    expected.backward()
                    for (name, actual), (_, wanted) in zip(fused.named_parameters(), reference.named_parameters()):
                        self.assertIsNotNone(actual.grad, name)
                        torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-3, rtol=2e-2, msg=name)
                finally:
                    handle.remove()


if __name__ == "__main__":
    unittest.main()
