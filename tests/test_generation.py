import unittest
from unittest.mock import patch

import torch
from jaxtyping import TypeCheckError

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from inference.generation import sample_token


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        config = FWQwen3Config(
            vocab_size=40, hidden_size=24, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=6, teacher_window_size=4, student_window_size=2,
            fast_weight_layers=[0], chunk_size=3, conv_kernel_size=2,
            attention_dropout=0.0, eos_token_id=39,
        )
        self.model = FWQwen3ForCausalLM(config).eval()
        self.ids = torch.randint(0, 30, (1, 9))

    @torch.inference_mode()
    def test_greedy_matches_manual_loop_and_returns_complete_state(self):
        flags = torch.arange(9) < 2
        result = self.model.generate(
            self.ids, max_new_tokens=5, do_sample=False, eos_token_id=[],
            execution_block_size=2, persistent_mask=flags,
        )
        manual = self.model.prefill(self.ids, 2, persistent_mask=flags)
        expected = []
        for _ in range(5):
            token = manual.logits[:, -1].argmax(-1, keepdim=True)
            expected.append(token)
            manual = self.model(token, state=manual.state, use_cache=True)
        torch.testing.assert_close(result.token_ids, torch.cat(expected, dim=1))
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(result.state.tokens_seen, 14)
        for actual, reference in zip(result.state.past_key_values.layers, manual.state.past_key_values.layers):
            torch.testing.assert_close(actual.keys, reference.keys)
            torch.testing.assert_close(actual.positions, reference.positions)
            self.assertEqual(actual.is_persistent.sum().item(), 2)
        for name, value in vars(result.state.mlp_states[0]).items():
            torch.testing.assert_close(value, getattr(manual.state.mlp_states[0], name))
        # The next prompt continues directly; generated tokens aren't replayed.
        continued = self.model.generate(self.ids[:, :1], state=result.state, max_new_tokens=1, eos_token_id=[])
        self.assertEqual(continued.state.tokens_seen, 16)

    @torch.inference_mode()
    def test_existing_history_and_eos_are_carried_into_state(self):
        for eos in (None, 39, [38, 39]):
            history = self.model.prefill(self.ids[:, :6], persistent_mask=torch.arange(6) < 2)
            with patch("architectures.qwen.causal_lm.sample_token", return_value=torch.tensor([[39]])) as sample:
                result = self.model.generate(self.ids[:, 6:], state=history.state, eos_token_id=eos)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(result.stop_reason, "eos")
            self.assertEqual(result.token_ids.tolist(), [[39]])
            self.assertEqual(result.state.tokens_seen, 10)
            self.assertEqual(result.state.past_key_values.get_seq_length(), 10)

    def test_seeded_sampling_is_reproducible(self):
        outputs = [self.model.generate(
            self.ids, max_new_tokens=8, eos_token_id=[],
            generator=torch.Generator().manual_seed(42),
        ).token_ids for _ in range(2)]
        torch.testing.assert_close(*outputs)

    def test_sampling_filters_and_greedy(self):
        logits = torch.tensor([[0., 1., 2., 3.]])
        for top_k, top_p in ((1, 1.0), (0, 0.01), (99, 0.01)):
            token = sample_token(logits, True, 0.7, top_k, top_p, torch.Generator().manual_seed(1))
            self.assertEqual(token.item(), 3)
        self.assertEqual(sample_token(logits, False, 0., 0, 0., None).item(), 3)
        # Uniform top-p=0.6 keeps three of four tokens (including the crossing token).
        with patch("torch.multinomial", return_value=torch.tensor([[0]])) as draw:
            sample_token(torch.zeros(1, 4), True, 1., 0, 0.6, None)
        self.assertEqual((draw.call_args.args[0] > 0).sum().item(), 3)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sample_token(torch.tensor([[float("nan")]]), False, 1., 0, 1., None)

    def test_invalid_arguments_are_rejected_before_prefill(self):
        for kwargs in (
            {"max_new_tokens": 0}, {"temperature": 0.}, {"temperature": float("nan")},
            {"top_p": 0.}, {"top_p": 1.1}, {"top_k": -1}, {"eos_token_id": 40},
        ):
            with self.assertRaises(ValueError):
                self.model.generate(self.ids, **kwargs)
        with self.assertRaises(TypeCheckError):
            self.model.generate(self.ids.expand(2, -1))
        self.model.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            self.model.generate(self.ids)


if __name__ == "__main__":
    unittest.main()
