import copy
import unittest

import torch

from architectures.cache.sliding_window import SlidingWindowKVCache
from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.causal_lm import FWQwen3ForCausalLM


class PersistentKVTests(unittest.TestCase):
    def test_persistence_budget_is_cumulative_and_rejection_is_atomic(self):
        cache = SlidingWindowKVCache(1, 2, max_persistent_tokens=2)
        x = torch.ones(1, 1, 3, 1)
        cache.update(x, x, 0, {"persistent_mask": torch.tensor([True, True, False])})
        layer = cache.layers[0]
        saved_keys = layer.keys.clone()
        saved_positions = layer.positions.clone()
        with self.assertRaisesRegex(ValueError, "max_persistent_tokens=2"):
            cache.update(x, x, 0, {"persistent_mask": torch.tensor([False, True, False])})
        self.assertEqual(cache.get_seq_length(), 3)
        torch.testing.assert_close(layer.keys, saved_keys)
        torch.testing.assert_close(layer.positions, saved_positions)
        cache.update(x, x, 0)
        self.assertEqual(layer.is_persistent.sum().item(), 2)

    def test_zero_and_invalid_budgets(self):
        cache = SlidingWindowKVCache(1, 2, max_persistent_tokens=0)
        x = torch.ones(1, 1, 1, 1)
        cache.update(x, x, 0)
        with self.assertRaisesRegex(ValueError, "max_persistent_tokens=0"):
            cache.update(x, x, 0, {"persistent_mask": torch.tensor([True])})
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                FWQwen3Config(max_persistent_tokens=invalid)
            with self.assertRaises(ValueError):
                SlidingWindowKVCache(1, 2, max_persistent_tokens=invalid)
        restored = FWQwen3Config.from_dict(FWQwen3Config(max_persistent_tokens=17).to_dict())
        self.assertEqual(restored.max_persistent_tokens, 17)

    @torch.inference_mode()
    def test_prefill_rejects_over_budget_before_updating_state(self):
        model = self.model([])
        model.config.max_persistent_tokens = 2
        ids = torch.ones(1, 4, dtype=torch.long)
        result = model.prefill(ids, 2, persistent_mask=torch.tensor([True, False, False, False]))
        with self.assertRaisesRegex(ValueError, "max_persistent_tokens=2"):
            model.prefill(ids, 1, state=result.state, persistent_mask=torch.ones(4, dtype=torch.bool))
        self.assertEqual(result.state.past_key_values.get_seq_length(), 4)
        with self.assertRaisesRegex(ValueError, "max_persistent_tokens=2"):
            model(ids, persistent_mask=torch.ones(4, dtype=torch.bool))

    def model(self, fast_layers, backend="sdpa"):
        torch.manual_seed(12)
        config = FWQwen3Config(
            vocab_size=40, hidden_size=24, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=6, teacher_window_size=4, student_window_size=2,
            fast_weight_layers=fast_layers, chunk_size=3, conv_kernel_size=2,
            attention_dropout=0.0,
        )
        config._attn_implementation = backend
        return FWQwen3ForCausalLM(config).eval()

    def test_cache_owns_flags_and_positions_after_eviction(self):
        for window in (1, 4):
            cache = SlidingWindowKVCache(1, window)
            x = torch.arange(8).float().reshape(1, 1, 8, 1)
            persistent = torch.tensor([True, True, False, False, False, True, False, False])
            full, _ = cache.update(x, x + 100, 0, {"persistent_mask": persistent})
            torch.testing.assert_close(full, x)
            layer = cache.layers[0]
            expected = torch.arange(8)[persistent | (torch.arange(8) >= 8 - (window - 1))]
            torch.testing.assert_close(layer.positions, expected)
            torch.testing.assert_close(layer.keys.flatten(), expected.float())
            torch.testing.assert_close(layer.values.flatten(), expected.float() + 100)
            next_positions, flags = layer.attention_metadata(torch.tensor([8]))
            torch.testing.assert_close(next_positions, torch.cat((expected, torch.tensor([8]))))
            self.assertFalse(flags[-1])
            cache.update(torch.tensor([[[[8.]]]]), torch.tensor([[[[108.]]]]), 0)
            self.assertEqual(layer.is_persistent.sum().item(), 3)
            with self.assertRaisesRegex(ValueError, "explicit attention_metadata"):
                cache.get_mask_sizes(torch.tensor([9]), 0)
            cache.reset()
            self.assertIsNone(layer.positions)
            self.assertIsNone(layer.is_persistent)

    def test_persistent_keys_are_visible_but_never_future_visible(self):
        model = self.model([0]).model
        positions = torch.arange(8)
        persistent = torch.tensor([True, False, False, False, False, False, False, True])
        masks = model._prepare_masks(None, torch.zeros(1, 8, 24), positions, positions, persistent)
        for name, window in (("teacher", 4), ("student", 2)):
            expected = (positions[:, None] >= positions[None, :]) & (
                persistent[None, :] | (positions[None, :] > positions[:, None] - window)
            )
            torch.testing.assert_close(masks[name][0, 0] == 0, expected)

    def test_checkpointed_training_matches_without_a_cache(self):
        model = self.model([0]).train()
        checked = copy.deepcopy(model)
        checked.gradient_checkpointing_enable()
        ids = torch.randint(0, 40, (1, 9))
        persistent = torch.arange(9) < 2
        outputs = []
        for candidate in (model, checked):
            output = candidate(ids, persistent_mask=persistent)
            self.assertIsNone(output.state.past_key_values)
            output.logits.square().mean().backward()
            outputs.append(output.logits)
        torch.testing.assert_close(*outputs)
        for (name, parameter), (_, other) in zip(model.named_parameters(), checked.named_parameters()):
            torch.testing.assert_close(parameter.grad, other.grad, msg=name)

    def test_invalid_ingestion_flags_do_not_mutate_cache(self):
        cache = SlidingWindowKVCache(1, 3)
        x = torch.ones(1, 1, 2, 1)
        for flags in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, "persistent_mask"):
                cache.update(x, x, 0, {"persistent_mask": flags})
            self.assertEqual(cache.get_seq_length(), 0)
            self.assertIsNone(cache.layers[0].positions)

    @torch.inference_mode()
    def test_cached_prefill_and_decode_match_uncached_reference(self):
        # Includes a prefix split across prefill blocks and a later persistent
        # token. Original positions must survive gaps in cache storage.
        for fast_layers in ([], [0]):
            for backend in ("eager", "sdpa"):
                model = self.model(fast_layers, backend)
                ids = torch.randint(0, 40, (2, 15))
                persistent = torch.zeros(15, dtype=torch.bool)
                persistent[:5] = True
                persistent[7] = True
                reference = model(ids, persistent_mask=persistent).logits
                for block_size in (1, 2, 6, 12):
                    result = model.prefill(ids[:, :12], block_size, persistent_mask=persistent[:12])
                    torch.testing.assert_close(result.logits, reference[:, 11:12], atol=2e-6, rtol=2e-5)
                    for position in range(12, 15):
                        result = model(ids[:, position:position + 1], state=result.state, use_cache=True)
                        torch.testing.assert_close(result.logits, reference[:, position:position + 1], atol=2e-6, rtol=2e-5)
                    for layer in result.state.past_key_values.layers:
                        expected = torch.tensor([0, 1, 2, 3, 4, 7, 12, 13, 14])
                        torch.testing.assert_close(layer.positions, expected)
                        torch.testing.assert_close(layer.is_persistent, persistent[expected])

    @torch.inference_mode()
    def test_omitting_flags_matches_explicit_false(self):
        model = self.model([0])
        ids = torch.randint(0, 40, (1, 12))
        a = model.prefill(ids, 3)
        b = model.prefill(ids, 3, persistent_mask=torch.zeros(12, dtype=torch.bool))
        torch.testing.assert_close(a.logits, b.logits)


if __name__ == "__main__":
    unittest.main()
