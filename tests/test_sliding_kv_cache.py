import unittest

import torch

from architectures.cache.sliding_window import SlidingWindowKVCache, SlidingWindowKVLayer


class SlidingKVCacheTests(unittest.TestCase):
    def test_returned_keys_retained_suffix_and_absolute_offsets(self):
        for window in (1, 2, 5):
            cache = SlidingWindowKVCache(num_layers=1, window_size=window)
            seen = 0
            for count in (1, 2, 9, 1, 3, 1):
                positions = torch.arange(seen, seen + count)
                length, offset = cache.get_mask_sizes(positions, layer_idx=0)
                expected_start = max(0, seen - (window - 1))
                self.assertEqual((length, offset), (seen + count - expected_start, expected_start))
                new_keys = positions.float().reshape(1, 1, count, 1)
                keys, values = cache.update(new_keys, new_keys + 100, layer_idx=0)
                seen += count
                expected_keys = torch.arange(expected_start, seen).float().reshape(1, 1, -1, 1)
                torch.testing.assert_close(keys, expected_keys)
                torch.testing.assert_close(values, expected_keys + 100)
                retained_start = max(0, seen - (window - 1))
                retained = torch.arange(retained_start, seen).float().reshape(1, 1, -1, 1)
                torch.testing.assert_close(cache.layers[0].keys, retained)
                torch.testing.assert_close(cache.layers[0].values, retained + 100)
                self.assertEqual(cache.get_seq_length(), seen)

    def test_retained_storage_does_not_pin_prefill_allocation(self):
        for window in (1, 5):
            cache = SlidingWindowKVCache(1, window)
            with torch.no_grad():
                x = torch.randn(2, 2, 100, 4)
                full_keys, full_values = cache.update(x, x + 1, 0)
            self.assertEqual(full_keys.shape[-2], 100)
            for stored, full in ((cache.layers[0].keys, full_keys), (cache.layers[0].values, full_values)):
                self.assertEqual(stored.shape[-2], window - 1)
                self.assertEqual(stored.untyped_storage().nbytes(), stored.numel() * stored.element_size())
                self.assertNotEqual(stored.untyped_storage().data_ptr(), full.untyped_storage().data_ptr())

    def test_retained_tensors_preserve_training_gradients(self):
        cache = SlidingWindowKVCache(1, 4)
        k = torch.randn(1, 1, 6, 2, requires_grad=True)
        v = torch.randn(1, 1, 6, 2, requires_grad=True)
        cache.update(k, v, 0)
        new_k = torch.randn(1, 1, 2, 2, requires_grad=True)
        new_v = torch.randn(1, 1, 2, 2, requires_grad=True)
        full_k, full_v = cache.update(new_k, new_v, 0)
        (full_k.sum() + full_v.sum()).backward()
        expected = torch.zeros_like(k)
        expected[..., -3:, :] = 1
        torch.testing.assert_close(k.grad, expected)
        torch.testing.assert_close(v.grad, expected)
        torch.testing.assert_close(new_k.grad, torch.ones_like(new_k))
        torch.testing.assert_close(new_v.grad, torch.ones_like(new_v))

    def test_reset_clears_history_and_positions(self):
        cache = SlidingWindowKVCache(2, 4)
        x = torch.randn(1, 1, 9, 2)
        for i in range(2):
            cache.update(x, x, i)
        old_keys = cache.layers[0].keys
        saved = old_keys.clone()
        cache.reset()
        torch.testing.assert_close(old_keys, saved)
        for i in range(2):
            self.assertEqual(cache.get_seq_length(i), 0)
            self.assertEqual(cache.get_mask_sizes(torch.arange(2), i), (2, 0))
            keys, _ = cache.update(x[..., :2, :], x[..., :2, :], i)
            torch.testing.assert_close(keys, x[..., :2, :])

    def test_invalid_configuration(self):
        for window in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                SlidingWindowKVLayer(window)
        with self.assertRaises(ValueError):
            SlidingWindowKVCache(0, 4)


if __name__ == "__main__":
    unittest.main()
