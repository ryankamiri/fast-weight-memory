import copy
import unittest

import torch

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from training.engine import collect_metrics, fast_weight_metrics, summarize_metrics, validate


class MetricCallbackTests(unittest.TestCase):
    def model(self, metrics_fn=None):
        config = FWQwen3Config(
            hidden_size=8, intermediate_size=12, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            vocab_size=32, fast_weight_layers=[0], chunk_size=2,
            teacher_window_size=4, student_window_size=2, use_conv=False,
        )
        config._attn_implementation = "sdpa"
        return FWQwen3ForCausalLM(config, metrics_fn=metrics_fn)

    def test_known_ratios(self):
        metrics = fast_weight_metrics(
            torch.ones(4, 8), torch.ones(2, 4, 8) * 2,
            torch.ones(2, 3, 4), torch.ones(2, 3, 4) * 3,
        )
        torch.testing.assert_close(metrics["state_relative_norm"], torch.full((2,), 2.))
        torch.testing.assert_close(metrics["read_relative_rms"], torch.full((2,), 3.))

    def test_optional_callback_and_validation(self):
        model = self.model()
        ids = torch.randint(0, 32, (2, 6))
        batch = {"input_ids": ids, "labels": ids}
        self.assertFalse(any("/fw/" in key for key in validate(model, [batch], torch.device("cpu"))))
        observed = self.model(fast_weight_metrics)
        observed.load_state_dict(model.state_dict())
        metrics = validate(observed, [batch], torch.device("cpu"))
        self.assertIn("val/fw/model.layers.0.mlp/state_relative_norm_mean", metrics)
        self.assertAlmostEqual(metrics["val/loss"], validate(model, [batch], torch.device("cpu"))["val/loss"])

    def test_checkpointed_backward_does_not_duplicate_metrics_or_change_gradients(self):
        model = self.model(fast_weight_metrics).train()
        reference = copy.deepcopy(model)
        model.gradient_checkpointing_enable()
        ids = torch.randint(0, 32, (2, 6))
        collected = {}
        for _ in range(2):
            output = model(input_ids=ids, labels=ids, use_cache=False)
            collect_metrics(model, collected)
            output.loss.backward()
            reference(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        # Two forwards, three chunks, two records. Backward adds no observations.
        for values in collected.values():
            self.assertEqual(torch.cat(values).numel(), 12)
            self.assertTrue(all(not value.requires_grad for value in values))
        self.assertIn("train/fw/model.layers.0.mlp/read_relative_rms_max", summarize_metrics(collected, "train"))
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            if expected.grad is not None:
                torch.testing.assert_close(parameter.grad, expected.grad)


if __name__ == "__main__":
    unittest.main()
