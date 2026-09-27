import unittest
from types import SimpleNamespace

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from training.taal_timing import TaalMicrobatchTimer


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.taal = torch.nn.Linear(4, 4)
        self.body = torch.nn.Linear(4, 4)

    def forward(self, inputs):
        return self.body(self.taal(inputs))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Decoder(), Decoder()])
        self.model.gradient_checkpointing = False
        self.config = SimpleNamespace(num_persistent_tokens=2)

    def forward(self, inputs):
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


class TaalTimingTests(unittest.TestCase):
    def test_first_batch_timing_preserves_outputs_and_removes_hooks(self):
        torch.manual_seed(8)
        model = Model()
        inputs = torch.randn(1, 5, 4)
        baseline = model(inputs)
        timer = TaalMicrobatchTimer(model, torch.device("cpu"), sequence_length=5)
        with timer:
            observed = model(inputs)
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
        record = timer.forward_record()
        self.assertEqual(record["kind"], "taal_first_microbatch_forward")
        self.assertEqual(record["expected_memory_update_and_read_calls"], 14)
        self.assertEqual(len(record["per_layer"]), 2)
        self.assertGreaterEqual(record["memory_cpu_dispatch_s"], 0)
        self.assertIsNone(record["per_layer"][0]["memory_gpu_timeline_ms"])
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        self.assertTrue(all(not layer.taal._forward_hooks for layer in model.model.layers))

    def test_timing_real_taal_forward_keeps_outer_gradient(self):
        config = TaalQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=2,
            memory_chunk_size=1,
            num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).train()
        ids = torch.tensor([[1, 2, 3]])
        baseline = model.forward_bridge_memory(
            input_ids=ids,
            delayed_labels=torch.tensor([[-100, -100, 3]]),
            all_token_loss_weight=0.0,
            delayed_answer_loss_weight=1.0,
        )
        timer = TaalMicrobatchTimer(model, torch.device("cpu"), sequence_length=3)
        with timer:
            result = model.forward_bridge_memory(
                input_ids=ids,
                delayed_labels=torch.tensor([[-100, -100, 3]]),
                all_token_loss_weight=0.0,
                delayed_answer_loss_weight=1.0,
            )
        torch.testing.assert_close(result.loss, baseline.loss, atol=0, rtol=0)
        record = timer.forward_record()
        self.assertEqual(record["expected_memory_update_and_read_calls"], 10)
        stages = record["stage_host_spans_all_layers"]
        self.assertEqual(stages["gradient_execution"]["calls"], 10)
        self.assertEqual(stages["read_execution"]["calls"], 10)
        self.assertEqual(stages["committed_state_construction"]["calls"], 10)
        self.assertEqual(stages["adapter_projection_in"]["calls"], 2)
        self.assertIsNone(model.model.layers[0].taal.timing_observer)
        self.assertIsNone(model.model.layers[0].taal.neural_memory.timing_observer)
        result.loss.backward()
        self.assertIsNotNone(model.model.layers[0].taal.memory_projection_in.weight.grad)


if __name__ == "__main__":
    unittest.main()
