import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import NeuralMemory
from training.taal_timing import TaalEvaluationTimings, TaalKernelSampler, TaalMicrobatchTimer, TaalStageTimings


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
    def test_separate_write_timings_preserve_outputs_and_outer_gradients(self):
        for B, C in ((1, 1), (1, 3), (2, 1), (2, 3)):
            with self.subTest(B=B, C=C):
                torch.manual_seed(19)
                reference = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=C)).train()
                observed = copy.deepcopy(reference)
                reference_input = torch.randn(B, 6, 4, requires_grad=True)
                observed_input = reference_input.detach().clone().requires_grad_(True)
                timer = TaalStageTimings(0)
                observed.timing_observer = timer

                expected, expected_state = reference(reference_input)
                actual, actual_state = observed(observed_input)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for name in expected_state.weights:
                    torch.testing.assert_close(actual_state.weights[name], expected_state.weights[name], atol=0, rtol=0)
                    torch.testing.assert_close(actual_state.momentum[name], expected_state.momentum[name], atol=0, rtol=0)

                stages = timer.summary()
                for stage in ("write_prediction", "write_loss", "gradient_calculation"):
                    # One batched explicit derivative evaluation per memory chunk.
                    self.assertEqual(stages[stage]["calls"], 6 // C)
                    self.assertGreaterEqual(stages[stage]["total_s"], 0)
                    self.assertGreaterEqual(stages[stage]["mean_ms"], 0)

                expected[:, -1].square().mean().backward()
                actual[:, -1].square().mean().backward()
                torch.testing.assert_close(observed_input.grad, reference_input.grad, atol=0, rtol=0)
                for (name, p), (_, q) in zip(reference.named_parameters(), observed.named_parameters()):
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(q.grad, p.grad, atol=0, rtol=0)

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
        self.assertEqual(stages["write_prediction"]["calls"], 10)
        self.assertEqual(stages["write_loss"]["calls"], 10)
        self.assertEqual(stages["gradient_calculation"]["calls"], 10)
        self.assertEqual(stages["read_execution"]["calls"], 10)
        self.assertEqual(stages["committed_state_construction"]["calls"], 10)
        self.assertEqual(stages["adapter_projection_in"]["calls"], 2)
        self.assertEqual(stages["write_mlp_linear"]["calls"], 20)
        self.assertEqual(stages["read_mlp_linear"]["calls"], 20)
        self.assertEqual(stages["write_mlp_silu"]["calls"], 10)
        self.assertEqual(stages["write_weight_gradient"]["calls"], 20)
        self.assertEqual(stages["write_key_gradient"]["calls"], 10)
        self.assertEqual(stages["write_silu_derivative"]["calls"], 10)
        self.assertIsNone(model.model.layers[0].taal.timing_observer)
        self.assertIsNone(model.model.layers[0].taal.neural_memory.timing_observer)
        result.loss.backward()
        self.assertIsNotNone(model.model.layers[0].taal.memory_projection_in.weight.grad)
        backward = timer.backward_record()["per_layer_neural_memory_backward"]
        self.assertEqual(len(backward), 2)
        self.assertTrue(all(row["wall_s"] >= 0 for row in backward))
        self.assertIsNone(model.model.layers[0].taal.neural_memory.memory_mlp.timing_observer)

    def test_multiple_forward_calls_are_aggregated_not_overwritten(self):
        model = Model()
        inputs = torch.randn(1, 5, 4)
        timer = TaalMicrobatchTimer(model, torch.device("cpu"), 5)
        with timer:
            model(inputs[:, :2])
            model(inputs[:, 2:])
        record = timer.forward_record()
        self.assertTrue(all(row["forward_calls"] == 2 for row in record["per_layer"]))

    def test_eval_prefill_timings_count_all_blocks_and_preserve_state(self):
        config = TaalQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, working_memory_size=4, memory_dim=8, memory_depth=2,
            memory_chunk_size=1, num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        expected = model.prefill(ids, execution_block_size=3)
        timings = TaalEvaluationTimings(model, torch.device("cpu"), True)
        with patch("training.taal_timing.print_taal_timing") as recorded:
            with timings.phase("prefill", tokens=5, detailed=True):
                actual = model.prefill(ids, execution_block_size=3)
        record = recorded.call_args_list[0].args[0]
        self.assertEqual(record["per_layer"][0]["forward_calls"], 2)
        self.assertEqual(record["expected_memory_update_and_read_calls"], 7)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for name, weight in expected.state.memory_states[0].weights.items():
            torch.testing.assert_close(actual.state.memory_states[0].weights[name], weight, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.state.past_key_values.layers[0].keys,
            expected.state.past_key_values.layers[0].keys, rtol=0, atol=0,
        )

    def test_kernel_sampler_is_bounded_and_preserves_outer_gradients(self):
        config = TaalQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, working_memory_size=4, memory_dim=8, memory_depth=2,
            memory_chunk_size=1, num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).train()
        reference = copy.deepcopy(model)
        ids = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, -100, 3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            sampler = TaalKernelSampler(model, torch.device("cpu"), path, warmup_calls=1, sample_calls=2)
            with patch("training.taal_timing.print_taal_timing") as recorded, patch("builtins.print"):
                with sampler:
                    result = model.forward_bridge_memory(
                        input_ids=ids, delayed_labels=labels, all_token_loss_weight=0.0,
                        delayed_answer_loss_weight=1.0,
                    )
            self.assertTrue(path.exists())
            self.assertTrue(sampler.finished)
            self.assertEqual(recorded.call_args.args[0]["mlp_calls"], 2)
        expected = reference.forward_bridge_memory(
            input_ids=ids, delayed_labels=labels, all_token_loss_weight=0.0,
            delayed_answer_loss_weight=1.0,
        )
        torch.testing.assert_close(result.loss, expected.loss, rtol=0, atol=0)
        result.loss.backward()
        expected.loss.backward()
        for observed, baseline in zip(model.parameters(), reference.parameters()):
            if baseline.grad is not None:
                torch.testing.assert_close(observed.grad, baseline.grad, rtol=0, atol=0)
        self.assertFalse(model.model.layers[0].taal.neural_memory.memory_mlp._forward_hooks)

    def test_eval_phase_timing_can_be_disabled(self):
        timer = TaalEvaluationTimings(Model(), torch.device("cpu"), False)
        with patch("training.taal_timing.print_taal_timing") as recorded:
            with timer.phase("disabled", detailed=True):
                pass
        self.assertFalse(timer.samples)
        recorded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
