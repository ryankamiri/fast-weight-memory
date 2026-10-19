import json
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.taal.configuration import TaalLayerConfig
from architectures.taal.layer import TaalLayer
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.titans.configuration import NeuralMemoryConfig
from evaluation.taal.numerical_diagnostic import (
    BlockMonitor, MemoryBlock, NumericalFailure, all_finite,
    check_write, independent_gradient, replay_block,
)
from evaluation.taal.longmemeval import create_trace_exporter, generate_taal_example
from tests.test_taal_longmemeval import Tokenizer


class NumericalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        self.adapter = TaalLayer(TaalLayerConfig(
            model_dim=8, memory=NeuralMemoryConfig(dim=8, conv_kernel_size=1),
            num_persistent_tokens=2,
        )).eval()

    def test_finite_replay_preserves_values_and_removes_hooks(self):
        hidden = torch.randn(1, 4, 8)
        block = MemoryBlock(3, 100, [1, 2, 3, 4], hidden, {"prepend_memory_tokens": False})
        with torch.inference_mode():
            expected, expected_state = self.adapter(hidden, prepend_memory_tokens=False)
            result = replay_block(self.adapter, block)
            actual, actual_state = self.adapter(hidden, prepend_memory_tokens=False)
        self.assertFalse(result["reproduced"])
        self.assertEqual([step["position"] for step in result["preceding_finite_steps"]], [100, 101, 102, 103])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name in actual_state.weights:
            torch.testing.assert_close(actual_state.weights[name], expected_state.weights[name], atol=0, rtol=0)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.adapter.modules()))
        json.dumps(result, allow_nan=False)

    def test_overflow_reports_exact_operation_position_and_independent_oracle(self):
        with torch.no_grad():
            for layer in self.adapter.neural_memory.memory_mlp.layers:
                layer.weight.fill_(1e15)
            self.adapter.memory_projection_in.weight.fill_(0.1)
            self.adapter.neural_memory.key_projection.weight.fill_(0.1)
            self.adapter.neural_memory.key_conv.weight.fill_(1)
        for persistent in (False, True):
            block = MemoryBlock(7, 100, [1], torch.ones(1, 1, 8), {"prepend_memory_tokens": persistent})
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "slice.pt"
                report = replay_block(self.adapter, block, target)
                self.assertTrue(target.is_file())
                restored = torch.load(target, weights_only=True)
                self.assertEqual(restored["block"]["layer"], 7)
            self.assertTrue(report["reproduced"])
            failure = report["first_nonfinite"]
            self.assertEqual(failure["layer"], 7)
            self.assertEqual(failure["position"], None if persistent else 100)
            self.assertEqual(failure["internal_index"], 0 if persistent else None)
            self.assertTrue(failure["source"])
            self.assertEqual(failure["operation"], "aten.bmm.default")
            self.assertTrue(all(value["inf"] == 0 and value["nan"] == 0 for value in failure["inputs"]))
            self.assertGreater(sum(value["inf"] for value in failure["outputs"]), 0)
            oracle = report["write_oracle"]
            self.assertEqual(oracle["status"], "compared")
            self.assertTrue(any(parameter["manual_vs_autograd_fp32"]["reference"]["inf"] > 0
                                for parameter in oracle["parameters"].values()))
            self.assertTrue(all(parameter["updated_fp64"]["inf"] == 0
                                for parameter in oracle["parameters"].values()))
            json.dumps(report, allow_nan=False)

    def test_stressed_finite_manual_gradient_matches_autograd_depths_and_bf16_controls(self):
        for depth in (1, 2, 3, 4):
            for scale in (1.0, 10.0, 100.0):
                with self.subTest(depth=depth, scale=scale):
                    adapter = TaalLayer(TaalLayerConfig(model_dim=8,
                        memory=NeuralMemoryConfig(dim=8, depth=depth, conv_kernel_size=1))).to(torch.bfloat16).eval()
                    memory = adapter.neural_memory
                    state = memory.initial_state(1)
                    state.weights = {name: value * scale for name, value in state.weights.items()}
                    state.momentum = {name: torch.randn_like(value) * 0.01 for name, value in state.weights.items()}
                    keys = torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1)
                    step = (state, keys, keys, torch.randn(1, 1, 8), torch.randn(1, 1, 8).bfloat16(),
                            torch.ones(1, 1, dtype=torch.bool), torch.bfloat16)
                    report = check_write(memory, step)
                    for parameter in report["parameters"].values():
                        self.assertLess(parameter["manual_vs_autograd_fp32"]["relative_l2_error"], 1e-5)
                        self.assertLess(parameter["updated_vs_autograd_fp32"]["relative_l2_error"], 1e-5)

    def test_repeated_updates_match_independent_autograd_recurrence(self):
        memory = self.adapter.neural_memory
        manual = memory.initial_state(1)
        weights = {name: value.clone() for name, value in manual.weights.items()}
        momentum = {name: value.clone() for name, value in manual.momentum.items()}
        with torch.no_grad():
            for position in range(128):
                inputs = torch.randn(1, 1, 8)
                keys = torch.nn.functional.normalize(torch.randn(1, 1, 8), dim=-1)
                values = torch.randn(1, 1, 8)
                mask = torch.tensor([[position % 7 != 0]])
                strength = memory.write_strength_projection(inputs).sigmoid().reshape(1, 1)
                strength = torch.where(mask, strength, 0.0)
                gradient = independent_gradient(weights, keys, values, strength, torch.float32)["gradients"]
                forget = memory.forget_projection(inputs[:, 0]).sigmoid().reshape(1, 1, 1)
                retention = memory.momentum_projection(inputs[:, 0]).sigmoid().reshape(1, 1, 1)
                momentum = {name: retention * momentum[name] - gradient[name] for name in weights}
                weights = {name: (1 - forget) * value + momentum[name] for name, value in weights.items()}
                manual = memory._update(manual, inputs, keys, values, mask)
                for name in weights:
                    torch.testing.assert_close(manual.weights[name], weights[name], atol=1e-6, rtol=1e-5)

    def test_monitor_generation_parity_and_exception_cleanup(self):
        config = TaalQwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            working_memory_size=4, max_persistent_kv_tokens=0, memory_dim=8, memory_depth=2,
            memory_conv_kernel_size=2, memory_chunk_size=1, num_persistent_tokens=2)
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).eval()
        ids = torch.randint(0, 32, (1, 257))
        options = {"execution_block_size": 4, "max_new_tokens": 3, "do_sample": False, "eos_token_id": []}
        with torch.inference_mode():
            expected = model.generate(ids, **options)
            with BlockMonitor(model, 257) as monitor:
                actual = model.generate(ids, **options)
            torch.testing.assert_close(actual.token_ids, expected.token_ids, atol=0, rtol=0)
            self.assertTrue(all_finite(actual.state))
            self.assertEqual(monitor.start, 259)
            model.model.layers[1].taal.memory_projection_out.weight.fill_(float("inf"))
            with self.assertRaises(NumericalFailure):
                with BlockMonitor(model, 257):
                    model.generate(ids, **options)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))

    def test_trace_enabled_failure_captures_block_before_bad_updates(self):
        config = TaalQwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            working_memory_size=4, max_persistent_kv_tokens=0, memory_dim=8, memory_depth=2,
            memory_conv_kernel_size=1, memory_chunk_size=1, num_persistent_tokens=2)
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).eval()
        with torch.no_grad():
            adapter = model.model.layers[1].taal
            adapter.persistent_tokens.fill_(1)
            adapter.neural_memory.key_projection.weight.fill_(0.1)
            adapter.neural_memory.key_conv.weight.fill_(1)
            for layer in adapter.neural_memory.memory_mlp.layers:
                layer.weight.fill_(1e15)
        settings = {"seed": 42, "model": {}, "generation": {"execution_block_size": 4, "max_new_tokens": 1}}
        example = {"input_ids": [1, 2, 3, 4, 5], "prompt_length": 5, "question_id": "test", "answer": "test"}
        with tempfile.TemporaryDirectory() as directory:
            exporter = create_trace_exporter(Path(directory), "test", config,
                {"tokenizer": "test", "tokenizer_revision": "test"}, settings, {})
            monitor = BlockMonitor(model, 5)
            with self.assertRaises(NumericalFailure):
                with monitor:
                    generate_taal_example(model, Tokenizer(), example, settings, torch.device("cpu"), exporter)
            self.assertEqual(monitor.last_block.layer, 1)
            self.assertTrue(all_finite(monitor.last_block.hidden_states))
            self.assertFalse(exporter.manifest["episodes"])
            self.assertIsNone(adapter.trace_observer)
            replay = replay_block(adapter, monitor.last_block)
            self.assertTrue(replay["reproduced"])
            self.assertEqual(replay["first_nonfinite"]["internal_index"], 0)


if __name__ == "__main__":
    unittest.main()
