import json
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.titans.state import NeuralMemoryState
from evaluation.taal.numerical_diagnostic import BlockMonitor, NumericalFailure, check_write
from evaluation.taal.longmemeval import create_trace_exporter, generate_taal_example
from evaluation.taal.runaway_diagnostic import RunawayRecorder, candidate_reasons, norm64
from tests.test_taal_longmemeval import Tokenizer


class RunawayDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        config = TaalQwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            working_memory_size=4, max_persistent_kv_tokens=0, memory_dim=8, memory_depth=2,
            memory_conv_kernel_size=2, memory_chunk_size=1, num_persistent_tokens=2)
        config._attn_implementation = "sdpa"
        self.model = TaalQwen3ForCausalLM(config).eval()
        self.ids = torch.randint(0, 32, (1, 9))
        self.options = {"execution_block_size": 4, "max_new_tokens": 3, "do_sample": False, "eos_token_id": []}

    def test_full_generation_bitwise_parity_and_complete_scalar_coverage(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.model.to(dtype)
            with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
                expected = self.model.generate(self.ids, **self.options)
                monitor = BlockMonitor(self.model, 9)
                folder = Path(directory) / "updates"
                with monitor, RunawayRecorder(monitor, folder, tail_size=2):
                    actual = self.model.generate(self.ids, **self.options)
                torch.testing.assert_close(actual.token_ids, expected.token_ids, atol=0, rtol=0)
                for index, state in actual.state.memory_states.items():
                    for name, weight in state.weights.items():
                        torch.testing.assert_close(weight, expected.state.memory_states[index].weights[name], atol=0, rtol=0)
                rows = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
                summary = json.loads((folder / "summary.json").read_text())
                self.assertEqual(summary["updates"], 28)
                self.assertEqual(summary["reads"], 24)
                self.assertEqual(len(summary["tail_slices"]), 4)
                self.assertEqual(sorted(row["update_order"] for row in rows), list(range(28)))
                for layer in (0, 1):
                    text = [row for row in rows if row["layer"] == layer and row["position"] is not None]
                    self.assertEqual([row["position"] for row in text], list(range(12)))
                    self.assertEqual([row["phase"] for row in text], ["prompt"] * 9 + ["generated"] * 3)
                    self.assertTrue(all("injection" in row["metrics"] for row in text))
                    internal = [row for row in rows if row["layer"] == layer and row["position"] is None]
                    self.assertEqual([row["internal_index"] for row in internal], [0, 1])
                    self.assertTrue(all("injection" not in row["metrics"] for row in internal))
                self.assertTrue(all(not row["nonfinite_metrics"] for row in rows))
                saved = torch.load(folder / summary["tail_slices"][0], weights_only=True)
                state, *rest = saved["step"]
                memory = self.model.model.layers[saved["identity"]["layer"]].taal.neural_memory
                oracle = check_write(memory, (NeuralMemoryState(**state), *rest))
                self.assertEqual(oracle["status"], "compared")
                for parameter in oracle["parameters"].values():
                    self.assertLess(parameter["manual_vs_autograd_fp32"]["relative_l2_error"], 1e-5)
            self.assert_clean()

    def assert_clean(self):
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in self.model.modules()))
        for layer in self.model.model.layers:
            self.assertNotIn("_process_chunk", layer.taal.neural_memory.__dict__)
            self.assertNotIn("_chunk_gradient", layer.taal.neural_memory.__dict__)

    def test_failure_keeps_prior_rows_safe_snapshots_and_cleanup(self):
        # Inject after parameter validation to test exception cleanup during a call.
        with torch.no_grad():
            self.model.model.layers[1].taal.memory_projection_out.weight.fill_(0.1)
        with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "updates"
            monitor = BlockMonitor(self.model, 9)
            with self.assertRaises(NumericalFailure):
                with monitor, RunawayRecorder(monitor, folder, tail_size=2):
                    self.model.model.layers[1].taal.memory_projection_out.weight.fill_(float("inf"))
                    self.model.generate(self.ids, **self.options)
            rows = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(any("injection" in row["nonfinite_metrics"] for row in rows))
            self.assertEqual(json.loads((folder / "summary.json").read_text())["exception"], "NumericalFailure")
            self.assert_clean()

    def test_candidate_thresholds_separate_growth_loss_and_residual(self):
        row = {"metrics": {"weight_before": 2., "weight_after": 30., "same_key_mse_before": 1.,
                           "same_key_mse_after": 101., "hidden_in": 2., "injection": 21.}}
        self.assertEqual(candidate_reasons(row, 2.), ["weight_growth", "same_key_loss_growth", "read_dominates_residual"])
        self.assertEqual(candidate_reasons({"metrics": {"weight_before": None, "weight_after": None}}, 1), [])
        self.assertEqual(candidate_reasons({"metrics": {"weight_before": 1., "weight_after": 2.}}, 1), [])

    def test_observation_norm_does_not_overflow_at_real_failure_scale(self):
        value = torch.full((128,), 1e28, dtype=torch.bfloat16)
        self.assertFalse(torch.isfinite(value.norm()))
        self.assertTrue(torch.isfinite(norm64(value)))

    def test_interventions_are_rejected_before_instrumentation(self):
        self.model.model.layers[0].taal.neural_memory.momentum_enabled = False
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unmodified recurrence"):
                with RunawayRecorder(BlockMonitor(self.model, 9), Path(directory) / "updates"):
                    pass
        self.assert_clean()

    def test_compact_trace_export_coexists_with_dense_update_recorder(self):
        settings = {"seed": 42, "model": {}, "generation": {"execution_block_size": 4, "max_new_tokens": 3}}
        example = {"input_ids": self.ids[0].tolist(), "prompt_length": 9, "question_id": "test", "answer": "test"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = create_trace_exporter(root, "test", self.model.config,
                {"tokenizer": "test", "tokenizer_revision": "test"}, settings, {})
            expected = generate_taal_example(self.model, Tokenizer(), example, settings, torch.device("cpu"))
            monitor = BlockMonitor(self.model, 9)
            with monitor, RunawayRecorder(monitor, root / "updates"):
                actual = generate_taal_example(self.model, Tokenizer(), example, settings, torch.device("cpu"), exporter)
            self.assertEqual(actual, expected)
            self.assertEqual(len(exporter.manifest["episodes"]), 1)
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
