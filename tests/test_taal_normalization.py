import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import MemoryMLP, NeuralMemory
from evaluation.run import load_model
from evaluation.taal.decoder_diagnostic import DecoderRecorder, FixedReplaySource, replay_fixed_tokens
from evaluation.taal.normalization_ablation import ARMS, ablation_config, run_fixed, run_generation
from evaluation.taal.numerical_diagnostic import independent_gradient, NumericalFailure
from tests.test_neural_memory import always_vmap_chunk_gradient
from tests.test_taal_longmemeval import Tokenizer


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_configuration_validation_and_d_width_write_oracle(self):
        for kwargs in ({"pre_rmsnorm": 1}, {"mlp_output_layernorm": "true"},
                       {"normalization_eps": 0}, {"normalization_eps": float("nan")},
                       {"normalization_eps": True}):
            with self.assertRaises(ValueError):
                NeuralMemoryConfig(dim=128, **kwargs)
        memory = NeuralMemory(NeuralMemoryConfig(dim=128, depth=2, mlp_output_layernorm=True)).eval()
        weights = memory.initial_state(1).weights
        keys, values, strength = torch.randn(1, 1, 128), torch.randn(1, 1, 128), torch.rand(1, 1)
        actual = memory._chunk_gradient(weights, keys, values, strength).chunk_gradients
        expected = independent_gradient(weights, keys, values, strength, torch.float64, True)
        for name in actual:
            torch.testing.assert_close(actual[name].double(), expected["gradients"][name], rtol=1e-4, atol=2e-6)

    def test_manual_write_and_outer_derivatives_match_autograd(self):
        for depth in (1, 2, 3):
            for B, C in ((1, 1), (2, 1), (1, 4), (2, 4)):
                with self.subTest(depth=depth, B=B, C=C):
                    memory = NeuralMemory(NeuralMemoryConfig(dim=8, depth=depth, mlp_output_layernorm=True)).train()
                    weights = memory.initial_state(B).weights
                    keys = torch.randn(B, C, 8, requires_grad=True)
                    values = torch.randn(B, C, 8, requires_grad=True)
                    strength = torch.rand(B, C, requires_grad=True)
                    actual = memory._chunk_gradient(weights, keys, values, strength, trace_tokens=True)
                    predicted = keys
                    for index in range(depth):
                        predicted = predicted @ weights[f"layers.{index}.weight"].transpose(-1, -2)
                        if index + 1 < depth:
                            predicted = F.silu(predicted)
                    predicted = F.layer_norm(predicted, (8,), eps=1e-6)
                    loss = ((predicted - values).square().mean(-1) * strength).sum()
                    expected = torch.autograd.grad(loss, tuple(weights.values()), create_graph=True)
                    for name, reference in zip(weights, expected):
                        torch.testing.assert_close(actual.chunk_gradients[name], reference, rtol=2e-5, atol=3e-6)
                        torch.testing.assert_close(actual.token_gradients[name].sum(1), reference, rtol=2e-5, atol=3e-6)
                    coefficients = [torch.randn_like(value) for value in expected]
                    actual_scalar = sum((actual.chunk_gradients[name] * coefficient).sum()
                                        for name, coefficient in zip(weights, coefficients))
                    expected_scalar = sum((value * coefficient).sum() for value, coefficient in zip(expected, coefficients))
                    inputs = (*weights.values(), keys, values, strength)
                    outer = torch.autograd.grad(actual_scalar, inputs, retain_graph=True)
                    reference_outer = torch.autograd.grad(expected_scalar, inputs)
                    for actual_value, reference in zip(outer, reference_outer):
                        torch.testing.assert_close(actual_value, reference, rtol=1e-4, atol=5e-5)

    def test_recurrent_delayed_gradients_and_split_calls(self):
        for B, C in ((1, 1), (2, 3)):
            memory = NeuralMemory(NeuralMemoryConfig(dim=4, chunk_size=C,
                pre_rmsnorm=True, mlp_output_layernorm=True))
            reference = copy.deepcopy(memory)
            reference._chunk_gradient = MethodType(always_vmap_chunk_gradient, reference)
            inputs = torch.randn(B, 7, 4, requires_grad=True)
            output, state = memory(inputs)
            expected, expected_state = reference(inputs)
            torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
            output[:, -1].square().sum().backward(retain_graph=True)
            expected[:, -1].square().sum().backward()
            for actual_parameter, reference_parameter in zip(memory.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, atol=5e-4, rtol=5e-4)
            memory.eval()
            full, full_state = memory(inputs.detach())
            first, partial = memory(inputs[:, :2].detach())
            second, partial = memory(inputs[:, 2:].detach(), state=partial)
            torch.testing.assert_close(torch.cat((first, second), 1), full, atol=2e-5, rtol=2e-4)
            for name in full_state.weights:
                torch.testing.assert_close(partial.weights[name], full_state.weights[name])

    def test_pre_norm_reaches_all_projections_and_controls_in_both_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            memory = NeuralMemory(NeuralMemoryConfig(dim=8, pre_rmsnorm=True)).to(dtype).eval()
            inputs = (1000 * torch.randn(1, 3, 8)).to(dtype)
            expected = F.rms_norm(inputs.float(), (8,), eps=1e-6).to(dtype)
            seen = {}
            handles = []
            for name in ("query_projection", "key_projection", "value_projection",
                         "forget_projection", "momentum_projection", "write_strength_projection"):
                def capture(module, args, label=name):
                    seen.setdefault(label, []).append(args[0].detach().clone().reshape(1, -1, 8))
                handles.append(getattr(memory, name).register_forward_pre_hook(capture))
            try:
                memory(inputs)
            finally:
                for handle in handles:
                    handle.remove()
            for values in seen.values():
                torch.testing.assert_close(torch.cat(values, dim=1), expected, atol=0, rtol=0)

    def test_output_norm_is_fixed_affine_and_read_write_consistent(self):
        mlp = MemoryMLP(8, 2, output_layernorm=True)
        keys = torch.randn(2, 3, 8)
        result = mlp(keys, return_intermediates=True)
        expected = F.layer_norm(mlp.layers[1](F.silu(mlp.layers[0](keys))), (8,), eps=1e-6)
        torch.testing.assert_close(result.predicted_values, expected, atol=0, rtol=0)
        torch.testing.assert_close(mlp(keys).predicted_values, expected, atol=0, rtol=0)
        self.assertEqual(set(dict(mlp.named_parameters())), {"layers.0.weight", "layers.1.weight"})
        self.assertTrue(torch.isfinite(mlp(torch.zeros_like(keys)).predicted_values).all())
        memory = NeuralMemory(NeuralMemoryConfig(dim=8, mlp_output_layernorm=True)).eval()
        weights = memory.initial_state(1).weights
        keys, values, strength = torch.randn(1, 1, 8), torch.randn(1, 1, 8), torch.rand(1, 1)
        actual = memory._chunk_gradient(weights, keys, values, strength).chunk_gradients
        expected = independent_gradient(weights, keys, values, strength, torch.float32, True)
        for name in actual:
            self.assertFalse(actual[name].requires_grad)
            torch.testing.assert_close(actual[name], expected["gradients"][name], rtol=2e-5, atol=3e-6)

    def tiny_config(self):
        return TaalQwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            working_memory_size=4, max_persistent_kv_tokens=0, memory_dim=8, memory_depth=2,
            memory_conv_kernel_size=2, memory_chunk_size=1, num_persistent_tokens=2)

    def test_old_checkpoint_strict_loading_all_arms_and_baseline_parity(self):
        config = self.tiny_config()
        original = TaalQwen3ForCausalLM(config).bfloat16().eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            path = Path(directory) / "config.json"
            saved = json.loads(path.read_text())
            for name in ("memory_pre_rmsnorm", "memory_mlp_output_layernorm", "memory_normalization_eps"):
                saved.pop(name)
            path.write_text(json.dumps(saved))
            old_config = TaalQwen3Config.from_pretrained(directory)
            expected, _ = replay_fixed_tokens(original, torch.tensor([[1, 2, 3]]), [4], 4, 1.0)
            for arm in ARMS:
                changed = ablation_config(old_config, arm)
                loaded = load_model(directory, changed).eval()
                self.assertEqual(loaded.config.taal_layer_config().memory.pre_rmsnorm, ARMS[arm][0])
                self.assertEqual(loaded.config.taal_layer_config().memory.mlp_output_layernorm, ARMS[arm][1])
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)
                output, _ = replay_fixed_tokens(loaded, torch.tensor([[1, 2, 3]]), [4], 4, 1.0)
                self.assertTrue(torch.isfinite(output.logits).all())
                if arm == "baseline":
                    torch.testing.assert_close(output.logits, expected.logits, atol=0, rtol=0)
                roundtrip = TaalQwen3Config.from_dict(changed.to_dict())
                self.assertEqual(roundtrip.taal_layer_config(), changed.taal_layer_config())

    def test_actual_runner_all_arms_complete_traces_fresh_generation_and_failure(self):
        baseline = TaalQwen3ForCausalLM(self.tiny_config()).eval()
        with torch.no_grad():
            for layer in baseline.model.layers:
                layer.taal.residual_gate.fill_(0.1)
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        forced = [20, 20]
        metadata = {"tokenizer": "test", "tokenizer_revision": "test"}
        settings = {"seed": 42, "save_traces": True, "model": {"memory_read_scale": 1.0},
                    "generation": {"execution_block_size": 4, "max_new_tokens": 2, "do_sample": False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decoder = DecoderRecorder(baseline, root / "reference", [1, 2, 3], 7)
            with decoder:
                replay_fixed_tokens(baseline, prompt, forced, 4, 1.0)
            reference = {(row["layer"], row["position"]): {
                "hidden_in": row["components"]["memory_input"]["norm_finite"],
                "hidden_out": row["components"]["memory_output"]["norm_finite"],
            } for row in decoder.rows}
            source = FixedReplaySource({"metadata": metadata, "settings": settings},
                {"question_id": "test", "prompt_length": 9, "input_ids": prompt[0].tolist(),
                 "answer": "test", "persistent_prefix_length": 0}, forced, reference)
            for arm in ARMS:
                model = TaalQwen3ForCausalLM(ablation_config(baseline.config, arm)).eval()
                model.load_state_dict(baseline.state_dict(), strict=True)
                manifest = {"arm": arm, "checkpoint": "test", "tokenizer": "test",
                    "layers": [1, 2, 3], "trace_start": 7, "execution_block_size": 4,
                    "normalization": ARMS[arm], "max_new_tokens": 2}
                result = run_fixed(model, Tokenizer(), source, manifest, root / arm)
                self.assertEqual(result["status"], "finite_replay")
                self.assertEqual(result["trace_status"], "exported")
                if arm == "baseline":
                    self.assertTrue(result["reference_comparison"]["matched_within_1e-5"])
                generated = run_generation(model, Tokenizer(), source, manifest, root / f"{arm}-generation")
                self.assertEqual(generated["status"], "finite_generation")
                self.assertEqual(generated["trace_status"], "exported")
                for folder in (root / arm, root / f"{arm}-generation"):
                    trace_manifest = json.loads((folder / "memory_traces/manifest.json").read_text())
                    self.assertEqual(len(trace_manifest["episodes"]), 1)
                    self.assertEqual(trace_manifest["run_metadata"]["layers"], [0, 1, 2, 3])
                    relative = next(iter(trace_manifest["episodes"].values()))
                    with gzip.open(folder / "memory_traces" / relative, "rt") as stream:
                        episode = json.load(stream)
                    self.assertEqual(len(episode["internal_prefixes"]), 4)
                    self.assertEqual(len(episode["reads"]), 4 * len(episode["tokens"]))
                    self.assertEqual(len(episode["writes"]), 4 * len(episode["tokens"]) + 8)
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            with patch("evaluation.taal.normalization_ablation.replay_fixed_tokens",
                       side_effect=NumericalFailure({"stage": "test", "layer": 2, "start": 0})):
                result = run_fixed(model, Tokenizer(), source, manifest, root / "failed")
            self.assertEqual(result["status"], "nonfinite_detected")
            self.assertEqual(result["trace_status"], "incomplete")
            self.assertTrue((root / "failed/report.json").is_file())
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_d_geometry_all_normalizations(self):
        config = TaalQwen3Config.from_dict(json.loads(Path("tests/fixtures/taal_d_final_config.json").read_text()))
        config.num_hidden_layers = 4
        config.layer_types = config.layer_types[:4]
        config.vocab_size = 32
        config.eos_token_id = 31
        for arm in ARMS:
            model = TaalQwen3ForCausalLM(ablation_config(config, arm)).bfloat16().eval()
            with torch.no_grad():
                for layer in model.model.layers:
                    layer.taal.residual_gate.fill_(0.05)
            output, _ = replay_fixed_tokens(model, torch.randint(0, 32, (1, 35)), [20, 20], 32, 1.0)
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertEqual(output.state.tokens_seen, 37)


if __name__ == "__main__":
    unittest.main()
