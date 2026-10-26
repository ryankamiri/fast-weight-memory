import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import NeuralMemory
from evaluation.taal.decoder_diagnostic import FixedReplaySource
from evaluation.taal.normalization_ablation import run_fixed, run_generation
from evaluation.taal.stability_cache import compare_cache
from evaluation.taal.stability_dynamics import length_examples
from evaluation.taal.stability_precision import write_comparison
from evaluation.taal.stability_interventions import scaled_gradient, intervene
from evaluation.taal.stability_suite import TRAINING_ARMS, arm_config_path
from evaluation.taal.trace_export import (
    TaalTraceRecorder,
    TaalTraceExporter,
    make_trace_tokens,
)
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.stability_observation import ReadRoundingRecorder
from tests.test_taal_longmemeval import Tokenizer
from training.config import load_config
from training.engine import build_scheduler
from training.optimization import build_optimizer, clip_training_gradients


def independent_prediction(keys, weights, depth, normalized, residual, scaled):
    value = keys
    for index in range(depth):
        value = value @ weights[f"layers.{index}.weight"].transpose(-1, -2)
        if index + 1 < depth:
            value = F.silu(value)
    if normalized:
        value = F.layer_norm(value, (keys.shape[-1],), eps=1e-6)
    if scaled:
        value = value * (1 + weights["norm_gamma"].unsqueeze(1))
    if residual:
        value = keys + value
    return value


class StabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_long_diagnostic_exports_new_state_shapes_and_transition_controls(self):
        source = FixedReplaySource(
            {
                "metadata": {"tokenizer": "fixture", "tokenizer_revision": "fixture"},
                "settings": {
                    "seed": 42,
                    "save_traces": True,
                    "model": {"memory_read_scale": 1.0},
                    "generation": {
                        "execution_block_size": 4,
                        "max_new_tokens": 2,
                        "do_sample": False,
                    },
                },
            },
            {
                "question_id": "fixture",
                "prompt_length": 9,
                "input_ids": list(range(1, 10)),
                "answer": "fixture",
                "persistent_prefix_length": 0,
            },
            [20, 20],
            {},
        )
        for arm in ("read_ln", "scale_residual_ln"):
            settings = load_config(arm_config_path(arm)).model
            config = TaalQwen3Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                working_memory_size=4,
                max_persistent_kv_tokens=0,
                memory_dim=8,
                num_persistent_tokens=2,
                memory_read_layernorm=settings.memory_read_layernorm,
                memory_mlp_output_layernorm=settings.memory_mlp_output_layernorm,
                memory_norm_learned_scale=settings.memory_norm_learned_scale,
                memory_norm_residual=settings.memory_norm_residual,
            )
            model = TaalQwen3ForCausalLM(config).eval()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = {
                    "arm": arm,
                    "checkpoint": str(root),
                    "tokenizer": "fixture",
                    "layers": [0, 1],
                    "trace_start": 0,
                    "execution_block_size": 4,
                    "max_new_tokens": 2,
                    "normalization": {},
                    "allow_transition_ablation": True,
                }
                with (
                    intervene(model, "no_momentum"),
                    ReadRoundingRecorder(model, root / "rounding.jsonl"),
                ):
                    fixed = run_fixed(
                        model, Tokenizer(), source, manifest, root / "fixed"
                    )
                    generated = run_generation(
                        model, Tokenizer(), source, manifest, root / "generated"
                    )
                self.assertEqual(fixed["status"], "finite_replay")
                self.assertEqual(fixed["trace_status"], "exported")
                self.assertEqual(generated["status"], "finite_generation")
                self.assertEqual(generated["trace_status"], "exported")
                self.assertTrue(
                    model.model.layers[0].taal.neural_memory.momentum_enabled
                )

    def test_explicit_inner_outer_derivatives_with_scale_residual_and_chunking(self):
        for B, C in ((1, 1), (2, 1), (1, 3), (2, 3)):
            for depth in (1, 2, 3):
                for scale, residual in ((True, False), (False, True), (True, True)):
                    with self.subTest(
                        B=B, C=C, depth=depth, scale=scale, residual=residual
                    ):
                        memory = NeuralMemory(
                            NeuralMemoryConfig(
                                dim=4,
                                depth=depth,
                                mlp_output_layernorm=True,
                                norm_learned_scale=scale,
                                norm_residual=residual,
                            )
                        ).train()
                        weights = memory.initial_state(B).weights
                        if scale:
                            weights["norm_gamma"] = (
                                torch.randn(B, 4) * 0.1
                            ).requires_grad_()
                        keys = torch.randn(B, C, 4, requires_grad=True)
                        values = torch.randn(B, C, 4, requires_grad=True)
                        strength = torch.rand(B, C, requires_grad=True)
                        actual = memory._chunk_gradient(
                            weights, keys, values, strength, trace_tokens=True
                        )
                        predicted = independent_prediction(
                            keys, weights, depth, True, residual, scale
                        )
                        loss = ((predicted - values).square().mean(-1) * strength).sum()
                        expected = torch.autograd.grad(
                            loss, tuple(weights.values()), create_graph=True
                        )
                        for name, gradient in zip(weights, expected):
                            torch.testing.assert_close(
                                actual.chunk_gradients[name],
                                gradient,
                                rtol=2e-4,
                                atol=1e-5,
                            )
                            torch.testing.assert_close(
                                actual.token_gradients[name].sum(1),
                                gradient,
                                rtol=2e-4,
                                atol=1e-5,
                            )
                        inputs = (*weights.values(), keys, values, strength)
                        first = sum(
                            g.square().sum() for g in actual.chunk_gradients.values()
                        )
                        second = sum(g.square().sum() for g in expected)
                        outer = torch.autograd.grad(first, inputs, retain_graph=True)
                        reference = torch.autograd.grad(second, inputs)
                        for a, b in zip(outer, reference):
                            torch.testing.assert_close(a, b, rtol=8e-4, atol=2e-4)

    def test_read_only_norm_keeps_writes_and_states_identical(self):
        raw = NeuralMemory(NeuralMemoryConfig(dim=8)).eval()
        read = NeuralMemory(NeuralMemoryConfig(dim=8, read_layernorm=True)).eval()
        read.load_state_dict(raw.state_dict())
        inputs = torch.randn(2, 9, 8)
        original, state = raw(inputs)
        normalized, next_state = read(inputs)
        torch.testing.assert_close(
            normalized, F.layer_norm(original, (8,), eps=1e-6), rtol=1e-5, atol=1e-5
        )
        for name in state.weights:
            torch.testing.assert_close(
                state.weights[name], next_state.weights[name], rtol=0, atol=0
            )
            torch.testing.assert_close(
                state.momentum[name], next_state.momentum[name], rtol=0, atol=0
            )

    def test_vector_fast_parameter_streaming_and_serialization(self):
        memory = NeuralMemory(
            NeuralMemoryConfig(
                dim=8,
                chunk_size=3,
                mlp_output_layernorm=True,
                norm_learned_scale=True,
                norm_residual=True,
            )
        ).eval()
        inputs = torch.randn(2, 8, 8)
        whole, state = memory(inputs)
        first, part = memory(inputs[:, :2])
        second, part = memory(inputs[:, 2:], state=part)
        torch.testing.assert_close(torch.cat([first, second], 1), whole)
        self.assertEqual(state.weights["norm_gamma"].shape, (2, 8))
        for name in state.weights:
            torch.testing.assert_close(state.weights[name], part.weights[name])
        config = TaalQwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            num_persistent_tokens=2,
            memory_mlp_output_layernorm=True,
            memory_norm_learned_scale=True,
            memory_norm_residual=True,
        )
        model = TaalQwen3ForCausalLM(config).eval()
        self.assertEqual(
            model.model.layers[0].taal.neural_memory.memory_mlp.norm_gamma.abs().sum(),
            0,
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = TaalQwen3ForCausalLM.from_pretrained(directory).eval()
            self.assertTrue(loaded.config.memory_norm_learned_scale)
            self.assertTrue(loaded.config.memory_norm_residual)
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            torch.testing.assert_close(
                model.prefill(ids).logits, loaded.prefill(ids).logits
            )

    def test_clipping_and_lr_are_independently_controlled(self):
        config = load_config(arm_config_path("clip_lr_ln"))
        model = torch.nn.Module()
        model.body = torch.nn.Linear(2, 2, bias=False)
        model.layer = torch.nn.Module()
        model.layer.taal = torch.nn.Module()
        model.layer.taal.residual_gate = torch.nn.Parameter(torch.tensor(0.05))
        model.body.weight.grad = torch.full_like(model.body.weight, 1e8)
        model.layer.taal.residual_gate.grad = torch.tensor(0.172)
        clipping = clip_training_gradients(model, config)
        self.assertAlmostEqual(
            model.layer.taal.residual_gate.grad.item(), 0.172, places=6
        )
        self.assertGreater(clipping.total_norm, 1e8)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"], 2
        )
        config.optimizer.gate_max_grad_norm = None
        model.body.weight.grad.fill_(1e8)
        model.layer.taal.residual_gate.grad = torch.tensor(0.172)
        clip_training_gradients(model, config)
        self.assertLess(model.layer.taal.residual_gate.grad.item(), 1e-8)

    def test_recipes_match_except_declared_interventions(self):
        reference = load_config(arm_config_path("baseline")).to_dict()
        for arm, changes in TRAINING_ARMS.items():
            actual = load_config(arm_config_path(arm)).to_dict()
            expected = copy.deepcopy(reference)
            for section, values in changes.items():
                expected[section].update(values)
            expected["wandb"]["name"] = actual["wandb"]["name"]
            self.assertEqual(actual, expected)
            self.assertTrue(actual["training"]["save_traces"])

    def test_independent_cache_reference_across_eviction_and_blocks(self):
        from_config = TaalQwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=4,
            num_persistent_tokens=2,
        )
        from_config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(from_config).eval()
        result = compare_cache(
            model,
            torch.tensor([[1, 4, 2, 5, 8, 3, 6, 2, 4, 1, 9, 2, 4]]),
            block_sizes=(1, 4),
        )
        for row in result.values():
            self.assertLess(row["logits"]["max_abs_error"], 1e-5)
            self.assertEqual(row["tail_argmax_agreement"], 1.0)

    def test_precision_probe_matches_independent_autograd(self):
        config = NeuralMemoryConfig(
            8, mlp_output_layernorm=True, norm_learned_scale=True, norm_residual=True
        )
        memory = NeuralMemory(config)
        state = memory.initial_state(1)
        result = write_comparison(
            config,
            state.weights,
            torch.randn(1, 1, 8),
            torch.randn(1, 1, 8),
            torch.full((1, 1), 0.1),
            torch.device("cpu"),
            "fp32",
        )
        for row in result["gradient"].values():
            self.assertLess(row["relative_l2_error"], 1e-4)

    def test_length_construction_retains_fact_query_and_pair(self):
        rows = [
            json.loads(line)
            for line in Path("data/taal_capacity_16_gap192.jsonl")
            .read_text()
            .splitlines()
        ][:4]
        for length in (206, 1024):
            result = length_examples(rows, [9, 10, 11], length)
            for old, new in zip(rows, result):
                self.assertEqual(len(new["input_ids"]), length)
                self.assertEqual(new["input_ids"][:32], old["input_ids"][:32])
                self.assertEqual(
                    new["input_ids"][new["final_query_position"] :],
                    old["input_ids"][186:],
                )
            a, b = result[:2]
            differences = [
                i
                for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"]))
                if x != y
            ]
            self.assertEqual(differences, [a["fact_position"]])

    def test_inference_interventions_restore_and_preserve_token_gradients(self):
        memory = NeuralMemory(NeuralMemoryConfig(8)).eval()
        state = memory.initial_state(1)
        keys, values = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
        strength = torch.ones(1, 2)
        original = memory._chunk_gradient(
            state.weights, keys, values, strength, trace_tokens=True
        )
        for arm in ("gradient_x0.1", "gradient_clip1"):
            changed = scaled_gradient(memory._chunk_gradient, arm)(
                state.weights, keys, values, strength, trace_tokens=True
            )
            for name in state.weights:
                torch.testing.assert_close(
                    changed.chunk_gradients[name], changed.token_gradients[name].sum(1)
                )
            if arm == "gradient_x0.1":
                for name in state.weights:
                    torch.testing.assert_close(
                        changed.chunk_gradients[name],
                        original.chunk_gradients[name] * 0.1,
                    )

    def test_new_normalization_modes_train_reload_and_export_token_traces(self):
        for arm in ("read_ln", "scale_ln", "residual_ln", "scale_residual_ln"):
            recipe = load_config(arm_config_path(arm))
            config = TaalQwen3Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                working_memory_size=4,
                max_persistent_kv_tokens=0,
                memory_dim=8,
                num_persistent_tokens=2,
                memory_read_layernorm=recipe.model.memory_read_layernorm,
                memory_mlp_output_layernorm=recipe.model.memory_mlp_output_layernorm,
                memory_norm_learned_scale=recipe.model.memory_norm_learned_scale,
                memory_norm_residual=recipe.model.memory_norm_residual,
            )
            model = TaalQwen3ForCausalLM(config).train()
            with torch.no_grad():
                for layer in model.model.layers:
                    layer.taal.residual_gate.fill_(0.05)
            optimizer = build_optimizer(model, recipe)
            ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
            output = model(ids, labels=ids, use_cache=False)
            output.loss.backward()
            clipping = clip_training_gradients(model, recipe)
            self.assertTrue(math.isfinite(clipping.total_norm))
            optimizer.step()
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                loaded = TaalQwen3ForCausalLM.from_pretrained(directory).eval()
                with (
                    TaalTraceRecorder(loaded, [0, 1], 9) as recorder,
                    ReadRoundingRecorder(loaded, Path(directory) / "rounding.jsonl"),
                ):
                    result = loaded.prefill(ids, execution_block_size=4)
                exporter = TaalTraceExporter(
                    Path(directory) / "traces", run_metadata={"layers": [0, 1]}
                )
                episode = TraceEpisode(
                    run_id="test",
                    example_id=arm,
                    condition_id="correct_full",
                    checkpoint=directory,
                    tokenizer_id="fixture",
                    tokens=make_trace_tokens(ids[0].tolist(), Tokenizer()),
                    writes=recorder.writes,
                    reads=recorder.reads,
                    internal_prefixes=recorder.internal_prefixes,
                )
                exporter.export(episode)
                self.assertEqual(len(recorder.reads), 18)
                self.assertEqual(
                    len(Path(directory, "rounding.jsonl").read_text().splitlines()), 6
                )

    def test_meta_loading_initializes_scale_and_preserves_host_weights(self):
        config = TaalQwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            num_persistent_tokens=2,
            memory_mlp_output_layernorm=True,
            memory_norm_learned_scale=True,
            memory_norm_residual=True,
        )
        base = Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            loaded = TaalQwen3ForCausalLM.from_pretrained(directory, config=config)
            self.assertEqual(
                loaded.model.layers[0]
                .taal.neural_memory.memory_mlp.norm_gamma.abs()
                .sum(),
                0,
            )
            torch.testing.assert_close(
                loaded.model.layers[0].mlp.gate_proj.weight,
                base.model.layers[0].mlp.gate_proj.weight,
            )
