import json
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.taal.decoder_diagnostic import DecoderRecorder, FixedReplaySource, load_replay_source, replay_fixed_tokens
from evaluation.taal.longmemeval_decoder_replay import compare_reference, run_replay_arm
from evaluation.taal.numerical_diagnostic import BlockMonitor, NumericalFailure
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens
from tests.test_taal_longmemeval import Tokenizer


class DecoderDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        torch.set_num_threads(1)
        config = TaalQwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            working_memory_size=4, max_persistent_kv_tokens=0, memory_dim=8, memory_depth=2,
            memory_conv_kernel_size=2, memory_chunk_size=1, num_persistent_tokens=2)
        config._attn_implementation = "sdpa"
        self.model = TaalQwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for layer in self.model.model.layers:
                layer.taal.residual_gate.fill_(0.1)
        self.prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        self.forced = [20, 20, 20, 20, 20]

    def assert_clean(self):
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.model.modules()))

    def test_hooks_preserve_exact_logits_memory_and_kv_both_precisions_and_scales(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.model.to(dtype)
            for scale in (1.0, 0.0):
                expected, expected_predictions = replay_fixed_tokens(self.model, self.prompt, self.forced, 4, scale)
                with tempfile.TemporaryDirectory() as directory:
                    recorder = DecoderRecorder(self.model, Path(directory) / "decoder", [1, 2, 3], 7, [10, 11])
                    with recorder, BlockMonitor(self.model, 9):
                        actual, predictions = replay_fixed_tokens(self.model, self.prompt, self.forced, 4, scale)
                    self.assertEqual(predictions, expected_predictions)
                    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                    self.assertEqual(actual.state.tokens_seen, 14)
                    for layer, state in actual.state.memory_states.items():
                        reference = expected.state.memory_states[layer]
                        for field in ("weights", "momentum"):
                            for name, value in getattr(state, field).items():
                                torch.testing.assert_close(value, getattr(reference, field)[name], atol=0, rtol=0)
                        left = actual.state.past_key_values.layers[layer]
                        right = expected.state.past_key_values.layers[layer]
                        torch.testing.assert_close(left.keys, right.keys, atol=0, rtol=0)
                        torch.testing.assert_close(left.values, right.values, atol=0, rtol=0)
                    self.assertEqual(len(recorder.rows), 21)
                    for row in recorder.rows:
                        components = row["components"]
                        self.assertEqual(components["attention_recomposition_error"]["norm_finite"], 0)
                        self.assertEqual(components["mlp_recomposition_error"]["norm_finite"], 0)
                        self.assertTrue(all(value["nan"] == value["inf"] == 0 for value in components.values()))
                        self.assertEqual(row["cache"]["allowed_positions"], list(range(max(0, row["position"] - 3), row["position"] + 1)))
                        self.assertIn("attention_context", components)
                        self.assertIn("mlp_product", components)
                        if scale == 0:
                            self.assertEqual(components["memory_delta"]["norm_finite"], 0)
                    saved = torch.load(Path(directory) / "decoder/layer-2-position-10.pt", weights_only=True)
                    self.assertEqual(saved["token_ids"], [20])
                    self.assertEqual(saved["cache_position"].tolist(), [10])
                    self.assertEqual(saved["key_positions"].tolist(), [7, 8, 9, 10])
                    weights = torch.load(Path(directory) / "decoder/decoder-2.pt", weights_only=True)
                    self.assertIn("mlp.down_proj.weight", weights)
                    self.assertFalse(any(name.startswith("taal.") for name in weights))
                    self.assert_clean()

    def test_forced_tokens_ignore_predictions_and_start_fresh(self):
        calls = []
        def capture(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            calls.extend(ids[0].tolist())
        handle = self.model.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            for scale in (1.0, 0.0):
                calls.clear()
                _, predictions = replay_fixed_tokens(self.model, self.prompt, self.forced, 4, scale)
                self.assertEqual(calls, self.prompt[0].tolist() + self.forced)
                self.assertEqual([row["forced_token_id"] for row in predictions], self.forced)
                self.assertTrue(any(row["predicted_token_id"] != 20 for row in predictions))
        finally:
            handle.remove()

    def test_compact_traces_coexist_and_internal_controls_align(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = TaalTraceRecorder(self.model, [0, 1, 2, 3], 14)
            recorder = DecoderRecorder(self.model, root / "decoder", [1, 2, 3], 0)
            with trace, recorder:
                replay_fixed_tokens(self.model, self.prompt, self.forced, 4, 0.0)
            self.assertEqual(len(recorder.rows), 42)
            self.assertEqual(len(trace.reads), 56)
            self.assertEqual(len(trace.writes), 64)
            exporter = TaalTraceExporter(root / "memory_traces", run_metadata={"evaluation": "test"})
            exporter.export(TraceEpisode(
                run_id="test", example_id="forced", condition_id="reads_off", checkpoint="test", tokenizer_id="test",
                tokens=make_trace_tokens(self.prompt[0].tolist(), Tokenizer())
                    + make_trace_tokens(self.forced, Tokenizer(), offset=9, phase="generated"),
                writes=trace.writes, reads=trace.reads, internal_prefixes=trace.internal_prefixes,
                outcome={"forced_token_ids": self.forced}, metadata={"forced_continuation": True},
            ))
            self.assertEqual(len(exporter.manifest["episodes"]), 1)
            self.assert_clean()

    def test_attention_and_mlp_perturbations_land_in_the_correct_component(self):
        for target, expected in (("self_attn", "attention_output"), ("mlp", "mlp_output")):
            module = getattr(self.model.model.layers[2], target)
            def increase(module, args, output):
                if isinstance(output, tuple):
                    return (output[0] + 100, *output[1:])
                return output + 100
            handle = module.register_forward_hook(increase)
            try:
                with tempfile.TemporaryDirectory() as directory:
                    recorder = DecoderRecorder(self.model, Path(directory) / "decoder", [2], 9)
                    with recorder:
                        replay_fixed_tokens(self.model, self.prompt, [20], 4, 0.0)
                    row = recorder.rows[0]["components"]
                    self.assertGreater(row[expected]["norm_finite"], 390)
                    self.assertEqual(row["attention_recomposition_error"]["norm_finite"], 0)
                    self.assertEqual(row["mlp_recomposition_error"]["norm_finite"], 0)
            finally:
                handle.remove()

    def test_failure_persists_partial_components_and_removes_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "decoder"
            recorder = DecoderRecorder(self.model, folder, [2], 0)
            with self.assertRaises(NumericalFailure):
                with recorder, BlockMonitor(self.model, 9):
                    with torch.no_grad():
                        self.model.model.layers[2].mlp.down_proj.weight.fill_(float("inf"))
                    replay_fixed_tokens(self.model, self.prompt, [20], 4, 1.0)
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["exception"], "NumericalFailure")
            self.assertIn("mlp_output", summary["partial_components"]["2"])
            self.assert_clean()

    def test_source_validation_and_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reads-1/updates").mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({"commit": "test"}))
            (root / "example.json").write_text(json.dumps({"input_ids": [1, 2], "prompt_length": 2}))
            rows = [{"layer": layer, "position": position, "token_id": token,
                     "metrics": {"hidden_in": 2., "hidden_out": 3.}}
                    for layer in range(3) for position, token in enumerate([1, 2, 20, 20])]
            path = root / "reads-1/updates/updates.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            source = load_replay_source(root, 2, [2])
            self.assertEqual(source.continuation, [20, 20])
            with self.assertRaisesRegex(ValueError, "every requested"):
                load_replay_source(root, 3, [2])
            rows[0]["token_id"] = 9
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_replay_source(root, 2, [2])
            row = {"layer": 2, "position": 0, "components": {
                "memory_input": {"norm_finite": 2.}, "memory_output": {"norm_finite": 3.}}}
            self.assertTrue(compare_reference([row], source.reference)["matched_within_1e-5"])
            row["components"]["memory_input"]["norm_finite"] = 4.
            self.assertFalse(compare_reference([row], source.reference)["matched_within_1e-5"])

    def test_actual_runner_writes_two_complete_forced_trace_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = DecoderRecorder(self.model, root / "reference", [1, 2, 3], 7)
            with recorder:
                replay_fixed_tokens(self.model, self.prompt, self.forced, 4, 1.0)
            reference = {(row["layer"], row["position"]): {
                "hidden_in": row["components"]["memory_input"]["norm_finite"],
                "hidden_out": row["components"]["memory_output"]["norm_finite"],
            } for row in recorder.rows}
            source = FixedReplaySource(
                {"metadata": {"tokenizer": "test", "tokenizer_revision": "test"}},
                {"question_id": "test", "prompt_length": 9, "input_ids": self.prompt[0].tolist()},
                self.forced, reference)
            manifest = {"checkpoint": "test", "source_run": "test", "layers": [1, 2, 3],
                        "trace_start": 7, "snapshot_positions": [10], "execution_block_size": 4}
            for scale in (1.0, 0.0):
                arm = root / f"reads-{scale:g}"
                report = run_replay_arm(self.model, Tokenizer(), source, manifest, arm, scale)
                self.assertEqual(report["status"], "finite_replay")
                self.assertEqual(report["trace_status"], "exported")
                self.assertEqual(report["tokens_seen"], 14)
                saved = json.loads((arm / "report.json").read_text())
                self.assertEqual(saved["forced_token_count"], 5)
                trace_manifest = json.loads((arm / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(trace_manifest["episodes"]), 1)
                if scale == 1:
                    self.assertTrue(saved["reference_comparison"]["matched_within_1e-5"])
            self.assert_clean()

    def test_d_memory_kv_and_decoder_geometry(self):
        config = TaalQwen3Config(vocab_size=32, hidden_size=1024, intermediate_size=3072,
            num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            working_memory_size=32, max_persistent_kv_tokens=0, memory_dim=128, memory_depth=2,
            memory_conv_kernel_size=4, memory_chunk_size=1, num_persistent_tokens=8)
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).bfloat16().eval()
        prompt = torch.randint(0, 32, (1, 35))
        with tempfile.TemporaryDirectory() as directory:
            recorder = DecoderRecorder(model, Path(directory) / "decoder", [1, 2, 3], 31, [35])
            with recorder:
                output, predictions = replay_fixed_tokens(model, prompt, [20, 20], 32, 1.0)
            self.assertEqual(output.state.tokens_seen, 37)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(recorder.rows), 18)
            row = next(row for row in recorder.rows if row["layer"] == 2 and row["position"] == 35)
            self.assertEqual(row["components"]["attention_q_proj"]["shape"], [2048])
            self.assertEqual(row["components"]["attention_k_proj"]["shape"], [1024])
            self.assertEqual(row["components"]["mlp_product"]["shape"], [3072])
            self.assertEqual(row["cache"]["allowed_positions"], list(range(4, 36)))


if __name__ == "__main__":
    unittest.main()
