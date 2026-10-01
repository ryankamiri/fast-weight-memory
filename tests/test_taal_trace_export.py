import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.taal.trace_contract import TraceComparison, TraceEpisode, TraceToken
from evaluation.taal.trace_export import (
    TaalTraceExporter,
    TaalTraceRecorder,
    trace_greedy_prediction,
)


class TaalTraceExportTests(unittest.TestCase):
    @staticmethod
    def model(memory_chunk_size: int = 1):
        config = TaalQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=2,
            memory_chunk_size=memory_chunk_size,
            num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        return TaalQwen3ForCausalLM(config).eval()

    @torch.inference_mode()
    def test_trace_records_internal_writes_and_does_not_change_outputs(self):
        torch.manual_seed(91)
        model = self.model()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.taal.residual_gate.fill_(0.7)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        baseline = model.prefill(ids, execution_block_size=3, memory_read_scale=1.0)

        recorder = TaalTraceRecorder(model, layers=[0, 1], token_count=5)
        with recorder:
            traced = model.prefill(ids, execution_block_size=3, memory_read_scale=1.0)

        torch.testing.assert_close(traced.logits, baseline.logits, atol=0, rtol=0)
        for layer in (0, 1):
            for name in baseline.state.memory_states[layer].weights:
                torch.testing.assert_close(
                    traced.state.memory_states[layer].weights[name],
                    baseline.state.memory_states[layer].weights[name],
                    atol=0,
                    rtol=0,
                )
        self.assertEqual(len(recorder.writes), 14)
        self.assertEqual(len(recorder.reads), 10)
        self.assertEqual(len(recorder.internal_prefixes), 2)
        self.assertEqual({event.before_position for event in recorder.internal_prefixes}, {0})
        self.assertEqual(
            [event.internal_index for event in recorder.writes[:2]],
            [0, 1],
        )
        self.assertTrue(all(event.state_timing == "post-write" for event in recorder.reads))
        self.assertIsNone(model.model.layers[0].taal.trace_observer)

    @torch.inference_mode()
    def test_masked_write_is_zero_but_state_can_still_move(self):
        torch.manual_seed(92)
        model = self.model()
        ids = torch.tensor([[1, 2, 3]])
        recorder = TaalTraceRecorder(model, layers=[0], token_count=3)
        with recorder:
            model.prefill(
                ids,
                execution_block_size=2,
                write_mask=torch.tensor([[True, False, True]]),
                memory_read_scale=0.0,
            )
        masked = next(event for event in recorder.writes if event.position == 1)
        self.assertFalse(masked.write_enabled)
        self.assertEqual(masked.proposed_write_norm, 0)
        self.assertEqual(masked.write_strength, 0)
        self.assertTrue(all(event.injected_norm == 0 for event in recorder.reads))
        self.assertTrue(all(not event.enabled for event in recorder.reads))

    @torch.inference_mode()
    def test_chunked_trace_preserves_outputs_and_marks_boundary(self):
        torch.manual_seed(94)
        model = self.model(memory_chunk_size=4)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        baseline = model.prefill(ids, execution_block_size=3)

        recorder = TaalTraceRecorder(model, layers=[0], token_count=5)
        with recorder:
            traced = model.prefill(ids, execution_block_size=3)

        torch.testing.assert_close(traced.logits, baseline.logits, atol=0, rtol=0)
        for name, weight in baseline.state.memory_states[0].weights.items():
            torch.testing.assert_close(
                traced.state.memory_states[0].weights[name], weight, atol=0, rtol=0
            )
        self.assertEqual(len(recorder.writes), 7)
        self.assertEqual(len(recorder.reads), 5)
        self.assertEqual([event.position for event in recorder.writes], [
            None, None, 0, 1, 2, 3, 4,
        ])
        self.assertEqual([event.chunk_boundary for event in recorder.writes], [
            False, False, False, True, False, False, False,
        ])
        self.assertEqual([event.state_timing for event in recorder.reads], [
            "pre-write", "post-write", "pre-write", "pre-write", "pre-write",
        ])
        self.assertEqual(recorder.writes[0].net_weight_change_norm, 0)
        self.assertGreater(recorder.writes[3].net_weight_change_norm, 0)
        self.assertIsNone(recorder.writes[3].other_movement_norm)
        self.assertEqual(recorder.internal_prefixes[0].net_weight_change_norm, 0)

    @torch.inference_mode()
    def test_token_gradients_sum_to_chunk_gradient(self):
        torch.manual_seed(95)
        memory = self.model(memory_chunk_size=4).model.layers[0].taal.neural_memory
        weights = memory.initial_state(1).weights
        keys = torch.randn(1, 4, memory.config.dim)
        values = torch.randn_like(keys)
        strength = torch.rand(1, 4)
        untraced = memory._chunk_gradient(weights, keys, values, strength)
        result = memory._chunk_gradient(
            weights, keys, values, strength, trace_tokens=True
        )
        self.assertIsNone(untraced.token_gradients)
        self.assertIsNotNone(result.token_gradients)
        for name in result.chunk_gradients:
            torch.testing.assert_close(
                untraced.chunk_gradients[name], result.chunk_gradients[name]
            )
            torch.testing.assert_close(
                result.token_gradients[name].sum(dim=1),
                result.chunk_gradients[name],
                atol=1e-6,
                rtol=1e-5,
            )

    def test_export_round_trip_and_contract_rejects_unaligned_event(self):
        torch.manual_seed(93)
        model = self.model()
        recorder = TaalTraceRecorder(model, layers=[0], token_count=2)
        with torch.inference_mode(), recorder:
            model.prefill(torch.tensor([[1, 2]]), execution_block_size=1)
        episode = TraceEpisode(
            run_id="pilot",
            example_id="example/1",
            condition_id="correct_full",
            checkpoint="local-checkpoint",
            tokenizer_id="test-tokenizer",
            tokens=[
                TraceToken(position=index, token_id=index + 1, text=str(index + 1), phase="prompt")
                for index in range(2)
            ],
            writes=recorder.writes,
            reads=recorder.reads,
            internal_prefixes=recorder.internal_prefixes,
            outcome={"score": 1},
        )
        with TemporaryDirectory() as directory:
            exporter = TaalTraceExporter(Path(directory), run_metadata={"checkpoint": "local-checkpoint"})
            path = exporter.export(episode)
            with gzip.open(path, "rt", encoding="utf-8") as file:
                saved = json.load(file)
            self.assertEqual(saved["schema"], "taal-memory-trace/v1")
            self.assertEqual(saved["outcome"], {"score": 1})
            self.assertEqual(len(saved["internal_prefixes"]), 1)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertIn("example/1/correct_full", manifest["episodes"])
            comparison = TraceComparison(
                run_id="pilot",
                example_id="example/1",
                baseline_condition="correct_full",
                variant_condition="reads_disabled",
                intervention="read_scale",
                scope="whole_query",
                identical_text_prefix=True,
                same_starting_kv=True,
                same_starting_memory=True,
                scored_position=1,
                scored_token_id=3,
                baseline_log_probability=-0.5,
                variant_log_probability=-1.0,
                difference_log_probability=0.5,
            )
            comparison_path = exporter.export_comparison(comparison)
            with gzip.open(comparison_path, "rt", encoding="utf-8") as file:
                saved_comparison = json.load(file)
            self.assertEqual(saved_comparison["difference_log_probability"], 0.5)
            self.assertEqual(saved_comparison["scope"], "whole_query")

        episode.tokens = episode.tokens[:1]
        with self.assertRaisesRegex(ValueError, "absent visible token"):
            episode.to_dict()

    def test_accepts_non_unit_memory_chunks(self):
        model = self.model(memory_chunk_size=2)
        recorder = TaalTraceRecorder(model, layers=[0], token_count=2)
        self.assertEqual(recorder.recorders[0].chunk_size, 2)

    @torch.inference_mode()
    def test_greedy_prediction_replay_records_events_after_scoring(self):
        model = self.model()
        prompt = model.prefill(torch.tensor([[1, 2, 3]]), execution_block_size=2)
        chosen = int(prompt.logits[0, -1].argmax().item())

        class Tokenizer:
            @staticmethod
            def decode(ids, **_kwargs):
                return f" token-{ids[0]}"

        traced = trace_greedy_prediction(
            model,
            Tokenizer(),
            token_id=chosen,
            position=3,
            state=prompt.state,
            layers=[0, 1],
            memory_read_scale=1.0,
        )
        self.assertEqual(traced.token.position, 3)
        self.assertEqual(traced.token.phase, "generated")
        self.assertEqual(traced.token.token_id, chosen)
        self.assertEqual(traced.token.text, f" token-{chosen}")
        self.assertEqual({event.position for event in traced.writes}, {3})
        self.assertEqual({event.position for event in traced.reads}, {3})
        self.assertEqual(len(traced.writes), 2)
        self.assertEqual(len(traced.reads), 2)
        self.assertEqual(traced.internal_prefixes, [])

    @torch.inference_mode()
    def test_resumed_query_keeps_absolute_positions_without_repeating_prefix(self):
        torch.manual_seed(94)
        model = self.model()
        prefix = TaalTraceRecorder(model, layers=[0], token_count=3)
        with prefix:
            prefix_output = model.prefill(
                torch.tensor([[1, 2, 3]]), execution_block_size=2
            )
        query = TaalTraceRecorder(model, layers=[0], token_count=2, position_offset=3)
        with query:
            model.prefill(
                torch.tensor([[4, 5]]),
                state=prefix_output.state,
                execution_block_size=1,
                prepend_memory_tokens=False,
            )
        self.assertEqual([event.position for event in query.reads], [3, 4])
        self.assertEqual(len(prefix.internal_prefixes), 1)
        self.assertEqual(query.internal_prefixes, [])
        self.assertEqual(
            [event.position for event in prefix.reads + query.reads],
            list(range(5)),
        )


if __name__ == "__main__":
    unittest.main()
