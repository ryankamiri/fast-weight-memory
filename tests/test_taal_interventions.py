from dataclasses import replace
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from evaluation.taal.interventions import (
    MemoryComponentSources,
    audit_conditions,
    compose_memory_states,
    donor_label_scores,
    memory_execution_controls,
)
from evaluation.taal.state_audit import fork_session_state, summarize
from evaluation.taal.trace_contract import TraceEpisode, TraceToken
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder
from tests import test_taal_trace_export as trace_fixture
from evaluation.taal import state_audit


class TaalInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(105)
        self.model = trace_fixture.TaalTraceExportTests.model()
        for layer in self.model.model.layers:
            layer.taal.residual_gate.data.fill_(0.25)

    @torch.inference_mode()
    def prefixes(self):
        own = self.model.prefill(torch.tensor([[1, 2, 3, 4, 5]]), execution_block_size=3)
        donor = self.model.prefill(torch.tensor([[1, 7, 3, 4, 5]]), execution_block_size=3)
        return own.state, donor.state

    def test_suites_cover_the_planned_component_factorial(self):
        self.assertEqual(len(audit_conditions("standard")), 6)
        split = audit_conditions("weights_vs_rest")
        self.assertEqual({condition.name for condition in split}, {
            "correct_full", "reads_disabled", "weights_swapped", "remainder_swapped", "swapped",
        })
        factorial = audit_conditions("factorial")
        self.assertEqual(len(factorial), 9)
        self.assertEqual(len({condition.name for condition in factorial}), 9)
        self.assertEqual(sum(condition.uses_donor for condition in factorial), 7)
        with self.assertRaisesRegex(ValueError, "Unknown audit_suite"):
            audit_conditions("typo")

    @torch.inference_mode()
    def test_components_preserve_provenance_without_aliasing(self):
        own, donor = self.prefixes()
        for sources in (
            MemoryComponentSources(), MemoryComponentSources("donor", "donor", "donor"),
            MemoryComponentSources("donor", "own", "donor"),
        ):
            mixed = compose_memory_states(own.memory_states, donor.memory_states, sources)
            bank = {"own": own.memory_states, "donor": donor.memory_states}
            for layer, state in mixed.items():
                for component in ("weights", "momentum"):
                    expected = getattr(bank[getattr(sources, component)][layer], component)
                    for name, tensor in getattr(state, component).items():
                        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
                        self.assertNotEqual(tensor.data_ptr(), expected[name].data_ptr())
                for name in ("query_conv_history", "key_conv_history", "value_conv_history"):
                    expected = getattr(bank[sources.convolution][layer], name)
                    actual = getattr(state, name)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    @torch.inference_mode()
    def test_hybrid_own_and_donor_endpoints_reproduce_full_state_controls(self):
        own, donor = self.prefixes()
        ids = torch.tensor([[6, 7]])
        for sources, expected in (
            (MemoryComponentSources(), own.memory_states),
            (MemoryComponentSources("donor", "donor", "donor"), donor.memory_states),
        ):
            mixed = compose_memory_states(own.memory_states, donor.memory_states, sources)
            direct = self.model.prefill(ids, state=fork_session_state(own, expected),
                                       prepend_memory_tokens=False)
            hybrid = self.model.prefill(ids, state=fork_session_state(own, mixed),
                                       prepend_memory_tokens=False)
            torch.testing.assert_close(direct.logits, hybrid.logits, atol=0, rtol=0)
        self.assertEqual(own.past_key_values.layers[0].cumulative_length, 5)

    @torch.inference_mode()
    def test_components_reject_pending_writes_and_incompatible_histories(self):
        own, donor = self.prefixes()
        bad = dict(donor.memory_states)
        bad[0] = replace(bad[0], pending_count=1)
        with self.assertRaisesRegex(ValueError, "no pending writes"):
            compose_memory_states(own.memory_states, bad, MemoryComponentSources())
        bad[0] = replace(donor.memory_states[0], query_conv_history=None)
        with self.assertRaisesRegex(ValueError, "incompatible query_conv_history"):
            compose_memory_states(own.memory_states, bad, MemoryComponentSources())

    @torch.inference_mode()
    def test_persistent_mask_leaves_text_writes_and_inputs_enabled(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        recorder = TaalTraceRecorder(self.model, layers=[0, 1], token_count=5)
        with memory_execution_controls(self.model, persistent_writes_enabled=False), recorder:
            self.model.prefill(ids, execution_block_size=3)
        internal = [event for event in recorder.writes if event.internal_index is not None]
        text = [event for event in recorder.writes if event.position is not None]
        self.assertEqual(len(internal), 4)
        self.assertTrue(all(not event.write_enabled and event.proposed_write_norm == 0
                            and event.write_strength == 0 for event in internal))
        self.assertTrue(all(event.write_enabled for event in text))
        self.assertEqual(len(recorder.reads), 10)
        self.assertEqual([prefix.count for prefix in recorder.internal_prefixes], [2, 2])
        self.assertTrue(all(layer.taal.persistent_writes_enabled for layer in self.model.model.layers))

    @torch.inference_mode()
    def test_fixed_query_records_proposals_but_freezes_weights_and_momentum(self):
        own, _ = self.prefixes()
        ids = torch.tensor([[6, 7, 8]])
        recorder = TaalTraceRecorder(self.model, layers=[0, 1], token_count=3)
        with memory_execution_controls(self.model, updates_enabled=False), recorder:
            output = self.model.prefill(ids, state=fork_session_state(own, own.memory_states),
                                        execution_block_size=2, prepend_memory_tokens=False)
        for layer, before in own.memory_states.items():
            after = output.state.memory_states[layer]
            for name in before.weights:
                torch.testing.assert_close(after.weights[name], before.weights[name], atol=0, rtol=0)
                torch.testing.assert_close(after.momentum[name], before.momentum[name], atol=0, rtol=0)
            self.assertFalse(torch.equal(before.query_conv_history, after.query_conv_history))
        self.assertEqual(len(recorder.writes), 6)
        self.assertTrue(any(event.proposed_write_norm > 0 for event in recorder.writes))
        self.assertTrue(all(not event.updates_enabled and not event.chunk_boundary
                            and event.net_weight_change_norm == 0 for event in recorder.writes))
        self.assertTrue(all(event.state_timing == "pre-write" for event in recorder.reads))
        with memory_execution_controls(self.model, updates_enabled=False):
            untraced = self.model.prefill(ids, state=fork_session_state(own, own.memory_states),
                                         execution_block_size=2, prepend_memory_tokens=False)
        torch.testing.assert_close(output.logits, untraced.logits, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            exporter = TaalTraceExporter(Path(directory), run_metadata={"query_updates_enabled": False})
            exporter.export(TraceEpisode(
                run_id="fixed", example_id="test", condition_id="correct_full",
                checkpoint="test", tokenizer_id="test",
                tokens=[TraceToken(i, int(ids[0, i]), str(i), "prompt") for i in range(3)],
                writes=recorder.writes, reads=recorder.reads, internal_prefixes=[],
            ))

    @torch.inference_mode()
    def test_fixed_query_is_not_equivalent_to_masking_new_gradients(self):
        own, _ = self.prefixes()
        query = torch.tensor([[6, 7]])
        masked = self.model.prefill(query, state=fork_session_state(own, own.memory_states),
                                   write_mask=torch.zeros_like(query, dtype=torch.bool),
                                   prepend_memory_tokens=False)
        self.assertTrue(any(not torch.equal(weight, masked.state.memory_states[layer].weights[name])
                            for layer, state in own.memory_states.items()
                            for name, weight in state.weights.items()))

    def test_controls_restore_after_errors_and_reject_training(self):
        with self.assertRaisesRegex(RuntimeError, "test error"):
            with memory_execution_controls(self.model, persistent_writes_enabled=False, updates_enabled=False):
                raise RuntimeError("test error")
        self.assertTrue(all(layer.taal.persistent_writes_enabled and layer.taal.neural_memory.updates_enabled
                            for layer in self.model.model.layers))
        self.model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            with memory_execution_controls(self.model):
                pass

    def test_configs_enable_all_traces_and_have_distinct_experiment_policies(self):
        root = Path(__file__).resolve().parents[1]
        configs = root / "evaluation/configs/taal"
        for name, suite, persistent, updates in (
            ("persistent_writes_off", "standard", False, True),
            ("component_split", "weights_vs_rest", True, True),
            ("component_factorial", "factorial", True, True),
            ("fixed_query_weights", "weights_vs_rest", True, False),
        ):
            settings = yaml.safe_load((configs / f"conflict_{name}.yaml").read_text())
            self.assertTrue(settings["save_traces"])
            self.assertEqual(settings["audit_suite"], suite)
            self.assertEqual(settings["persistent_writes_enabled"], persistent)
            self.assertEqual(settings["query_updates_enabled"], updates)
            self.assertEqual(settings["dataset"]["end"], 16)
        launcher = (root / "evaluation/sbatch/taal/fs_qwen_conflict_carriers_short.sbatch").read_text()
        for text in ("--partition=gpu-short", "--gres=gpu:h200:1", "--time=02:00:00",
                     "utils.explorer_preflight", "EXPECTED_COMMIT"):
            self.assertIn(text, launcher)

    def test_donor_scores_measure_label_preference_and_reject_nonfinite_logits(self):
        scores = donor_label_scores(torch.tensor([0.0, 3.0, 1.0]), 1, 2)
        self.assertAlmostEqual(scores["own_minus_donor_label_log_probability"], 2.0)
        self.assertFalse(scores["donor_target_vocabulary_top_1"])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            donor_label_scores(torch.tensor([0.0, float("nan"), 1.0]), 1, 2)

    def test_runner_exports_and_resumes_every_experiment_on_cpu_fixture(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU test fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp,
        )
        examples = [{
            "example_id": f"pair-{i}", "fact_id": i, "record_name": "same record",
            "answer": str(9 + i), "condition": "micro_conflict", "query_variant": "exact",
            "input_ids": [1, 2 + i, 3, 4, 5], "final_query_position": 3,
            "target_token_id": 9 + i, "candidate_token_ids": [9, 10],
        } for i in range(2)]
        for suite, persistent, updates in (
            ("standard", False, True), ("weights_vs_rest", True, True),
            ("factorial", True, True), ("weights_vs_rest", True, False),
        ):
            with self.subTest(suite=suite, persistent=persistent, updates=updates), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset = root / "data.jsonl"
                dataset.write_text("\n".join(json.dumps(example) for example in examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({
                    "tokenizer": "CPU fixture", "tokenizer_revision": "a" * 40,
                }))
                settings = {
                    "dataset": {"path": str(dataset), "start": 0, "end": 2,
                                "conditions": ["micro_conflict"], "query_variants": ["exact"]},
                    "seed": 42, "execution_block_size": 3, "save_traces": True,
                    "swap_group_field": "record_name", "audit_suite": suite,
                    "persistent_writes_enabled": persistent, "query_updates_enabled": updates,
                }
                config = root / "config.yaml"
                config.write_text(yaml.safe_dump(settings))
                output = root / "out"
                argv = ["audit", "--config", str(config), "--checkpoint", str(root), "--output-dir", str(output)]
                with (
                    patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples),
                    patch.object(state_audit, "checkpoint_fingerprint", return_value="fixture-sha"),
                    patch.object(state_audit, "load_model", return_value=self.model),
                    patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                    redirect_stdout(io.StringIO()),
                ):
                    state_audit.main()
                    # Resume after a missing final row: completed snapshots and
                    # traces must be reusable without rerunning the whole bank.
                    rows_path = output / "results.jsonl"
                    rows = rows_path.read_text().splitlines()
                    rows_path.write_text("\n".join(rows[:-1]) + "\n")
                    state_audit.main()
                    state_audit.main()
                conditions = audit_conditions(suite)
                manifest = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 2 * len(conditions))
                self.assertEqual(len(manifest["comparisons"]), 2 * (len(conditions) - 1))
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(len(summary["label_preferences"]), len(conditions))
                for relative in manifest["episodes"].values():
                    with gzip.open(output / "memory_traces" / relative, "rt") as file:
                        episode = json.load(file)
                    self.assertEqual(len(episode["tokens"]), 6)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(episode["metadata"]["query_updates_enabled"], updates)
                    if not updates:
                        query_events = [event for event in episode["writes"]
                                        if event["position"] is not None and event["position"] >= 3]
                        self.assertEqual(len(query_events), 6)
                        self.assertTrue(all(not event["updates_enabled"] and event["net_weight_change_norm"] == 0
                                            for event in query_events))


if __name__ == "__main__":
    unittest.main()
