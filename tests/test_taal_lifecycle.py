from contextlib import redirect_stdout
from itertools import product
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from evaluation.taal import state_audit
from evaluation.taal.interventions import audit_conditions, memory_execution_controls
from evaluation.taal.lifecycle import lifecycle_boundaries, lifecycle_stores, save_lifecycle_prefixes
from evaluation.taal.prefix_store import PrefixStore
from evaluation.taal.state_audit import fork_session_state
from tests import test_taal_trace_export as trace_fixture


class LifecycleTokenizer:
    def encode(self, text, **kwargs):
        return [4, 5] if text.startswith(".\nAssistant:") else [1, 2]

    def decode(self, ids, **kwargs):
        return "".join(str(token) for token in ids)


def examples():
    return [{
        "example_id": f"pair-{i}", "fact_id": i, "record_name": "Record R0000",
        "answer": str(9 + i), "condition": "micro_conflict", "query_variant": "exact",
        "input_ids": [1, 2, 9 + i, 4, 5, 6, 7, 6, 7, 6, 7, 8, 2],
        "fact_position": 2, "final_query_position": 11,
        "target_token_id": 9 + i, "candidate_token_ids": [9, 10],
    } for i in range(2)]


class TaalLifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(202)
        self.model = trace_fixture.TaalTraceExportTests.model()
        for layer in self.model.model.layers:
            layer.taal.residual_gate.data.fill_(0.25)
        self.tokenizer = LifecycleTokenizer()

    def test_config_suites_and_counts_are_derived(self):
        for suite, count in (("encoding", 12), ("retention", 13), ("query_transitions", 19)):
            conditions = audit_conditions(suite)
            self.assertEqual(len(conditions), count)
            self.assertEqual(len({condition.name for condition in conditions}), count)
            config = yaml.safe_load(Path(f"evaluation/configs/taal/conflict_{suite}.yaml").read_text())
            self.assertEqual(config["audit_suite"], suite)
            self.assertIs(config["save_traces"], True)
            self.assertNotIn("expected_examples", config)
        policies = {(
            condition.fresh_writes_enabled, condition.momentum_enabled,
            condition.forgetting_enabled,
        ) for condition in audit_conditions("query_transitions") if condition.updates_enabled}
        self.assertEqual(policies, set(product((True, False), repeat=3)))

    @torch.inference_mode()
    def test_each_transition_matches_its_equation_and_restores_controls(self):
        memory = self.model.model.layers[0].taal.neural_memory
        state = memory.initial_state(1)
        for value in state.momentum.values():
            value.fill_(0.03)
        inputs = torch.randn(1, 1, memory.config.dim)
        keys = torch.randn_like(inputs)
        values = torch.randn_like(inputs)
        strength = memory.write_strength_projection(inputs).sigmoid().reshape(1, 1)
        forget = memory.forget_projection(inputs[:, 0]).sigmoid().reshape(1, 1, 1)
        retention = memory.momentum_projection(inputs[:, 0]).sigmoid().reshape(1, 1, 1)
        gradient = memory._chunk_gradient(state.weights, keys, values, strength).chunk_gradients
        for fresh, momentum, forgetting in product((True, False), repeat=3):
            mask = torch.full((1, 1), fresh, dtype=torch.bool)
            with memory_execution_controls(
                self.model, momentum_enabled=momentum, forgetting_enabled=forgetting,
            ):
                actual = memory._update(state, inputs, keys, values, mask)
            for name, weight in state.weights.items():
                expected_momentum = (
                    (retention if momentum else 0) * state.momentum[name]
                    - (gradient[name] if fresh else 0)
                )
                expected_weight = (1 - (forget if forgetting else 0)) * weight + expected_momentum
                torch.testing.assert_close(actual.momentum[name], expected_momentum)
                torch.testing.assert_close(actual.weights[name], expected_weight)
            self.assertTrue(memory.momentum_enabled and memory.forgetting_enabled)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with memory_execution_controls(self.model, momentum_enabled=False, forgetting_enabled=False):
                raise RuntimeError("restore")
        self.assertTrue(memory.momentum_enabled and memory.forgetting_enabled)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            memory.momentum_enabled = False
            memory._update(state, inputs, keys, values, torch.ones(1, 1, dtype=torch.bool))

    @torch.inference_mode()
    def test_snapshot_boundaries_frozen_gap_and_execution_partition_parity(self):
        example = examples()[0]
        self.assertEqual(lifecycle_boundaries(example, self.tokenizer), {
            "label": 3, "fact": 5, "middle": 8, "final": 11,
        })
        bad = {**example, "fact_position": 1}
        with self.assertRaisesRegex(ValueError, "template"):
            lifecycle_boundaries(bad, self.tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = PrefixStore(root / "final", {})
            stages = lifecycle_stores(root / "stages", {})
            save_lifecycle_prefixes(
                self.model, self.tokenizer, example, final, stages,
                device="cpu", layers=[0, 1], execution_block_size=3,
                persistent_writes_enabled=True, capture_gap_frozen=True,
            )
            direct = self.model.prefill(torch.tensor([example["input_ids"][:11]]), execution_block_size=3)
            saved = final.load(example, "cpu")
            fact = stages["fact"].load(example, "cpu")
            frozen = stages["gap_frozen"].load(example, "cpu")
            self.assertEqual(stages["label"].load(example, "cpu").state.tokens_seen, 3)
            self.assertEqual(stages["middle"].load(example, "cpu").state.tokens_seen, 8)
            self.assertEqual(frozen.state.tokens_seen, 11)
            self.assertEqual(len(saved.trace.writes), 2 * (11 + 2))
            self.assertEqual(len(saved.trace.internal_prefixes), 2)
            for layer, memory in saved.state.memory_states.items():
                for name, weight in memory.weights.items():
                    torch.testing.assert_close(weight, direct.state.memory_states[layer].weights[name], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(fact.state.memory_states[layer].weights[name], frozen.state.memory_states[layer].weights[name], atol=0, rtol=0)
                    torch.testing.assert_close(fact.state.memory_states[layer].momentum[name], frozen.state.memory_states[layer].momentum[name], atol=0, rtol=0)
            query = torch.tensor([example["input_ids"][11:]])
            split_answer = self.model.prefill(query, state=saved.state, prepend_memory_tokens=False)
            direct_answer = self.model.prefill(query, state=direct.state, prepend_memory_tokens=False)
            torch.testing.assert_close(split_answer.logits, direct_answer.logits, atol=1e-6, rtol=1e-5)
            gap_events = [event for event in frozen.trace.writes if event.position is not None and event.position >= 5]
            self.assertTrue(all(not event.updates_enabled and event.net_weight_change_norm == 0 for event in gap_events))

    @torch.inference_mode()
    def test_all_off_matches_fixed_reads_and_reads_off_is_policy_independent(self):
        prefix = self.model.prefill(torch.tensor([[1, 2, 3, 4, 5]]))
        query = torch.tensor([[6, 7]])
        with memory_execution_controls(self.model, updates_enabled=False):
            fixed = self.model.prefill(query, state=fork_session_state(prefix.state, prefix.state.memory_states), prepend_memory_tokens=False)
        with memory_execution_controls(self.model, momentum_enabled=False, forgetting_enabled=False):
            all_off = self.model.prefill(query, write_mask=torch.zeros_like(query, dtype=torch.bool),
                                        state=fork_session_state(prefix.state, prefix.state.memory_states), prepend_memory_tokens=False)
        torch.testing.assert_close(fixed.logits, all_off.logits, atol=0, rtol=0)
        baseline = self.model.prefill(query, state=fork_session_state(prefix.state, prefix.state.memory_states),
                                     memory_read_scale=0.0, prepend_memory_tokens=False)
        for fresh, momentum, forgetting in product((True, False), repeat=3):
            with memory_execution_controls(self.model, momentum_enabled=momentum, forgetting_enabled=forgetting):
                output = self.model.prefill(query, write_mask=torch.full_like(query, fresh, dtype=torch.bool),
                    state=fork_session_state(prefix.state, prefix.state.memory_states), memory_read_scale=0.0, prepend_memory_tokens=False)
            torch.testing.assert_close(baseline.logits, output.logits, atol=0, rtol=0)

    def test_full_runner_traces_scoring_and_resume_for_all_three_suites(self):
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, bool=torch.bool, full=torch.full, logsumexp=torch.logsumexp,
        )
        for suite in ("encoding", "retention", "query_transitions"):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "data.jsonl"
                dataset.write_text("\n".join(json.dumps(row) for row in examples()))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": "fixture", "tokenizer_revision": "a" * 40}))
                settings = {
                    "dataset": {"path": str(dataset), "start": 0, "end": 2, "conditions": ["micro_conflict"], "query_variants": ["exact"]},
                    "seed": 42, "execution_block_size": 3, "save_traces": True,
                    "swap_group_field": "record_name", "audit_suite": suite,
                }
                config = root / "config.yaml"
                config.write_text(yaml.safe_dump(settings))
                output = root / "out"
                with (
                    patch("sys.argv", ["audit", "--config", str(config), "--checkpoint", str(root), "--output-dir", str(output)]),
                    patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples()),
                    patch.object(state_audit, "checkpoint_fingerprint", return_value="fixture"),
                    patch.object(state_audit, "load_model", return_value=self.model),
                    patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                    redirect_stdout(io.StringIO()),
                ):
                    state_audit.main()
                    rows_path = output / "results.jsonl"
                    rows = rows_path.read_text().splitlines()
                    rows_path.write_text("\n".join(rows[:-1]) + "\n")
                    state_audit.main()
                    state_audit.main()
                rows = [json.loads(row) for row in rows_path.read_text().splitlines()]
                self.assertEqual(len(rows), 2 * len(audit_conditions(suite)))
                self.assertEqual(len({row["evaluation_id"] for row in rows}), len(rows))
                manifest = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), len(rows))
                self.assertEqual(len(manifest["comparisons"]), len(rows) - 2)
                summary = json.loads((output / "summary.json").read_text())
                self.assertTrue(summary["matched_memory_contrasts"])
                for relative in manifest["episodes"].values():
                    with gzip.open(output / "memory_traces" / relative, "rt") as file:
                        episode = json.load(file)
                    self.assertEqual(len(episode["tokens"]), 14)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(episode["metadata"]["memory_intervention_at_position"], 11)
                    events = [event for event in episode["writes"] if event["position"] is not None and event["position"] >= 11]
                    self.assertEqual(len(events), 6)
                    if not episode["metadata"]["query_updates_enabled"]:
                        self.assertTrue(all(event["net_weight_change_norm"] == 0 for event in events))
                    if not episode["metadata"]["query_fresh_writes_enabled"]:
                        self.assertTrue(all(not event["write_enabled"] and event["proposed_write_norm"] == 0 for event in events))


if __name__ == "__main__":
    unittest.main()
