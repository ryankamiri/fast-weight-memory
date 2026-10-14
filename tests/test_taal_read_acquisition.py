"""Snapshot reads and pre-projection writer crossings on real tiny TaaL models."""

from collections import Counter
from contextlib import redirect_stdout
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
from evaluation.taal.interventions import AUDIT_CONDITIONS, audit_conditions
from evaluation.taal.query_write_replay import QueryWriteReplay
from evaluation.taal.read_acquisition import FinalReadReplay, assert_same_state, evolve_shadow_writer
from evaluation.taal.state_audit import clone_memory_states, fork_session_state
from evaluation.taal.trace_export import TaalTraceRecorder
from tests import test_taal_query_layer_reads as fixtures


ROOT = Path(__file__).resolve().parents[1]


class ReadAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conditions_and_real_config_counts(self):
        for suite, count in (("read_snapshots", 98), ("writer_cross", 14)):
            config = yaml.safe_load((ROOT / f"evaluation/configs/taal/d_{suite}.yaml").read_text())
            self.assertTrue(config["save_traces"])
            conditions = audit_conditions(suite, num_layers=28, target_layer=27, query_token_count=20)
            self.assertEqual(conditions[:6], AUDIT_CONDITIONS)
            self.assertEqual(len({c.name for c in conditions}), count)
            self.assertTrue(all(c.replay_query_writes for c in conditions[6:]))
            for bad in (None, True, 26, 28):
                with self.assertRaises(ValueError):
                    audit_conditions(suite, num_layers=28, target_layer=bad, query_token_count=20)
        for bad in (None, True, 0):
            with self.assertRaises(ValueError):
                audit_conditions("read_snapshots", num_layers=28, target_layer=27, query_token_count=bad)

    @staticmethod
    def capture(dtype=torch.float32):
        model = fixtures.TaalQueryLayerReadTests.model().to(dtype=dtype)
        own = model.prefill(torch.tensor([[1, 2, 3, 4, 5, 6]]), execution_block_size=3).state
        donor = model.prefill(torch.tensor([[1, 9, 3, 4, 5, 6]]), execution_block_size=3).state
        states = {"own": own.memory_states, "donor": donor.memory_states}
        query = torch.tensor([[7, 8, 9]])
        references, outputs = {}, {}
        for name, memory in states.items():
            reference = QueryWriteReplay(model, 3)
            with reference:
                outputs[name] = model.prefill(query, state=fork_session_state(own, memory),
                                             execution_block_size=3, prepend_memory_tokens=False)
            references[name] = reference
        return model, own, states, query, references, outputs

    @torch.inference_mode()
    def test_shadow_endpoints_inputs_convolution_and_no_slow_parameter_changes(self):
        for dtype in (torch.float32, torch.bfloat16):
            model, prefix, states, query, references, outputs = self.capture(dtype)
            memory = model.model.layers[2].taal.neural_memory
            parameters = {name: value.clone() for name, value in model.state_dict().items()}
            for initial_name in ("own", "donor"):
                for inputs_name in ("own", "donor"):
                    original = clone_memory_states(states[initial_name])
                    shadow = evolve_shadow_writer(memory, original[2], references[inputs_name], 2, position_offset=6)
                    inputs = torch.cat([event.inputs for event in references[inputs_name].events[2]], dim=1)
                    _, independently_recomputed = memory(inputs, state=clone_memory_states(states[initial_name])[2])
                    assert_same_state(shadow.state, independently_recomputed)
                    assert_same_state(original[2], states[initial_name][2])
                    self.assertEqual([w["position"] for w in shadow.writes], [6, 7, 8])
                    self.assertTrue(all(w["net_weight_change_norm"] > 0 for w in shadow.writes))
                    self.assertIsNone(memory.trace_observer)
                    if initial_name == inputs_name:
                        assert_same_state(shadow.state, outputs[initial_name].state.memory_states[2])
            # Corrupt saved donor k/v: the shadow must ignore these and rebuild
            # them from raw inputs using its own initial convolution histories.
            before = evolve_shadow_writer(memory, clone_memory_states(states["own"])[2], references["donor"], 2, position_offset=6)
            for event in references["donor"].events[2]:
                event.keys.fill_(float("nan"))
                event.values.fill_(float("nan"))
            after = evolve_shadow_writer(memory, clone_memory_states(states["own"])[2], references["donor"], 2, position_offset=6)
            assert_same_state(before.state, after.state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)

    @torch.inference_mode()
    def test_final_read_exact_snapshot_and_unchanged_background(self):
        model, prefix, states, query, references, outputs = self.capture()
        memory = model.model.layers[2].taal.neural_memory
        for background in ("own", "donor"):
            for source in ("own", "donor"):
                for index in range(3):
                    weights = references[source].events[2][index].weights_after
                    fixed_query = references[background].events[2][-1].queries
                    read = FinalReadReplay(model, 2, 3, weights, fixed_query)
                    observed = []
                    hook = memory.register_forward_hook(lambda _, args, result: observed.append(result[0].clone()))
                    try:
                        with QueryWriteReplay(model, 3, references[background]), read:
                            result = model.prefill(query, state=fork_session_state(prefix, states[background]),
                                                   execution_block_size=3, prepend_memory_tokens=False)
                    finally:
                        hook.remove()
                    for layer in range(3):
                        assert_same_state(result.state.memory_states[layer], outputs[background].state.memory_states[layer])
                    expected = memory.memory_mlp(fixed_query.float(), weights=weights).predicted_values
                    torch.testing.assert_close(observed[0][:, -1:], expected, rtol=0, atol=0)
                    self.assertEqual(len(read.events), 1)
                    if background == source and index == 2:
                        torch.testing.assert_close(result.logits, outputs[background].logits, rtol=0, atol=0)

    @torch.inference_mode()
    def test_invalid_reads_and_shadow_restore_hooks(self):
        model, prefix, states, query, references, outputs = self.capture()
        memory = model.model.layers[2].taal.neural_memory
        original = memory._process_chunk
        fixed_query = references["own"].events[2][-1].queries
        weights = references["own"].events[2][-1].weights_after
        for invalid in ("query", "weights", "short"):
            candidate_query = fixed_query + 1 if invalid == "query" else fixed_query
            candidate_weights = {k: v * float("nan") for k, v in weights.items()} if invalid == "weights" else weights
            with self.assertRaises(ValueError):
                with FinalReadReplay(model, 2, 3, candidate_weights, candidate_query):
                    model.prefill(query[:, :1] if invalid == "short" else query,
                                  state=fork_session_state(prefix, states["own"]),
                                  execution_block_size=3, prepend_memory_tokens=False)
            self.assertEqual(memory._process_chunk, original)
        with patch.object(memory, "forward", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                evolve_shadow_writer(memory, states["own"][2], references["own"], 2, position_offset=6)
        self.assertIsNone(memory.trace_observer)

    def test_real_runner_all_examples_layers_traces_and_resume(self):
        tokenizer = SimpleNamespace(
            decode=lambda ids, **kwargs: str(ids[0]),
            encode=lambda text, **kwargs: [1] if text.startswith("User:") else [3, 4],
        )
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp, equal=torch.equal,
        )
        examples = [{
            "example_id": f"pair-{i}", "fact_id": i, "record_name": f"record-{i // 2}",
            "answer": str(9 + i), "condition": "micro_conflict", "query_variant": "exact",
            "input_ids": [1, 9 + i, 3, 4, 5, 6, 7, 8, 6, 7], "final_query_position": 8,
            "fact_position": 1, "target_token_id": 9 + i,
            "candidate_token_ids": [9 + 2 * (i // 2), 10 + 2 * (i // 2)],
        } for i in range(16)]
        for suite, count in (("read_snapshots", 26), ("writer_cross", 14)):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = fixtures.TaalQueryLayerReadTests.model(layers=28)
                model.save_pretrained(root / "checkpoint")
                dataset = root / "data.jsonl"
                dataset.write_text("\n".join(json.dumps(e) for e in examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": "CPU fixture", "tokenizer_revision": "a" * 40}))
                settings = yaml.safe_load((ROOT / f"evaluation/configs/taal/d_{suite}.yaml").read_text())
                settings["dataset"]["path"] = str(dataset)
                settings["execution_block_size"] = 3
                config = root / "config.yaml"
                config.write_text(yaml.safe_dump(settings))
                out = root / "out"
                argv = ["audit", "--config", str(config), "--checkpoint", str(root / "checkpoint"), "--output-dir", str(out)]
                with (patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                      patch.object(state_audit, "load_dataset", return_value=examples),
                      patch.object(state_audit, "load_model", return_value=model),
                      patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                      redirect_stdout(io.StringIO())):
                    state_audit.main()
                    # Resume a single own-only endpoint: both reference sources
                    # are still captured, independent of which conditions remain.
                    endpoint = ("snapshot_context_own_weights_own_query_01" if suite == "read_snapshots"
                                else "writer_context_own_initial_own_inputs_own")
                    rows = (out / "results.jsonl").read_text().splitlines()
                    (out / "results.jsonl").write_text("\n".join(r for r in rows if json.loads(r)["evaluation_id"] != f"pair-0/{endpoint}") + "\n")
                    state_audit.main()
                    state_audit.main()
                rows = [json.loads(r) for r in (out / "results.jsonl").read_text().splitlines()]
                self.assertEqual(len({r["evaluation_id"] for r in rows}), 16 * count)
                self.assertEqual(set(Counter(r["audit_condition"] for r in rows).values()), {16})
                index = json.loads((out / "memory_traces/manifest.json").read_text())
                self.assertEqual((len(index["episodes"]), len(index["comparisons"])), (16 * count, 16 * (count - 1)))
                episodes = {key: json.loads(gzip.decompress((out / "memory_traces" / path).read_bytes())) for key, path in index["episodes"].items()}
                for row in rows:
                    episode = episodes[row["evaluation_id"]]
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(episode["tokens"][-1]["token_id"], row["vocabulary_top_token_id"])
                    self.assertEqual(len(episode["reads"]), 28 * 11)
                    if not row.get("saved_weight_read_events"):
                        continue
                    self.assertTrue(row["saved_weight_read_events"][0]["identical_query_verified"])
                    self.assertFalse(row["read_intervention_changes_recurrent_weights"])
                    substituted = [r for r in episode["reads"] if r["state_timing"] == "substituted"]
                    self.assertEqual([(r["layer"], r["position"]) for r in substituted], [(27, 9)])
                    baseline = "correct_full" if row["memory_source"] == "correct" else "swapped"
                    native = episodes[f"{row['example_id']}/{baseline}"]
                    query_writes = lambda p: [w for w in p["writes"] if w["position"] is not None and 8 <= w["position"] < 10]
                    self.assertEqual(query_writes(episode), query_writes(native))
                    comparison = json.loads(gzip.decompress((out / "memory_traces" / index["comparisons"][f"{row['example_id']}/{baseline}/{row['audit_condition']}"]).read_bytes()))
                    self.assertEqual(comparison["scope"], "prediction_token")
                    self.assertEqual(comparison["intervention"], "memory_read")
                    if row["native_endpoint_verified"]:
                        self.assertEqual(row["target_log_probability"], native["outcome"]["target_log_probability"])
                    if suite == "writer_cross":
                        self.assertEqual([w["position"] for w in row["shadow_writer_writes"]], [8, 9])


if __name__ == "__main__":
    unittest.main()
