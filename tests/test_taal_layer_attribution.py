"""Layer ownership and read-factor interventions, including real compact exports."""

from collections import Counter
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from evaluation.taal import state_audit
from evaluation.taal.interventions import AUDIT_CONDITIONS, ReadCrossPolicy, audit_conditions
from evaluation.taal.query_write_replay import QueryWriteReplay
from evaluation.taal.read_cross import ReadCrossReplay
from evaluation.taal.state_audit import fork_session_state
from evaluation.taal.trace_export import TaalTraceRecorder
from tests import test_taal_query_layer_reads as fixtures


ROOT = Path(__file__).resolve().parents[1]


class TaalLayerAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conditions_cover_full_factorials_and_preserve_controls(self):
        for suite, count in (("layer_sources", 10), ("read_cross", 22)):
            conditions = audit_conditions(suite, num_layers=28, target_layer=27)
            self.assertEqual(conditions[:6], AUDIT_CONDITIONS)
            self.assertEqual(len({c.name for c in conditions}), count)
            self.assertTrue(all(c.replay_query_writes for c in conditions[6:]))
            for bad in (None, True, -1, 26, 28):
                with self.assertRaises(ValueError):
                    audit_conditions(suite, num_layers=28, target_layer=bad)
            config = yaml.safe_load((ROOT / f"evaluation/configs/taal/d_{suite}.yaml").read_text())
            self.assertTrue(config["save_traces"])
            self.assertEqual(config["dataset"]["end"] - config["dataset"]["start"], 16)
            self.assertEqual(config["target_layer"], 27)

    @staticmethod
    def capture():
        model = fixtures.TaalQueryLayerReadTests.model()
        a = model.prefill(torch.tensor([[1, 2, 3, 4, 5, 6]]), execution_block_size=3).state
        b = model.prefill(torch.tensor([[1, 9, 3, 4, 5, 6]]), execution_block_size=3).state
        query = torch.tensor([[7, 8, 9]])
        states = {"own": a.memory_states, "donor": b.memory_states}
        references, outputs, traces = {}, {}, {}
        for name, memory in states.items():
            replay = QueryWriteReplay(model, 3)
            recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
            with replay, recorder:
                outputs[name] = model.prefill(query, state=fork_session_state(a, memory),
                                             execution_block_size=2, prepend_memory_tokens=False)
            references[name], traces[name] = replay, recorder
        return model, a, states, query, references, outputs, traces

    @torch.inference_mode()
    def test_per_layer_replay_matches_each_source_and_preserves_snapshots(self):
        model, prefix, states, query, references, outputs, traces = self.capture()
        for lower in ("own", "donor"):
            for last in ("own", "donor"):
                selected = {i: last if i == 2 else lower for i in range(3)}
                memory = {i: states[source][i] for i, source in selected.items()}
                recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
                with QueryWriteReplay(model, 3, references_by_layer={i: references[s] for i, s in selected.items()}), recorder:
                    result = model.prefill(query, state=fork_session_state(prefix, memory),
                                           execution_block_size=2, prepend_memory_tokens=False)
                for layer, source in selected.items():
                    self.assertEqual([w for w in recorder.writes if w.layer == layer],
                                     [w for w in traces[source].writes if w.layer == layer])
                    for component in ("weights", "momentum"):
                        for name, value in getattr(result.state.memory_states[layer], component).items():
                            torch.testing.assert_close(value, getattr(outputs[source].state.memory_states[layer], component)[name], rtol=0, atol=0)
                if lower == last:
                    torch.testing.assert_close(result.logits, outputs[lower].logits, rtol=0, atol=0)
        self.assertEqual(prefix.tokens_seen, 6)
        with self.assertRaisesRegex(ValueError, "cover every"):
            QueryWriteReplay(model, 3, references_by_layer={2: references["own"]})
        # Wrong starting state must fail instead of quietly changing the writer.
        with self.assertRaisesRegex(ValueError, "weights differ"):
            with QueryWriteReplay(model, 3, references["donor"]):
                model.prefill(query, state=fork_session_state(prefix, states["own"]), prepend_memory_tokens=False)

    @torch.inference_mode()
    def test_crossed_read_is_exact_mlp_of_selected_factors_without_state_changes(self):
        model, prefix, states, query, references, outputs, traces = self.capture()
        memory = model.model.layers[2].taal.neural_memory
        originals = [layer.taal.neural_memory._process_chunk for layer in model.model.layers]
        for scope in ("whole_query", "prediction_token"):
            for background in ("own", "donor"):
                for weights in ("own", "donor"):
                    for address in ("own", "donor"):
                        policy = ReadCrossPolicy(2, weights, address, scope)
                        crossed = ReadCrossReplay(model, 3, policy, references)
                        recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
                        observed = []
                        hook = memory.register_forward_hook(lambda _, args, output: observed.append(output[0].clone()))
                        try:
                            with recorder, QueryWriteReplay(model, 3, references[background]), crossed:
                                output = model.prefill(query, state=fork_session_state(prefix, states[background]),
                                                       execution_block_size=2, prepend_memory_tokens=False)
                        finally:
                            hook.remove()
                        self.assertEqual(recorder.writes, traces[background].writes)
                        actual = torch.cat(observed, dim=1)
                        for index in range(3):
                            w_source, q_source = weights, address
                            if scope == "prediction_token" and index != 2:
                                w_source = q_source = background
                            expected = memory.memory_mlp(
                                references[q_source].events[2][index].queries.float(),
                                weights=references[w_source].events[2][index].weights_after,
                            ).predicted_values
                            torch.testing.assert_close(actual[:, index:index + 1].float(), expected, rtol=0, atol=0)
                        self.assertEqual(len(crossed.events), 3 if scope == "whole_query" else 1)
                        if weights == address == background:
                            torch.testing.assert_close(output.logits, outputs[background].logits, rtol=0, atol=0)
                        self.assertEqual([layer.taal.neural_memory._process_chunk for layer in model.model.layers], originals)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with QueryWriteReplay(model, 3, references["own"]), ReadCrossReplay(model, 3, policy, references):
                raise RuntimeError("fixture")
        self.assertEqual([layer.taal.neural_memory._process_chunk for layer in model.model.layers], originals)

    @torch.inference_mode()
    def test_nonfinite_read_and_incomplete_scope_abort_and_restore(self):
        model, prefix, states, query, references, _, _ = self.capture()
        originals = [layer.taal.neural_memory._process_chunk for layer in model.model.layers]
        policy = ReadCrossPolicy(2, "donor", "own", "whole_query")
        with self.assertRaisesRegex(ValueError, "every prompt-query token"):
            with ReadCrossReplay(model, 3, policy, references):
                model.prefill(query[:, :1], state=fork_session_state(prefix, states["own"]), prepend_memory_tokens=False)
        for value in references["donor"].events[2][0].weights_after.values():
            value.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite crossed read"):
            with ReadCrossReplay(model, 3, policy, references):
                model.prefill(query, state=fork_session_state(prefix, states["own"]), prepend_memory_tokens=False)
        self.assertEqual([layer.taal.neural_memory._process_chunk for layer in model.model.layers], originals)

    def test_launcher_guards_checkpoint_data_and_propagates_command_failure(self):
        launcher_path = ROOT / "evaluation/sbatch/taal/fs_qwen_d_attribution_short.sbatch"
        launcher = launcher_path.read_text()
        syntax = subprocess.run(["bash", "-n", str(launcher_path)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        guard = launcher.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        fingerprint = "285dab0ad8f638d95caefb283aeb67598ca959105974e57453a8e925a4359100"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps({"num_hidden_layers": 28, "working_memory_size": 32, "memory_chunk_size": 1}))
            (path / "training_metadata.json").write_text(json.dumps({"progress": {"step": 449}, "reason": "signal"}))
            with (patch("sys.argv", ["guard", folder]),
                  patch("evaluation.taal.prefix_store.checkpoint_fingerprint", return_value=fingerprint),
                  redirect_stdout(io.StringIO())):
                exec(compile(guard, "launcher-parent-guard", "exec"), {})
                (path / "training_metadata.json").write_text(json.dumps({"progress": {"step": 425}, "reason": "signal"}))
                with self.assertRaisesRegex(ValueError, "final449"):
                    exec(compile(guard, "launcher-parent-guard", "exec"), {})
        suffix = launcher[launcher.index('python -m utils.explorer_preflight'):]
        for failing in ("none", "preflight", "audit"):
            commands = """set -e
python() {
    case "$2" in
        utils.explorer_preflight) [[ "$FAIL" == preflight ]] && return 7 ;;
        evaluation.taal.state_audit) [[ "$FAIL" == audit ]] && return 8 ;;
    esac
    return 0
}
config=fixture
""" + suffix
            result = subprocess.run(["bash", "-c", commands, "fixture", "read_cross", "checkpoint", "output", "sha"],
                                    env={"FAIL": failing}, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, failing == "none")

    def test_runner_16_examples_28_layers_exports_metadata_endpoints_and_resume(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp, equal=torch.equal,
        )
        examples = [{
            "example_id": f"pair-{i}", "fact_id": i, "record_name": f"record-{i // 2}",
            "answer": str(9 + i), "condition": "micro_conflict", "query_variant": "exact",
            "input_ids": [1, 2 + i, 3, 4, 5, 6, 7, 8, 6, 7], "final_query_position": 8,
            "target_token_id": 9 + i, "candidate_token_ids": [9 + 2 * (i // 2), 10 + 2 * (i // 2)],
        } for i in range(16)]
        for suite, count in (("layer_sources", 10), ("read_cross", 22)):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = fixtures.TaalQueryLayerReadTests.model(layers=28)
                model.save_pretrained(root / "checkpoint")
                parameters = {name: value.clone() for name, value in model.state_dict().items()}
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
                    # Resume only a no-donor endpoint: capture still needs BOTH sources.
                    rows = (out / "results.jsonl").read_text().splitlines()
                    endpoint = "layers_own_last_own" if suite == "layer_sources" else "read_whole_query_context_own_weights_own_query_own"
                    (out / "results.jsonl").write_text("\n".join(r for r in rows if json.loads(r)["evaluation_id"] != f"pair-0/{endpoint}") + "\n")
                    state_audit.main()
                    state_audit.main()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
                rows = [json.loads(r) for r in (out / "results.jsonl").read_text().splitlines()]
                self.assertEqual(len({r["evaluation_id"] for r in rows}), 16 * count)
                self.assertTrue(all(n == 16 for n in Counter(r["audit_condition"] for r in rows).values()))
                index = json.loads((out / "memory_traces/manifest.json").read_text())
                self.assertEqual((len(index["episodes"]), len(index["comparisons"])), (16 * count, 16 * (count - 1)))
                episodes = {key: json.loads(gzip.decompress((out / "memory_traces" / path).read_bytes())) for key, path in index["episodes"].items()}
                for row in rows:
                    e = episodes[row["evaluation_id"]]
                    self.assertEqual(e["tokens"][-1]["phase"], "generated")
                    self.assertEqual(e["tokens"][-1]["token_id"], row["vocabulary_top_token_id"])
                    self.assertEqual(len([w for w in e["writes"] if w["position"] is not None]), 28 * 11)
                    self.assertEqual(len(e["reads"]), 28 * 11)
                    if not row.get("query_write_replay"):
                        continue
                    self.assertTrue(row["query_write_replay_verified"])
                    baseline = "correct_full" if row["memory_source"] == "correct" else "swapped"
                    native = episodes[f"{row['example_id']}/{baseline}"]
                    if suite == "read_cross":
                        writes = lambda p: [w for w in p["writes"] if w["position"] is not None and 8 <= w["position"] < 10]
                        self.assertEqual(writes(e), writes(native))
                        policy = row["read_cross"]
                        self.assertEqual(len(row["read_cross_events"]), 2 if policy["scope"] == "whole_query" else 1)
                        self.assertFalse(e["metadata"]["read_cross_changes_recurrent_weights"])
                        path = index["comparisons"][f"{row['example_id']}/{baseline}/{row['audit_condition']}"]
                        comparison = json.loads(gzip.decompress((out / "memory_traces" / path).read_bytes()))
                        self.assertEqual(comparison["intervention"], "memory_read")
                        self.assertEqual(comparison["scope"], policy["scope"])
                        self.assertTrue(comparison["same_starting_memory"])
                    if row["native_endpoint_verified"]:
                        self.assertEqual(row["target_log_probability"], native["outcome"]["target_log_probability"])


if __name__ == "__main__":
    unittest.main()
