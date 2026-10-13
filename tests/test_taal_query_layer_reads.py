"""Query-only layer read ablations with ordinary memory write rules."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
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

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal import state_audit
from evaluation.taal.interventions import AUDIT_CONDITIONS, audit_conditions, memory_execution_controls
from evaluation.taal.query_write_replay import QueryWriteReplay
from evaluation.taal.state_audit import fork_session_state
from evaluation.taal.trace_export import TaalTraceRecorder
from tests import test_taal_state_audit


ROOT = Path(__file__).resolve().parents[1]


class TaalQueryLayerReadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def model(layers=3):
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.num_hidden_layers = layers
        config.layer_types = ["full_attention"] * layers
        torch.manual_seed(73)
        model = TaalQwen3ForCausalLM(config).eval()
        for layer in model.model.layers:
            layer.taal.residual_gate.data.fill_(0.5)
        return model

    def test_suite_preserves_controls_and_validates_checkpoint_indices(self):
        for disabled in ([27], list(range(12))):
            conditions = audit_conditions("query_layer_reads", num_layers=28, read_layers_off=disabled)
            self.assertEqual(conditions[:6], AUDIT_CONDITIONS)
            self.assertEqual(len(conditions), 8)
            own, donor = conditions[-2:]
            self.assertEqual(own.read_layers, tuple(i for i in range(28) if i not in disabled))
            self.assertTrue(all(c.fresh_writes_enabled and c.momentum_enabled
                                and c.forgetting_enabled and c.updates_enabled for c in conditions))
            policy = asdict(own)
            policy.update(name=donor.name, memory_source="swapped")
            self.assertEqual(policy, asdict(donor))
        for disabled in (None, (), [], [True], [-1], [28], [1, 1], "27"):
            with self.subTest(disabled=disabled), self.assertRaisesRegex(ValueError, "query_read_layers_off"):
                audit_conditions("query_layer_reads", num_layers=28, read_layers_off=disabled)
        for count in (None, True, 0):
            with self.assertRaisesRegex(ValueError, "checkpoint layer count"):
                audit_conditions("query_layer_reads", num_layers=count, read_layers_off=[0])

    def test_configs_change_only_the_named_query_read_mask(self):
        base = yaml.safe_load((ROOT / "evaluation/configs/taal/capacity_d_distance_diversity_audit.yaml").read_text())
        for name, disabled in (("l27", [27]), ("early", list(range(12)))):
            actual = yaml.safe_load((ROOT / f"evaluation/configs/taal/d_{name}_query_reads_off.yaml").read_text())
            expected = dict(base, audit_suite="query_layer_reads", query_read_layers_off=disabled,
                            persistent_writes_enabled=True, query_updates_enabled=True)
            self.assertEqual(actual, expected)
            self.assertTrue(actual["save_traces"])

    @torch.inference_mode()
    def test_last_layer_mask_keeps_exact_writes_and_all_on_off_endpoints(self):
        model = self.model()
        prefix = model.prefill(torch.tensor([[1, 2, 3, 4, 5, 6]]), execution_block_size=3).state
        query = torch.tensor([[7, 8, 9]])
        def run(layers, scale=1.0):
            recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
            with memory_execution_controls(model, read_layers=layers), recorder:
                output = model.prefill(query, state=fork_session_state(prefix, prefix.memory_states),
                                       execution_block_size=2, prepend_memory_tokens=False,
                                       memory_read_scale=scale)
            return output, recorder
        baseline, baseline_trace = run(None)
        all_on, _ = run((0, 1, 2))
        torch.testing.assert_close(all_on.logits, baseline.logits, rtol=0, atol=0)
        masked, masked_trace = run((0, 1))
        self.assertFalse(torch.equal(masked.logits, baseline.logits))
        self.assertEqual(masked_trace.writes, baseline_trace.writes)
        for layer in range(3):
            for name in prefix.memory_states[layer].weights:
                torch.testing.assert_close(masked.state.memory_states[layer].weights[name],
                                           baseline.state.memory_states[layer].weights[name], rtol=0, atol=0)
                torch.testing.assert_close(masked.state.memory_states[layer].momentum[name],
                                           baseline.state.memory_states[layer].momentum[name], rtol=0, atol=0)
        self.assertTrue(all(event.enabled == (event.layer != 2) for event in masked_trace.reads))
        self.assertTrue(all(event.injected_norm == 0 for event in masked_trace.reads if event.layer == 2))
        layer_off, _ = run(())
        scale_off, _ = run(None, 0.0)
        torch.testing.assert_close(layer_off.logits, scale_off.logits, rtol=0, atol=0)
        self.assertEqual(prefix.tokens_seen, 6)
        self.assertTrue(all(layer.taal.reads_enabled for layer in model.model.layers))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with memory_execution_controls(model, read_layers=(2,)):
                raise RuntimeError("fixture")
        self.assertTrue(all(layer.taal.reads_enabled for layer in model.model.layers))

    @torch.inference_mode()
    def test_early_mask_leaves_write_rules_on_but_can_change_downstream_writes(self):
        model = self.model()
        prefix = model.prefill(torch.tensor([[1, 2, 3, 4, 5, 6]]), execution_block_size=3).state
        query = torch.tensor([[7, 8, 9]])
        baseline = model.prefill(query, state=fork_session_state(prefix, prefix.memory_states),
                                 execution_block_size=2, prepend_memory_tokens=False)
        recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
        with memory_execution_controls(model, read_layers=(2,)), recorder:
            masked = model.prefill(query, state=fork_session_state(prefix, prefix.memory_states),
                                   execution_block_size=2, prepend_memory_tokens=False)
        self.assertTrue(all(event.write_enabled and event.updates_enabled for event in recorder.writes))
        self.assertTrue(all(event.enabled == (event.layer == 2) for event in recorder.reads))
        self.assertTrue(any(not torch.equal(value, masked.state.memory_states[2].weights[name])
                            for name, value in baseline.state.memory_states[2].weights.items()))
        self.assertTrue(all(layer.taal.neural_memory.momentum_enabled
                            and layer.taal.neural_memory.forgetting_enabled
                            for layer in model.model.layers))

    @torch.inference_mode()
    def test_exact_replay_isolates_early_read_mask_and_restores_hooks(self):
        model = self.model()
        prefix = model.prefill(torch.tensor([[1, 2, 3, 4, 5, 6]]), execution_block_size=3).state
        query = torch.tensor([[7, 8, 9]])
        original_methods = [layer.taal.neural_memory._process_chunk for layer in model.model.layers]
        reference = QueryWriteReplay(model, token_count=3)
        native_trace = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
        with reference, native_trace:
            native = model.prefill(query, state=fork_session_state(prefix, prefix.memory_states),
                                   execution_block_size=2, prepend_memory_tokens=False)
        for mask in ((0, 1, 2), (2,), ()):
            recorder = TaalTraceRecorder(model, [0, 1, 2], 3, position_offset=6)
            with QueryWriteReplay(model, 3, reference), memory_execution_controls(model, read_layers=mask), recorder:
                output = model.prefill(query, state=fork_session_state(prefix, prefix.memory_states),
                                       execution_block_size=2, prepend_memory_tokens=False)
            self.assertEqual(recorder.writes, native_trace.writes)
            for layer in range(3):
                for name, expected in native.state.memory_states[layer].weights.items():
                    torch.testing.assert_close(output.state.memory_states[layer].weights[name], expected, rtol=0, atol=0)
                    torch.testing.assert_close(output.state.memory_states[layer].momentum[name],
                                               native.state.memory_states[layer].momentum[name], rtol=0, atol=0)
            if len(mask) == 3:
                torch.testing.assert_close(output.logits, native.logits, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(output.logits, native.logits))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with QueryWriteReplay(model, 3, reference):
                raise RuntimeError("fixture")
        self.assertEqual([layer.taal.neural_memory._process_chunk for layer in model.model.layers], original_methods)
        with self.assertRaisesRegex(ValueError, "every token"):
            with QueryWriteReplay(model, 4):
                model.prefill(query, state=fork_session_state(prefix, prefix.memory_states), prepend_memory_tokens=False)
        self.assertEqual([layer.taal.neural_memory._process_chunk for layer in model.model.layers], original_methods)

    def test_bundle_guard_requires_d_and_propagates_either_audit_failure(self):
        launcher = (ROOT / "evaluation/sbatch/taal/fs_qwen_d_query_layers_short.sbatch").read_text()
        for text in ("--partition=gpu-short", "--gres=gpu:h200:1", "--time=02:00:00",
                     "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy",
                     "git rev-parse HEAD", "utils.explorer_preflight", "for ablation in l27 early"):
            self.assertIn(text, launcher)
        guard = launcher.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"num_hidden_layers": 28, "working_memory_size": 32, "memory_chunk_size": 1}
            metadata = {"progress": {"step": 449}, "reason": "signal"}
            (root / "config.json").write_text(json.dumps(config))
            (root / "training_metadata.json").write_text(json.dumps(metadata))
            valid = subprocess.run(["python3", "-c", guard, directory], capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            metadata["progress"]["step"] = 425
            (root / "training_metadata.json").write_text(json.dumps(metadata))
            invalid = subprocess.run(["python3", "-c", guard, directory], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
        # Exercise the actual shell bundle, replacing expensive GPU invocations.
        bundle = launcher[launcher.index("audit_status=0"):]
        for fail in ("none", "l27", "early", "both"):
            script = """python() {
                if [[ "$1" == '-m' && "$2" == 'utils.explorer_preflight' ]]; then return 0; fi
                case "$*" in *d_l27_*) [[ "$FAIL" == l27 || "$FAIL" == both ]] && return 7;;
                *d_early_*) [[ "$FAIL" == early || "$FAIL" == both ]] && return 8;; esac
                return 0;
            }
            """ + bundle
            result = subprocess.run(["bash", "-c", script, "fixture", "checkpoint", "output"],
                                    env={"FAIL": fail}, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, fail == "none", result.stderr)

    def test_runner_all_16_examples_28_layers_conditions_traces_and_resume(self):
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
        for ablation in ("l27", "early"):
            with self.subTest(ablation=ablation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = self.model(layers=28)
                model.save_pretrained(root / "checkpoint")
                parameters = {name: value.clone() for name, value in model.state_dict().items()}
                dataset = root / "data.jsonl"
                dataset.write_text("\n".join(json.dumps(example) for example in examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({
                    "tokenizer": "CPU fixture", "tokenizer_revision": "a" * 40,
                }))
                settings = yaml.safe_load((ROOT / f"evaluation/configs/taal/d_{ablation}_query_reads_off.yaml").read_text())
                settings["dataset"]["path"] = str(dataset)
                settings["execution_block_size"] = 3
                config = root / "audit.yaml"
                config.write_text(yaml.safe_dump(settings))
                output = root / "out"
                argv = ["audit", "--config", str(config), "--checkpoint", str(root / "checkpoint"),
                        "--output-dir", str(output)]
                with (
                    patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples),
                    patch.object(state_audit, "load_model", return_value=model),
                    patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                    redirect_stdout(io.StringIO()),
                ):
                    state_audit.main()
                    rows_path = output / "results.jsonl"
                    rows = rows_path.read_text().splitlines()
                    rows_path.write_text("\n".join(rows[:-1]) + "\n")
                    state_audit.main()
                    state_audit.main()
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
                rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
                self.assertEqual(len({row["evaluation_id"] for row in rows}), 128)
                self.assertTrue(all(count == 16 for count in Counter(row["audit_condition"] for row in rows).values()))
                summary = json.loads((output / "summary.json").read_text())
                self.assertIn("query_layers_off", summary["matched_memory_contrasts"])
                index = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual((len(index["episodes"]), len(index["comparisons"])), (128, 112))
                prefixes = {}
                payloads = {}
                for key, relative in index["episodes"].items():
                    episode = json.loads(gzip.decompress((output / "memory_traces" / relative).read_bytes()))
                    payloads[key] = episode
                    self.assertEqual(len(episode["tokens"]), 11)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    before = [event for event in episode["reads"] if event["position"] < 8]
                    self.assertTrue(all(event["enabled"] for event in before))
                    eid = episode["example_id"]
                    self.assertEqual(prefixes.setdefault(eid, before), before)
                    writes = [event for event in episode["writes"] if event["position"] is not None and event["position"] >= 8]
                    self.assertEqual(len(writes), 3 * 28)
                    self.assertTrue(all(event["write_enabled"] and event["updates_enabled"] for event in writes))
                    if episode["condition_id"].startswith("query_layers_off"):
                        self.assertTrue(episode["metadata"]["query_write_replay_verified"])
                        self.assertEqual(episode["metadata"]["query_write_replay_scope"], "prompt_query_only")
                        self.assertTrue(all(event["enabled"] == (event["layer"] not in settings["query_read_layers_off"])
                                            for event in episode["reads"] if event["position"] >= 8))
                for example in examples:
                    for variant, baseline in (("query_layers_off", "correct_full"),
                                              ("query_layers_off_swapped", "swapped")):
                        native = payloads[f"{example['example_id']}/{baseline}"]
                        replayed = payloads[f"{example['example_id']}/{variant}"]
                        query_writes = lambda payload: [event for event in payload["writes"]
                                                       if event["position"] is not None and 8 <= event["position"] < 10]
                        self.assertEqual(query_writes(native), query_writes(replayed))
                relative = index["comparisons"]["pair-0/correct_full/query_layers_off"]
                comparison = json.loads(gzip.decompress((output / "memory_traces" / relative).read_bytes()))
                self.assertEqual(comparison["intervention"], "read_scale")
                self.assertTrue(comparison["same_starting_kv"])
                self.assertTrue(comparison["same_starting_memory"])


if __name__ == "__main__":
    unittest.main()
