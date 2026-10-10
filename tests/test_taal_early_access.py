import gzip
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import torch
from datasets import Dataset
from unittest.mock import patch

from data.dataloader import BridgeMemoryCollator, create_dataloader
from data.taal_state_probe import build_probe_examples
from evaluation.taal.interventions import audit_conditions, memory_execution_controls
from evaluation.taal.state_audit import fork_session_state
from evaluation.taal.state_readout_probe import (
    StateFeatures, StateProjector, collect_example, filler_forward, fit_probes, validate_splits,
)
from evaluation.taal import state_readout_probe
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder
from tests.test_taal_lifecycle import LifecycleTokenizer, examples
from tests import test_taal_trace_export as trace_fixture
from training.config import load_config
from training.engine import forward_batch


class EarlyStateAccessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(107)
        self.model = trace_fixture.TaalTraceExportTests.model()
        for layer in self.model.model.layers:
            layer.taal.residual_gate.data.fill_(0.25)

    @torch.inference_mode()
    def test_read_masks_preserve_frozen_states_and_record_disabled_layers(self):
        own = self.model.prefill(torch.tensor([[1, 2, 3, 4, 5]]))
        query = torch.tensor([[6, 7]])
        for read_layers in (None, (), (0,), (1,)):
            state = fork_session_state(own.state, own.state.memory_states)
            recorder = TaalTraceRecorder(self.model, [0, 1], 2)
            with memory_execution_controls(self.model, updates_enabled=False, read_layers=read_layers), recorder:
                output = self.model.prefill(query, state=state, prepend_memory_tokens=False)
            for layer, memory in own.state.memory_states.items():
                for component in ("weights", "momentum"):
                    for name, value in getattr(memory, component).items():
                        torch.testing.assert_close(value, getattr(output.state.memory_states[layer], component)[name], atol=0, rtol=0)
                reads = [event for event in recorder.reads if event.layer == layer]
                enabled = read_layers is None or layer in read_layers
                self.assertTrue(all(event.enabled == enabled for event in reads))
                if not enabled:
                    self.assertTrue(all(event.injected_norm == 0 for event in reads))
            if read_layers == ():
                disabled = self.model.prefill(query, state=fork_session_state(own.state, own.state.memory_states),
                                             memory_read_scale=0.0, prepend_memory_tokens=False)
                torch.testing.assert_close(output.logits, disabled.logits, atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with memory_execution_controls(self.model, read_layers=()):
                raise RuntimeError("restore")
        self.assertTrue(all(layer.taal.reads_enabled for layer in self.model.model.layers))
        for invalid in ((2,), (0, 0), (-1,), (True,)):
            with self.assertRaisesRegex(ValueError, "layer indices"):
                with memory_execution_controls(self.model, read_layers=invalid):
                    pass

    def test_suite_counts_and_native_policy(self):
        groups = {"lower": list(range(14)), "middle": list(range(14, 24)), "upper": list(range(24, 28))}
        conditions = audit_conditions("layer_access", layer_groups=groups)
        self.assertEqual(len(conditions), 51)
        self.assertEqual(len({condition.name for condition in conditions}), 51)
        self.assertTrue(all(not condition.updates_enabled and condition.recipient_convolution
                            for condition in conditions[3:]))
        for stage in ("label", "fact", "final"):
            for mask in ("all", "none", "lower", "middle", "upper", "without_lower", "without_middle", "without_upper"):
                own = next(condition for condition in conditions if condition.name == f"{stage}_{mask}_correct_fixed")
                donor = next(condition for condition in conditions if condition.name == f"{stage}_{mask}_swapped_fixed")
                self.assertEqual(own.read_layers, donor.read_layers)
        native = audit_conditions("no_fresh_query")
        self.assertEqual(len(native), 9)
        self.assertTrue(all(not condition.fresh_writes_enabled for condition in native[:6]))
        self.assertTrue(all(condition.fresh_writes_enabled for condition in native[6:]))

    def test_training_recipe_keeps_matched_schedule_and_stops_at_control_step(self):
        root = Path("training/configs/taal")
        baseline = load_config(root / "qwen3_0_6b_conflict_gate_start.yaml")
        actual = load_config(root / "qwen3_0_6b_conflict_no_fresh_query.yaml")
        baseline.data.query_fresh_writes_enabled = False
        baseline.wandb.name = actual.wandb.name
        baseline.training.max_steps = 125
        self.assertEqual(actual.to_dict(), baseline.to_dict())
        actual.model.memory_chunk_size = 4
        with self.assertRaisesRegex(ValueError, "chunk_size=1"):
            actual.validate()

    def test_training_mask_boundaries_loader_and_backward_are_not_detached(self):
        rows = examples()
        collator = BridgeMemoryCollator(0.0, 1.0, query_fresh_writes_enabled=False)
        batch = collator(rows)
        self.assertTrue(bool(batch["write_mask"][:, :11].all()))
        self.assertFalse(bool(batch["write_mask"][:, 11:].any()))
        config = load_config("training/configs/taal/qwen3_0_6b_conflict_no_fresh_query.yaml")
        source = Dataset.from_list(rows).to_iterable_dataset()
        with patch("data.dataloader.load_dataset", return_value=source):
            loader = create_dataloader(config.data, config.data.train, config.loss, shuffle=False, seed=42)
        self.assertTrue(all("write_mask" in row for row in loader))
        with self.assertRaisesRegex(ValueError, "final_query_position"):
            collator([{key: value for key, value in rows[0].items() if key != "final_query_position"}])
        recorder = TaalTraceRecorder(self.model, [0, 1], 14)
        with recorder:
            output = forward_batch(self.model, collator([rows[0]]), config.loss)
        writes = [event for event in recorder.writes if event.position is not None]
        self.assertTrue(all(not event.write_enabled and event.proposed_write_norm == 0
                            for event in writes if event.position >= 11))
        self.assertTrue(any(event.proposed_write_norm > 0 for event in writes if event.position < 11))
        self.assertTrue(any(event.net_weight_change_norm > 0 for event in writes if event.position >= 11))
        self.model.train()
        output = forward_batch(self.model, collator([rows[0]]), config.loss)
        output.loss.backward()
        memory = self.model.model.layers[0].taal.neural_memory
        self.assertTrue(any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                            for parameter in memory.memory_mlp.parameters()))

    @torch.inference_mode()
    def test_collection_masks_absolute_positions_exports_snapshots_and_all_traces(self):
        row = {**examples()[0], "input_ids": [1, 2, 9, 4, 5] + [6, 7] * 9 + [8, 2],
               "final_query_position": 23, "context_family": 0, "probe_split": "train"}
        initial = self.model.model.initial_memory_states(1)
        projector = StateProjector(initial, dimension=4, seed=73)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = TaalTraceExporter(root / "memory_traces", run_metadata={"checkpoint": "fixture", "tokenizer": "fixture"})
            features = collect_example(self.model, LifecycleTokenizer(), row, projector=projector,
                root=root, exporter=exporter, execution_block_size=3, save_memory_states=True)
            self.assertEqual(set(features), {"label", "fact", "normal/middle", "normal/final",
                "fact_hidden/middle", "fact_hidden/final", "unrelated_hidden/middle", "unrelated_hidden/final"})
            self.assertEqual(len(list((root / "memory_states").rglob("*.pt"))), 8)
            self.assertEqual(len(exporter.manifest["episodes"]), 3)
            for relative in exporter.manifest["episodes"].values():
                with gzip.open(root / "memory_traces" / relative, "rt") as file:
                    trace = json.load(file)
                self.assertEqual(len(trace["tokens"]), 26)
                self.assertEqual(trace["tokens"][-1]["phase"], "generated")
                self.assertEqual(len(trace["reads"]), 52)
                self.assertEqual(len(trace["internal_prefixes"]), 2)
            # Resuming an interrupted example overwrites exports, not score rows.
            collect_example(self.model, LifecycleTokenizer(), row, projector=projector,
                root=root, exporter=exporter, execution_block_size=3, save_memory_states=True)
            self.assertEqual(len((root / "results.jsonl").read_text().splitlines()), 3)
        prefix = self.model.prefill(torch.tensor([[1, 2, 9, 4, 5]]), execution_block_size=3)
        direct = self.model.prefill(torch.tensor([[6, 7, 6]]), state=fork_session_state(prefix.state, prefix.state.memory_states),
                                   execution_block_size=3, prepend_memory_tokens=False)
        normal = filler_forward(self.model, row["input_ids"], fork_session_state(prefix.state, prefix.state.memory_states),
                                start=5, end=8, execution_block_size=3, hidden_span=None)
        torch.testing.assert_close(normal.logits, direct.logits, atol=0, rtol=0)
        absent = filler_forward(self.model, row["input_ids"], fork_session_state(prefix.state, prefix.state.memory_states),
                                start=5, end=8, execution_block_size=3, hidden_span=(100, 110))
        torch.testing.assert_close(normal.logits, absent.logits, atol=0, rtol=0)

    def test_probe_split_leakage_and_positive_negative_controls(self):
        rows = [{"example_id": f"{family}-{label}-{repeat}", "context_family": family,
                 "probe_split": split, "target_token_id": label}
                for family, split in enumerate(("train", "val", "test"))
                for label in range(4) for repeat in range(8)]
        validate_splits(rows)
        leaked = [*rows, {**rows[0], "example_id": "leak", "probe_split": "test"}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(leaked)
        labels = torch.tensor([row["target_token_id"] for row in rows])
        informative = torch.nn.functional.one_hot(labels, 4).float().unsqueeze(1)
        constant = torch.zeros_like(informative)
        features = {"label": StateFeatures(informative, constant), "initial": StateFeatures(constant, constant)}
        summary, readouts = fit_probes(rows, features, groups={"layer-0": [0]}, strengths=[0.1, 1], shuffled_controls=5, seed=73)
        self.assertEqual(summary["probes"]["label/layer-0/weights"]["test_accuracy"], 1)
        self.assertLess(summary["probes"]["label/layer-0/weights"]["shuffled_mean"], 0.6)
        self.assertEqual(summary["probes"]["initial/layer-0/weights"]["test_accuracy"], 0.25)
        self.assertEqual(readouts["label/layer-0/both"]["weights"].shape, (9, 4))

    def test_probe_main_scores_fits_and_resumes_without_duplicate_rows(self):
        import yaml
        rows = []
        for family, split in enumerate(("train", "val", "test")):
            for original in examples():
                rows.append({**original, "example_id": f"{family}-{original['example_id']}",
                    "context_family": family, "probe_split": split,
                    "input_ids": original["input_ids"][:5] + [6, 7] * 9 + [8, 2],
                    "final_query_position": 23})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": "fixture", "tokenizer_revision": "a" * 40}))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"dataset": {"path": str(dataset)}, "seed": 73,
                "projection_dim": 4, "execution_block_size": 3, "save_traces": True,
                "save_memory_states": True, "shuffled_controls": 1,
                "ridge_strengths": [0.1, 1.0], "layer_groups": {"lower": [0], "upper": [1]}}))
            output = root / "out"
            with (patch("sys.argv", ["probe", "--config", str(config), "--checkpoint", str(root), "--output-dir", str(output)]),
                  patch.object(state_readout_probe.torch.cuda, "is_available", return_value=True),
                  patch.object(state_readout_probe, "checkpoint_fingerprint", return_value="fixture"),
                  patch.object(state_readout_probe, "load_model", return_value=self.model),
                  patch.object(self.model, "to", return_value=self.model),
                  patch.object(state_readout_probe.AutoTokenizer, "from_pretrained", return_value=LifecycleTokenizer()),
                  redirect_stdout(io.StringIO())):
                state_readout_probe.main()
                first = (output / "summary.json").read_text()
                state_readout_probe.main()
            self.assertEqual((output / "summary.json").read_text(), first)
            self.assertEqual(len((output / "results.jsonl").read_text().splitlines()), 18)
            summary = json.loads(first)
            self.assertEqual(len(summary["model_query_scores"]), 9)
            self.assertEqual(summary["split_counts"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(len(json.loads((output / "memory_traces/manifest.json").read_text())["episodes"]), 18)


if __name__ == "__main__":
    unittest.main()
