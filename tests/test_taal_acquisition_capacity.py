import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_min_capacity import build_capacity_examples
from evaluation.taal.acquisition_diagnostic import (
    STAGES, acquisition_rows, collect_acquisition, opposing_donors,
)
from evaluation.taal.trace_export import TaalTraceExporter
from evaluation.taal.training_trace import export_training_traces
from tests import test_taal_state_audit
from training.config import load_config
from training.engine import Progress, build_scheduler, train
from training.stopping import reached_validation_target
from training.train import configure_trainable_parameters
from data.dataloader import create_dataloader


class AcquisitionCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        cls.examples = build_capacity_examples(cls.tokenizer)
        cls.root = Path(__file__).resolve().parents[1]

    def model(self):
        torch.manual_seed(42)
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.vocab_size = len(self.tokenizer)
        config.working_memory_size = 32
        model = TaalQwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.taal.residual_gate.fill_(0.05)
        return model

    def test_minimum_pair_differs_only_at_source_label(self):
        own, donor = self.examples
        self.assertEqual([row["answer"] for row in self.examples], ["amber", "anchor"])
        differences = [position for position, (a, b) in enumerate(zip(own["input_ids"], donor["input_ids"])) if a != b]
        self.assertEqual(differences, [own["fact_position"]])
        self.assertEqual(own["input_ids"][own["final_query_position"]:], donor["input_ids"][donor["final_query_position"]:])
        self.assertEqual(len(own["input_ids"]) - own["fact_position"], 64)
        self.assertEqual(len(own["candidate_token_ids"]), 2)
        self.assertGreater(own["final_query_position"] - own["fact_position"], 32)
        self.assertEqual(opposing_donors(self.examples)[own["example_id"]], donor)
        with self.assertRaisesRegex(ValueError, "opposing-label"):
            opposing_donors(self.examples[:1])

    def test_config_budget_trace_and_convergence_gate(self):
        for name, count in (("single", 1), ("paired", 2)):
            config = load_config(str(self.root / f"training/configs/taal/qwen3_0_6b_capacity_{name}.yaml"))
            self.assertEqual(config.data.train.end, count)
            self.assertEqual(config.data.val.end, count)
            self.assertEqual(config.model.memory_chunk_size, 1)
            self.assertEqual(config.model.working_memory_size, 32)
            self.assertEqual(config.training.gradient_accumulation_steps, 1)
            self.assertEqual(config.training.max_steps, 1000)
            self.assertEqual(config.scheduler.schedule_steps, 1000)
            self.assertEqual(config.scheduler.final_lr, config.optimizer.lr)
            self.assertTrue(config.training.save_traces)
            metric = config.training.early_stop_accuracy_metric
            self.assertTrue(reached_validation_target({"val/loss": 0.09, metric: 1.0}, config.training))
            self.assertFalse(reached_validation_target({"val/loss": 0.2, metric: 1.0}, config.training))
            self.assertFalse(reached_validation_target({"val/loss": 0.09, metric: 0.5}, config.training))
            self.assertFalse(reached_validation_target({"val/loss": float("nan"), metric: 1.0}, config.training))
            config.training.early_stop_accuracy_metric = None
            with self.assertRaisesRegex(ValueError, "both"):
                config.validate()
        old = load_config(str(self.root / "training/configs/taal/qwen3_0_6b_conflict_gate_start.yaml"))
        self.assertFalse(reached_validation_target({}, old.training))

    @torch.inference_mode()
    def test_acquisition_native_parity_controls_and_compact_export(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            exporter = TaalTraceExporter(Path(directory) / "memory_traces", run_metadata={
                "checkpoint": "test", "tokenizer": MODEL_ID, "layers": [0],
            })
            records = [collect_acquisition(model, self.tokenizer, example, exporter, 32) for example in self.examples]
            self.assertEqual(len(exporter.manifest["episodes"]), 2)
            for record in records:
                # Staging prefill must not alter the native prediction.
                native = model.prefill(torch.tensor([record.example["input_ids"]]), execution_block_size=32)
                self.assertEqual(int(native.logits[0, -1].argmax()), record.outcome["vocabulary_top_token_id"])
                self.assertEqual(set(record.snapshots), {"after_fact", "middle", "pre_query"})
                self.assertEqual(set(record.label_writes), {0})
            rows, vectors = acquisition_rows(model, records[0], records[1])
            self.assertEqual(len(rows), len(STAGES) * 2)
            self.assertEqual({row["stage"] for row in rows}, set(STAGES))
            self.assertEqual({row["address"] for row in rows}, {"own", "opposing"})
            self.assertTrue(all(row["native_update_replay_exact"] for row in rows))
            self.assertTrue(all(row["value_pair_distance"] > 0 for row in rows))
            before = next(row for row in rows if row["stage"] == "before_label" and row["address"] == "own")
            fresh = next(row for row in rows if row["stage"] == "fresh_only" and row["address"] == "own")
            self.assertLess(fresh["mse"], before["mse"])
            json.dumps(vectors, allow_nan=False)
            for relative in exporter.manifest["episodes"].values():
                with gzip.open(exporter.root / relative, "rt") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["tokens"][-1]["phase"], "generated")
                self.assertEqual(len(payload["reads"]), len(payload["tokens"]))
                self.assertEqual(len(payload["writes"]), len(payload["tokens"]) + model.config.num_persistent_tokens)

    def test_training_step_validation_and_all_selected_trace_export(self):
        model = self.model().train()
        config = load_config(str(self.root / "training/configs/taal/qwen3_0_6b_capacity_paired.yaml"))
        config.training.max_steps = 1
        config.training.eval_every_steps = 1
        parameters = configure_trainable_parameters(model, "taal_only")
        optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.lr)
        scheduler = build_scheduler(optimizer, config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "capacity.jsonl"
            dataset.write_text("".join(json.dumps(example) + "\n" for example in self.examples))
            config.data.dataset_id = str(dataset)
            train_loader = create_dataloader(config.data, config.data.train, config.loss, shuffle=True, seed=42)
            val_loader = create_dataloader(config.data, config.data.val, config.loss, shuffle=False, seed=42)
            metrics = []
            progress = train(model, train_loader, val_loader, optimizer, scheduler, config,
                             torch.device("cpu"), metrics.append, lambda: False)
            self.assertEqual(progress.step, 1)
            self.assertEqual(progress.skipped_updates, 0)
            self.assertEqual(metrics[-1]["val/records"], 2)
            self.assertIn(config.training.early_stop_accuracy_metric, metrics[-1])
            export_training_traces(model, config, root, progress)
            self.assertTrue(model.training)
            manifest = json.loads((root / "memory_traces/manifest.json").read_text())
            self.assertEqual(len(manifest["episodes"]), 2)
            for relative in manifest["episodes"].values():
                with gzip.open(root / "memory_traces" / relative, "rt") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["tokens"][-1]["phase"], "generated")
                json.dumps(payload, allow_nan=False)

    @torch.inference_mode()
    def test_bf16_slow_weights_fp32_fast_state_exact_write_replay(self):
        model = self.model().to(dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            exporter = TaalTraceExporter(Path(directory) / "memory_traces", run_metadata={
                "checkpoint": "bf16-test", "tokenizer": MODEL_ID, "layers": [0],
            })
            own, donor = [collect_acquisition(model, self.tokenizer, example, exporter, 32)
                          for example in self.examples]
            rows, vectors = acquisition_rows(model, own, donor)
            self.assertTrue(all(row["native_update_replay_exact"] for row in rows))
            json.dumps(vectors, allow_nan=False)
            for state in own.snapshots.values():
                self.assertEqual(state[0].weights["layers.0.weight"].dtype, torch.float32)

    def test_short_launchers_and_eval_trace_contract(self):
        launchers = (
            "training/sbatch/taal/fs_qwen_capacity_short.sbatch",
            "evaluation/sbatch/taal/fs_qwen_acquisition_short.sbatch",
            "evaluation/sbatch/taal/fs_qwen_capacity_audit_short.sbatch",
        )
        for name in launchers:
            script = (self.root / name).read_text()
            self.assertIn("#SBATCH --partition=gpu-short", script)
            self.assertIn("#SBATCH --gres=gpu:h200:1", script)
            self.assertIn("#SBATCH --time=02:00:00", script)
            self.assertIn("utils.explorer_preflight", script)
            self.assertIn("git rev-parse HEAD", script)
            self.assertIn("http_proxy", script)
        for name in ("acquisition_diagnostic", "min_capacity_audit"):
            config = yaml.safe_load((self.root / f"evaluation/configs/taal/{name}.yaml").read_text())
            self.assertTrue(config["save_traces"])


if __name__ == "__main__":
    unittest.main()
