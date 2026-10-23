from collections import Counter
from contextlib import redirect_stdout
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from data.dataloader import create_dataloader
from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_min_capacity import build_capacity_examples
from evaluation.taal import normalization_training_audit, state_audit
from evaluation.taal.decoder_diagnostic import FixedReplaySource
from evaluation.taal.interventions import audit_conditions
from evaluation.taal.normalization_ablation import ARMS
from evaluation.taal.training_trace import export_training_traces
from tests import test_taal_longmemeval, test_taal_state_audit
from training.checkpoints import ModelCheckpoints
from training.config import load_config
from training.engine import build_scheduler, train
from training.train import configure_trainable_parameters, initialize_taal_gates


ROOT = Path(__file__).resolve().parents[1]


def recipe(arm):
    return load_config(ROOT / f"training/configs/taal/qwen3_0_6b_norm_{arm}.yaml")


class NormalizationTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        cls.examples = build_capacity_examples(cls.tokenizer, pairs=8, gap=192)

    def test_only_normalization_differs_with_matched_d_budget(self):
        reference = load_config(ROOT / "training/configs/taal/qwen3_0_6b_capacity_d_distance_diversity.yaml")
        reference.training.max_steps = 400
        reference.training.eval_every_steps = 100
        reference.training.early_stop_val_loss = None
        reference.training.early_stop_accuracy_metric = None
        reference.checkpoints.save_steps = [100, 200, 300, 400]
        for arm, (inputs, outputs) in ARMS.items():
            expected = copy.deepcopy(reference)
            actual = recipe(arm)
            expected.model.memory_pre_rmsnorm = inputs
            expected.model.memory_mlp_output_layernorm = outputs
            expected.wandb.name = actual.wandb.name
            self.assertEqual(actual, expected)
            self.assertTrue(actual.training.save_traces)
        audit = yaml.safe_load(normalization_training_audit.AUDIT_CONFIG.read_text())
        self.assertTrue(audit["save_traces"])
        self.assertEqual(audit["swap_group_field"], "record_name")
        self.assertEqual(audit["dataset"]["end"], 16)
        self.assertEqual(len(audit_conditions("standard")), 6)

    def test_guard_rejects_wrong_arm_step_recipe_and_skipped_updates(self):
        settings = recipe("both")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = test_taal_state_audit.TaalStateAuditTests.config()
            for name in ("working_memory_size", "memory_dim", "memory_depth", "memory_chunk_size",
                         "memory_conv_kernel_size", "num_persistent_tokens", "max_persistent_kv_tokens",
                         "memory_pre_rmsnorm", "memory_mlp_output_layernorm"):
                setattr(model_config, name, getattr(settings.model, name))
            model_config.save_pretrained(root)
            metadata = {"progress": {"step": 100, "skipped_updates": 0},
                        "reason": "matched optimizer-step snapshot", "training_config": settings.to_dict()}
            target = root / "training_metadata.json"
            target.write_text(json.dumps(metadata))
            normalization_training_audit.validate_snapshot(root, "both", 100, settings)
            for arm, step in (("input", 100), ("both", 400)):
                with self.assertRaises(ValueError):
                    normalization_training_audit.validate_snapshot(root, arm, step, settings)
            for modified in (metadata | {"reason": "exception"},
                             metadata | {"progress": {"step": 100, "skipped_updates": 1}}):
                target.write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    normalization_training_audit.validate_snapshot(root, "both", 100, settings)
            target.write_text(json.dumps(metadata))
            changed = copy.deepcopy(settings)
            changed.optimizer.lr *= 2
            with self.assertRaises(ValueError):
                normalization_training_audit.validate_snapshot(root, "both", 100, changed)

    def test_missing_milestone_never_falls_back_to_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "final").mkdir()
            with patch.object(normalization_training_audit.subprocess, "run") as launch:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    normalization_training_audit.audit_training_run(root, "baseline", root, root / "audit")
                launch.assert_not_called()
            summary = json.loads((root / "audit/summary.json").read_text())
            self.assertEqual(summary["long_probe"]["status"], "skipped_missing_step_400")

    def test_trained_long_probe_preserves_flags_and_exports_both_phases(self):
        metadata = {"tokenizer": "fixture", "tokenizer_revision": "fixture"}
        settings = {"seed": 42, "save_traces": True, "model": {"memory_read_scale": 1.0},
                    "generation": {"execution_block_size": 32, "max_new_tokens": 256, "do_sample": False}}
        source = FixedReplaySource({"metadata": metadata, "settings": settings, "fingerprint": "old-D"},
            {"question_id": "fixture", "prompt_length": 9, "input_ids": list(range(1, 10)),
             "answer": "fixture", "persistent_prefix_length": 0}, [20, 20], {})
        for arm, (inputs, outputs) in ARMS.items():
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = test_taal_state_audit.TaalStateAuditTests.config()
                config.num_hidden_layers = 4
                config.layer_types = ["full_attention"] * 4
                config.working_memory_size = 32
                config.memory_pre_rmsnorm = inputs
                config.memory_mlp_output_layernorm = outputs
                model = TaalQwen3ForCausalLM(config).eval()
                initialize_taal_gates(model, 0.05)
                config.save_pretrained(root)
                with (patch.object(normalization_training_audit, "load_replay_source", return_value=source),
                      patch.object(normalization_training_audit, "checkpoint_fingerprint", return_value="new-trained"),
                      patch.object(normalization_training_audit.AutoTokenizer, "from_pretrained",
                                   return_value=test_taal_longmemeval.Tokenizer()),
                      patch.object(normalization_training_audit, "load_model", return_value=model) as load,
                      patch.object(model, "to", return_value=model),
                      patch.object(torch.cuda, "get_device_name", return_value="CPU fixture")):
                    reports = normalization_training_audit.run_long_probe(root, arm, root, root / "probe")
                self.assertEqual(load.call_args.args[1].memory_pre_rmsnorm, inputs)
                self.assertEqual(load.call_args.args[1].memory_mlp_output_layernorm, outputs)
                self.assertEqual(reports["fixed"]["status"], "finite_replay")
                self.assertNotIn("reference_comparison", reports["fixed"])
                self.assertEqual(reports["generation"]["status"], "finite_generation")
                for phase in ("fixed", "generation"):
                    self.assertEqual(reports[phase]["trace_status"], "exported")
                    manifest = json.loads((root / f"probe/{phase}/memory_traces/manifest.json").read_text())
                    self.assertEqual(len(manifest["episodes"]), 1)

    def test_cpu_training_reload_and_six_condition_trace_export_all_arms(self):
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp,
        )
        initial = None
        for arm, (inputs, outputs) in ARMS.items():
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "data.jsonl"
                dataset.write_text("".join(json.dumps(row) + "\n" for row in self.examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({
                    "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
                }))
                settings = recipe(arm)
                settings.data.dataset_id = str(dataset)
                settings.training.max_steps = 1
                settings.training.eval_every_steps = 1
                settings.checkpoints.save_steps = [1]
                settings.checkpoints.output_dir = str(root)
                torch.manual_seed(42)
                config = test_taal_state_audit.TaalStateAuditTests.config()
                config.vocab_size = len(self.tokenizer)
                config.working_memory_size = 32
                config.memory_dim = 128
                config.memory_conv_kernel_size = 4
                config.num_persistent_tokens = 8
                config.memory_pre_rmsnorm = inputs
                config.memory_mlp_output_layernorm = outputs
                model = TaalQwen3ForCausalLM(config).train()
                initialize_taal_gates(model, settings.training.initial_taal_gate)
                if initial is None:
                    initial = copy.deepcopy(model.state_dict())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
                parameters = configure_trainable_parameters(model, "taal_only")
                optimizer = torch.optim.AdamW(parameters, lr=settings.optimizer.lr,
                    betas=(settings.optimizer.beta1, settings.optimizer.beta2), weight_decay=0)
                scheduler = build_scheduler(optimizer, settings)
                loader = create_dataloader(settings.data, settings.data.train, settings.loss, shuffle=True, seed=42)
                validation = create_dataloader(settings.data, settings.data.val, settings.loss, shuffle=False, seed=42)
                saves = ModelCheckpoints(settings, "training")
                metrics = []
                progress = train(model, loader, validation, optimizer, scheduler, settings, torch.device("cpu"),
                                 metrics.append, lambda: False, on_validation=saves.on_validation)
                self.assertEqual(progress.step, 1)
                self.assertEqual(progress.skipped_updates, 0)
                self.assertEqual(metrics[-1]["val/records"], 16)
                self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))
                self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in model.state_dict().items()
                                    if ".taal." in name))
                for name, value in model.state_dict().items():
                    if ".taal." not in name:
                        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
                model = state_audit.load_model(root / "training/step-1").float().eval()
                self.assertEqual(model.config.memory_pre_rmsnorm, inputs)
                self.assertEqual(model.config.memory_mlp_output_layernorm, outputs)
                export_training_traces(model, settings, root / "training", progress)
                exported = json.loads((root / "training/memory_traces/manifest.json").read_text())
                self.assertEqual(len(exported["episodes"]), 16)
                audit = yaml.safe_load(normalization_training_audit.AUDIT_CONFIG.read_text())
                audit["dataset"]["path"] = str(dataset)
                config_path = root / "audit.yaml"
                config_path.write_text(yaml.safe_dump(audit))
                output = root / "audit"
                argv = ["audit", "--config", str(config_path), "--checkpoint", str(root / "training/step-1"),
                        "--output-dir", str(output)]
                with (patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                      patch.object(state_audit, "load_dataset", return_value=self.examples),
                      patch.object(state_audit, "load_model", return_value=model),
                      patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                      redirect_stdout(io.StringIO())):
                    state_audit.main()
                rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual(Counter(row["audit_condition"] for row in rows),
                                 {condition.name: 16 for condition in audit_conditions("standard")})
                manifest = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 96)
                self.assertEqual(len(manifest["comparisons"]), 80)
                for relative in manifest["episodes"].values():
                    with gzip.open(output / "memory_traces" / relative, "rt") as stream:
                        episode = json.load(stream)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(len(episode["reads"]), 207)
                    self.assertEqual(len(episode["writes"]), 215)
                    json.dumps(episode, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
