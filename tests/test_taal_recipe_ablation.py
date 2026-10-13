"""Matched LR, accumulation and attention-window experiment contracts."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import gzip
import io
import json
import math
from pathlib import Path
import subprocess
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
from evaluation.taal import state_audit
from evaluation.taal.interventions import audit_conditions
from evaluation.taal.training_trace import export_training_traces
from tests import test_taal_state_audit
from tests.test_training import TinyLoader, TinyModel, batch
from training.checkpoints import ModelCheckpoints
from training.config import CausalLMLossConfig, load_config
from training.engine import Progress, build_scheduler, train
from training.train import configure_trainable_parameters, initialize_taal_gates


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("reference", "cosine150", "ga4", "kv128")


def recipe(name):
    return load_config(ROOT / f"training/configs/taal/qwen3_0_6b_recipe_{name}.yaml")


class TaalRecipeAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        cls.examples = build_capacity_examples(cls.tokenizer, pairs=8, gap=192)

    def test_reference_preserves_d_except_explicit_matching_controls(self):
        expected = load_config(ROOT / "training/configs/taal/qwen3_0_6b_capacity_d_distance_diversity.yaml")
        actual = recipe("reference")
        expected.training.max_steps = 400
        expected.training.eval_every_steps = 100
        expected.training.early_stop_val_loss = None
        expected.training.early_stop_accuracy_metric = None
        expected.checkpoints.save_steps = [100]
        expected.wandb.name = actual.wandb.name
        self.assertEqual(actual, expected)

    def test_cells_change_only_named_factor_and_derived_budget(self):
        reference = recipe("reference")
        for cell in CELLS:
            with self.subTest(cell=cell):
                actual = recipe(cell)
                expected = copy.deepcopy(reference)
                expected.wandb.name = actual.wandb.name
                if cell == "cosine150":
                    expected.scheduler.schedule_steps = 150
                    expected.scheduler.final_lr = 1e-5
                elif cell == "ga4":
                    expected.training.gradient_accumulation_steps = 4
                    expected.training.max_steps = 100
                    expected.training.eval_every_steps = 25
                elif cell == "kv128":
                    expected.model.working_memory_size = 128
                self.assertEqual(actual, expected)
                self.assertEqual(actual.training.max_steps * actual.training.gradient_accumulation_steps, 400)
                self.assertEqual(actual.training.eval_every_steps * actual.training.gradient_accumulation_steps, 100)
                self.assertIsNone(actual.training.early_stop_val_loss)
                self.assertEqual(actual.model.memory_chunk_size, 1)
                self.assertEqual(actual.model.max_persistent_kv_tokens, 0)
                self.assertTrue(actual.training.save_traces)
                self.assertEqual(actual.data.train.end, 16)
                self.assertEqual(actual.data.val.end, 16)
        audit = yaml.safe_load((ROOT / "evaluation/configs/taal/recipe_matched_audit.yaml").read_text())
        self.assertEqual(audit["dataset"]["path"], reference.data.dataset_id)
        self.assertEqual(audit["execution_block_size"], 32)
        self.assertEqual(audit["swap_group_field"], "record_name")
        self.assertTrue(audit["save_traces"])
        self.assertEqual(len(audit_conditions("standard")), 6)

    def test_learning_rates_preserve_warmup_and_cosine_reaches_floor(self):
        curves = {}
        for cell in CELLS:
            config = recipe(cell)
            parameter = torch.nn.Parameter(torch.zeros(()))
            optimizer = torch.optim.SGD([parameter], lr=config.optimizer.lr)
            scheduler = build_scheduler(optimizer, config)
            rates = []
            for _ in range(400):
                rates.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
            self.assertAlmostEqual(rates[0], 1e-5)
            self.assertAlmostEqual(rates[9], 1e-4)
            curves[cell] = rates
        self.assertEqual(curves["reference"], curves["ga4"])
        self.assertEqual(curves["reference"], curves["kv128"])
        self.assertTrue(all(math.isclose(rate, 1e-4) for rate in curves["reference"][9:]))
        self.assertTrue(all(math.isclose(rate, 1e-5) for rate in curves["cosine150"][149:]))
        self.assertLess(curves["cosine150"][99], curves["reference"][99])

    def test_accumulation_averages_independent_microbatches_not_session_states(self):
        for ga in (1, 4):
            with self.subTest(ga=ga):
                config = recipe("reference")
                # Use the scalar causal-LM fixture to measure the exact update.
                config.loss = CausalLMLossConfig()
                config.training.gradient_accumulation_steps = ga
                config.training.max_steps = 1
                config.training.max_grad_norm = 100
                config.training.eval_at_end = False
                model = TinyModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                progress = train(model, TinyLoader([batch(value) for value in range(1, ga + 1)]),
                                 TinyLoader([]), optimizer, scheduler, config,
                                 torch.device("cpu"), lambda _: None, lambda: False)
                expected_gradient = 2 * (.5 - (ga + 1) / 2)
                self.assertAlmostEqual(model.weight.item(), .5 - .01 * expected_gradient, places=6)
                self.assertEqual(progress.step, 1)
                self.assertEqual(progress.tokens_seen, ga * 4)
                self.assertEqual(len(model.calls), ga)
                self.assertTrue(all(state is None and cache is False for state, cache, _, _ in model.calls))

    def test_milestones_validate_save_reload_and_do_not_replace_best(self):
        config = recipe("reference")
        for steps in (None, [True], [0], [401], [100, 100], [99]):
            with self.subTest(steps=steps):
                invalid = copy.deepcopy(config)
                invalid.checkpoints.save_steps = steps
                with self.assertRaisesRegex(ValueError, "save_steps"):
                    invalid.validate()
        with tempfile.TemporaryDirectory() as directory:
            config.checkpoints.output_dir = directory
            model = TaalQwen3ForCausalLM(test_taal_state_audit.TaalStateAuditTests.config())
            saves = ModelCheckpoints(config, "fixture")
            saves.on_validation(model, SimpleNamespace(step=99), {"val/loss": float("nan")})
            # Real dataclass metadata, finite complete validation at selected step.
            with patch.object(saves, "save", wraps=saves.save) as write:
                saves.on_validation(model, Progress(step=100), {"val/loss": 1.0})
                saves.on_validation(model, Progress(step=100), {"val/loss": 1.0})
                self.assertEqual([call.args[2] for call in write.call_args_list], ["best", "step-100"])
            milestone = Path(directory) / "fixture/step-100"
            metadata = json.loads((milestone / "training_metadata.json").read_text())
            self.assertEqual(metadata["progress"]["step"], 100)
            self.assertIn("No optimizer", metadata["contents"])
            restored = TaalQwen3ForCausalLM.from_pretrained(milestone)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            self.assertEqual(saves.best_metric_value, 1.0)

    def test_matched_launcher_refuses_wrong_or_exception_parent_step(self):
        launcher = (ROOT / "evaluation/sbatch/taal/fs_qwen_recipe_matched_audit_short.sbatch").read_text()
        for field in ("#SBATCH --partition=gpu-short", "#SBATCH --gres=gpu:h200:1",
                      "#SBATCH --time=02:00:00", "HF_HOME", "http_proxy", "https_proxy",
                      "utils.explorer_preflight", "git rev-parse HEAD"):
            self.assertIn(field, launcher)
        guard = launcher.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"progress": {"step": 400, "skipped_updates": 0},
                        "reason": "completed", "training_config": recipe("reference").to_dict()}
            path = Path(directory) / "training_metadata.json"
            path.write_text(json.dumps(metadata))
            result = subprocess.run(["python3", "-c", guard, directory, "400"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"successful_supervised_examples": 400', result.stdout)
            for step, reason in ((399, "signal"), (400, "exception")):
                metadata["progress"]["step"] = step
                metadata["reason"] = reason
                path.write_text(json.dumps(metadata))
                rejected = subprocess.run(["python3", "-c", guard, directory, "400"], capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)

    def test_real_cpu_training_export_kv_eviction_and_all_audit_modes(self):
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp,
        )
        audit_recipe = yaml.safe_load((ROOT / "evaluation/configs/taal/recipe_matched_audit.yaml").read_text())
        initial = None
        for cell in CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "data.jsonl"
                dataset.write_text("".join(json.dumps(row) + "\n" for row in self.examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION}))
                config = recipe(cell)
                config.data.dataset_id = str(dataset)
                config.training.max_steps = 1
                config.training.eval_every_steps = 1
                config.checkpoints.save_steps = [1]
                torch.manual_seed(config.training.seed)
                model_config = test_taal_state_audit.TaalStateAuditTests.config()
                model_config.vocab_size = len(self.tokenizer)
                model_config.working_memory_size = config.model.working_memory_size
                model = TaalQwen3ForCausalLM(model_config).train()
                initialize_taal_gates(model, config.training.initial_taal_gate)
                if initial is None:
                    initial = copy.deepcopy(model.state_dict())
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
                parameters = configure_trainable_parameters(model, "taal_only")
                optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.lr)
                scheduler = build_scheduler(optimizer, config)
                loader = create_dataloader(config.data, config.data.train, config.loss, shuffle=True, seed=42)
                val = create_dataloader(config.data, config.data.val, config.loss, shuffle=False, seed=42)
                config.checkpoints.output_dir = str(root)
                checkpoints = ModelCheckpoints(config, "training")
                metrics = []
                progress = train(model, loader, val, optimizer, scheduler, config, torch.device("cpu"),
                                 metrics.append, lambda: False, on_validation=checkpoints.on_validation)
                self.assertEqual(progress.step, 1)
                self.assertEqual(progress.tokens_seen, 207 * config.training.gradient_accumulation_steps)
                self.assertEqual(progress.skipped_updates, 0)
                self.assertEqual(metrics[-1]["val/records"], 16)
                self.assertIn("val/loss_memory_read_scale_0", metrics[-1])
                export_training_traces(model, config, root / "training", progress)
                native = json.loads((root / "training/memory_traces/manifest.json").read_text())
                self.assertEqual(len(native["episodes"]), 16)
                with torch.inference_mode():
                    model.eval()
                    prefix, _ = state_audit.split_episode(self.examples[0])
                    state = model.prefill(torch.tensor([prefix]), execution_block_size=32).state
                    retained = state.past_key_values.layers[0].positions.tolist()
                    self.assertTrue(set(retained).isdisjoint(range(21)))
                    self.assertLessEqual(len(retained), config.model.working_memory_size)
                settings = copy.deepcopy(audit_recipe)
                settings["dataset"]["path"] = str(dataset)
                path = root / "audit.yaml"
                path.write_text(yaml.safe_dump(settings))
                output = root / "audit"
                argv = ["audit", "--config", str(path), "--checkpoint", str(root), "--output-dir", str(output)]
                with (patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                      patch.object(state_audit, "load_dataset", return_value=self.examples),
                      patch.object(state_audit, "checkpoint_fingerprint", return_value="CPU-fixture"),
                      patch.object(state_audit, "load_model", return_value=model),
                      patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                      redirect_stdout(io.StringIO())):
                    state_audit.main()
                    state_audit.main()
                rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual(Counter(row["audit_condition"] for row in rows), {condition.name: 16 for condition in audit_conditions("standard")})
                self.assertEqual(len({row["evaluation_id"] for row in rows}), 96)
                manifest = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 96)
                self.assertEqual(len(manifest["comparisons"]), 80)
                for relative in manifest["episodes"].values():
                    with gzip.open(output / "memory_traces" / relative, "rt") as handle:
                        episode = json.load(handle)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(episode["tokens"][-1]["token_id"], episode["outcome"]["vocabulary_top_token_id"])
                    self.assertEqual(len(episode["reads"]), 207)
                    self.assertEqual(len(episode["writes"]), 207 + model_config.num_persistent_tokens)
                    json.dumps(episode, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
