"""Workshop contracts, exact checkpoint initialization and real CPU smoke paths."""

from collections import Counter
from contextlib import redirect_stdout
import copy
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
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from data.dataloader import create_dataloader
from data.taal_workshop import ROOT as DATA, MODEL_ID, MODEL_REVISION, SOURCES, build_bundle, jsonl
from evaluation.taal import state_audit
from evaluation.taal.prefix_store import PrefixStore
from evaluation.taal.training_trace import export_training_traces
from scripts import taal_workshop as workshop
from tests import test_taal_state_audit
from training.checkpoints import ModelCheckpoints
from training.config import load_config
from training.engine import build_scheduler, train
from training.optimization import build_optimizer
from training.train import configure_trainable_parameters, load_model


PROJECT = Path(__file__).resolve().parents[1]


def recipe(arm):
    return load_config(PROJECT / f"training/configs/taal/workshop_{arm}.yaml")


class WorkshopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        # Source-independent fixture proves construction without downloading corpora.
        filler = cls.tokenizer.encode(" The river flowed past the village while people prepared their evening meal.", add_special_tokens=False)
        pools = {}
        for source in SOURCES:
            pools[source] = [{"document_sha256": f"{source}-{index:03}", "input_ids": filler * 50,
                              "source": source, "source_record": index, "document_index": 0}
                             for index in range(30)]
        cls.files, cls.metadata = build_bundle(cls.tokenizer, pools)

    def test_recipe_changes_only_context_and_name_between_arms(self):
        narrow, diverse = recipe("narrow"), recipe("diverse")
        expected = copy.deepcopy(narrow)
        expected.data.dataset_id = diverse.data.dataset_id
        expected.wandb.name = diverse.wandb.name
        self.assertEqual(diverse, expected)
        self.assertEqual(narrow.training.max_steps, 200)
        self.assertEqual(narrow.checkpoints.save_steps, [100, 200])
        self.assertIsNone(narrow.training.initial_taal_gate)
        self.assertIsNone(narrow.training.early_stop_val_loss)
        self.assertTrue(narrow.training.save_traces)
        self.assertEqual(narrow.model.memory_max_write_grad_norm, 1.0)
        self.assertEqual(narrow.model.memory_chunk_size, 1)
        self.assertEqual(narrow.model.working_memory_size, 32)
        narrow.training.initial_taal_gate = 0.05
        with self.assertRaisesRegex(ValueError, "preserve learned gates"):
            narrow.validate()

    def test_training_masks_order_and_development_are_matched(self):
        narrow, diverse = self.files["narrow.jsonl"], self.files["diverse.jsonl"]
        self.assertEqual(narrow[128:], diverse[128:])
        self.assertEqual([row["target_token_id"] for row in narrow], [row["target_token_id"] for row in diverse])
        self.assertEqual(len({tuple(row["input_ids"]) for row in narrow[:128]}), 16)
        self.assertGreater(len({tuple(row["input_ids"]) for row in diverse[:128]}), 16)
        split_hashes = {split: {row["document_sha256"] for row in rows} for split, rows in self.metadata["documents"].items()}
        self.assertFalse(split_hashes["train"] & split_hashes["dev"])
        self.assertFalse(split_hashes["train"] & split_hashes["test"])
        self.assertFalse(split_hashes["dev"] & split_hashes["test"])
        with tempfile.TemporaryDirectory() as directory:
            sequences = []
            for arm in ("narrow", "diverse"):
                path = Path(directory) / f"{arm}.jsonl"
                path.write_text(jsonl(self.files[f"{arm}.jsonl"]))
                settings = recipe(arm)
                settings.data.dataset_id = str(path)
                loader = create_dataloader(settings.data, settings.data.train, settings.loss, shuffle=True, seed=42)
                sequence = []
                for batch in loader:
                    target = int(batch["target_token_ids"][0])
                    sequence.append(target)
                    self.assertEqual(int((batch["delayed_labels"] != -100).sum()), 1)
                    self.assertEqual(int(batch["delayed_labels"][0, -1]), target)
                    self.assertEqual(int(batch["input_ids"][0, -1]), target)
                    self.assertNotIn("labels", batch)
                self.assertEqual(len(sequence), 128)
                sequences.append(sequence)
            self.assertEqual(*sequences)

    def test_selection_preserves_label_set_swaps_assignment_and_reuses_prefix(self):
        examples = self.files["evaluation.jsonl"]
        self.assertEqual(Counter(row["condition"] for row in examples),
                         {"reference": 16, "filler": 8, "position": 8, "distance": 8, "combined": 8, "selection": 16})
        donors = state_audit.swap_sources(examples, "pair_group")
        aliases = state_audit.shared_prefix_examples(examples, "prefix_group")
        for row in examples:
            prefix, query = state_audit.split_episode(row)
            self.assertTrue(query)
            self.assertTrue(all(end <= len(prefix) - 32 for _, end in row["fact_spans"]))
            donor = donors[row["example_id"]]
            self.assertNotEqual(row["target_token_id"], donor["target_token_id"])
            self.assertEqual(query, state_audit.split_episode(donor)[1])
            if row["fact_count"] == 2:
                self.assertEqual(len(aliases[row["example_id"]]), 2)
                self.assertEqual(Counter(row["input_ids"]), Counter(donor["input_ids"]))
                differences = [i for i, (a, b) in enumerate(zip(row["input_ids"], donor["input_ids"])) if a != b]
                self.assertEqual(differences, row["fact_positions"])
        bad = copy.deepcopy(examples[-4:])
        bad[1]["input_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "identical prefix"):
            state_audit.shared_prefix_examples(bad, "prefix_group")

    def tiny_parent(self, directory):
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.vocab_size = len(self.tokenizer)
        config.working_memory_size = 32
        model = TaalQwen3ForCausalLM(config)
        with torch.no_grad():
            model.model.layers[0].taal.residual_gate.fill_(0.1234)
        model.save_pretrained(directory)
        settings = recipe("narrow")
        settings.training.initialize_from_checkpoint = str(directory)
        for name in asdict(settings.model):
            if name not in ("model_id", "revision", "memory_max_write_grad_norm"):
                setattr(settings.model, name, getattr(config, name))
        return model, settings

    def test_warm_start_preserves_every_parameter_and_rejects_architecture_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, settings = self.tiny_parent(Path(directory))
            initialized = load_model(settings)
            for name, value in parent.state_dict().items():
                torch.testing.assert_close(initialized.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(initialized.model.layers[0].taal.neural_memory.config.max_write_grad_norm, 1.0)
            self.assertEqual(initialized.model.layers[0].taal.residual_gate.item(), parent.model.layers[0].taal.residual_gate.item())
            settings.model.working_memory_size = 64
            with self.assertRaisesRegex(ValueError, "changes working_memory_size"):
                load_model(settings)

    def test_real_cpu_continuation_backward_checkpoint_and_trace_export(self):
        for arm in ("narrow", "diverse"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent, settings = self.tiny_parent(root / "parent")
                rows = copy.deepcopy(self.files[f"{arm}.jsonl"])
                # Two real examples and the same two held-out records exercise
                # optimizer/backprop/checkpoints without pretending to test convergence.
                selected = [rows[16], rows[17], rows[128], rows[129]]
                for index, row in enumerate(selected):
                    row["fact_id"] = index
                dataset = root / "data.jsonl"
                dataset.write_text(jsonl(selected))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION}))
                settings.data.dataset_id = str(dataset)
                settings.data.train.end = 2
                settings.data.val.start, settings.data.val.end = 2, 4
                settings.training.max_steps = 2
                settings.training.eval_every_steps = 1
                settings.checkpoints.save_steps = [1, 2]
                settings.checkpoints.output_dir = str(root / "checkpoints")
                model = load_model(settings).train()
                parameters = configure_trainable_parameters(model, "taal_only")
                before = [parameter.detach().clone() for parameter in parameters]
                optimizer = build_optimizer(model, settings)
                self.assertFalse(optimizer.state)
                scheduler = build_scheduler(optimizer, settings)
                train_loader = create_dataloader(settings.data, settings.data.train, settings.loss, True, 42)
                val_loader = create_dataloader(settings.data, settings.data.val, settings.loss, False, 42)
                checkpoints = ModelCheckpoints(settings, arm)
                progress = train(model, train_loader, val_loader, optimizer, scheduler, settings,
                                 torch.device("cpu"), lambda _: None, lambda: False,
                                 on_validation=checkpoints.on_validation)
                self.assertEqual(progress.step, 2)
                self.assertEqual(progress.skipped_updates, 0)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, parameters)))
                for name, value in model.named_parameters():
                    if ".taal." not in name:
                        torch.testing.assert_close(value, parent.state_dict()[name], rtol=0, atol=0)
                self.assertTrue((checkpoints.directory / "step-2/model.safetensors").is_file())
                export_training_traces(model, settings, checkpoints.directory, progress)
                manifest = json.loads((checkpoints.directory / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 2)

    def test_cpu_audit_clips_exports_generated_tokens_and_reuses_actual_state(self):
        examples = self.files["evaluation.jsonl"][-4:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(jsonl(examples))
            dataset.with_suffix(".metadata.json").write_text(json.dumps({"tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION}))
            settings = yaml.safe_load((PROJECT / "evaluation/configs/taal/workshop.yaml").read_text())
            settings["dataset"] = {"path": str(dataset)}
            config_path = root / "audit.yaml"
            config_path.write_text(yaml.safe_dump(settings))
            model, _ = self.tiny_parent(root / "parent")
            model.eval()
            cpu_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
                                        device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
                                        long=torch.long, logsumexp=torch.logsumexp)
            output_dir = root / "audit"
            argv = ["audit", "--config", str(config_path), "--checkpoint", str(root / "parent"), "--output-dir", str(output_dir)]
            prefix_calls = []
            forward = model.prefill

            def record_prefill(input_ids, **kwargs):
                if kwargs.get("state") is None:
                    prefix_calls.append(input_ids.tolist())
                return forward(input_ids, **kwargs)

            with (patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                  patch.object(state_audit, "load_dataset", return_value=examples),
                  patch.object(state_audit, "load_model", return_value=model),
                  patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                  patch.object(model, "prefill", side_effect=record_prefill), redirect_stdout(io.StringIO())):
                state_audit.main()
                state_audit.main()
            self.assertEqual(len(prefix_calls), 2)  # Four questions, exactly two pre-query states.
            summary = workshop.summarize_audit(output_dir, examples)
            self.assertEqual(summary["rows"], 24)
            self.assertEqual(summary["selection_groups"], 1)
            prefix_manifest = json.loads((output_dir / "prefix_states/manifest.json").read_text())
            store = PrefixStore(output_dir / "prefix_states", {k: v for k, v in prefix_manifest.items()
                                                             if k not in ("schema", "torch_version", "transformers_version")})
            first, second = store.load(examples[0], "cpu"), store.load(examples[1], "cpu")
            torch.testing.assert_close(first.state.past_key_values.layers[0].keys, second.state.past_key_values.layers[0].keys, rtol=0, atol=0)
            retained = first.state.past_key_values.layers[0].positions.tolist()
            self.assertTrue(all(position not in retained for start, end in examples[0]["fact_spans"] for position in range(start, end)))
            manifest = json.loads((output_dir / "memory_traces/manifest.json").read_text())
            for relative in manifest["episodes"].values():
                with gzip.open(output_dir / "memory_traces" / relative, "rt") as handle:
                    trace = json.load(handle)
                self.assertEqual(trace["tokens"][-1]["phase"], "generated")
                self.assertEqual(trace["tokens"][-1]["token_id"], trace["outcome"]["vocabulary_top_token_id"])
                self.assertTrue(all(event["proposed_write_norm"] <= 1.00001 for event in trace["writes"]))
                json.dumps(trace, allow_nan=False)

    def test_slurm_accepts_both_gpu_families_and_bundles_dependencies(self):
        for kind in ("training", "evaluation"):
            path = PROJECT / kind / "sbatch/taal/fs_qwen_workshop_short.sbatch"
            text = path.read_text()
            subprocess.run(["bash", "-n", str(path)], check=True)
            self.assertIn("--gres=gpu:1", text)
            self.assertIn("--mem=48G", text)
            self.assertIn("--time=02:00:00", text)
            self.assertNotIn("--gres=gpu:h200", text)
            self.assertIn("git rev-parse HEAD", text)
            self.assertIn("titans-impl", text)

    def test_full_context_host_reference_has_clickable_prediction_and_no_fake_memory(self):
        config = Qwen3Config(vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4)
        model = Qwen3ForCausalLM(config).eval()
        rows = self.files["evaluation.jsonl"][-4:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workshop.score_full_context_reference(model, self.tokenizer, rows, root)
            workshop.score_full_context_reference(model, self.tokenizer, rows, root)
            manifest = json.loads((root / "memory_traces/manifest.json").read_text())
            self.assertEqual(len(workshop.read_jsonl(root / "results.jsonl")), 4)
            for relative in manifest["episodes"].values():
                with gzip.open(root / "memory_traces" / relative, "rt") as handle:
                    episode = json.load(handle)
                self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                self.assertEqual(episode["tokens"][-1]["token_id"], episode["outcome"]["vocabulary_top_token_id"])
                self.assertEqual(episode["reads"], [])
                self.assertEqual(episode["writes"], [])

    def test_continuation_guard_preserves_skips_and_rejects_wrong_step(self):
        settings = recipe("narrow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            progress = {"step": 200, "skipped_updates": 2, "nan_losses": 0, "nan_gradients": 0,
                        "infinite_losses": 0, "infinite_gradients": 2, "tokens_seen": 42000}
            metadata = {"progress": progress, "reason": "matched optimizer-step snapshot", "training_config": settings.to_dict()}
            (path / "training_metadata.json").write_text(json.dumps(metadata))
            (path / "config.json").write_text(json.dumps({"memory_max_write_grad_norm": 1.0}))
            self.assertEqual(workshop.validate_continuation(path, "narrow", 200)["progress"], progress)
            with self.assertRaisesRegex(ValueError, "100 successful"):
                workshop.validate_continuation(path, "narrow", 100)
            with self.assertRaisesRegex(ValueError, "changed data"):
                workshop.validate_continuation(path, "diverse", 200)

    def test_fresh_training_serialization_keeps_prior_checkpoint_identity(self):
        settings = load_config(PROJECT / "training/configs/taal/cpt_A.yaml")
        self.assertNotIn("initialize_from_checkpoint", settings.to_dict()["training"])


if __name__ == "__main__":
    unittest.main()
