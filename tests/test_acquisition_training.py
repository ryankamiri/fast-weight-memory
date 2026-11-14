"""Retrieval split integrity, family probes and exact interrupted-update continuation."""

from dataclasses import asdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.shared.contracts import Prediction

from data.balanced_retrieval import RetrievalConfig, export_bundle
from data.dataloader import CausalLMCollator, create_dataloader
from data.retrieval import RetrievalDataset, load_retrieval_splits
from evaluation.storage import file_sha256
from training.checkpoints import ModelCheckpoints
from training.config import load_config
from training.engine import Progress, build_scheduler, train
from training.resume import TrainingState
from training.train import configure_trainable_parameters, load_model
from training.retrieval_validation import RetrievalValidation, select_probe_families
from tests.test_balanced_retrieval import tiny_tokenizer


RECIPE = Path("training/configs/taal/retrieval_acquisition.yaml")


def fixture_config(root):
    tokenizer = tiny_tokenizer()
    destination = root / "data"
    export_bundle(tokenizer, RetrievalConfig(
        train_families=2, val_families=2, test_families=1,
        max_sequence_tokens=2048,
    ), destination)
    config = load_config(RECIPE)
    config.data.seq_len = 2048
    config.data.manifest_path = str(destination / "manifest.json")
    config.data.split_files = {name: str(destination / f"{name}.jsonl") for name in ("train", "val")}
    config.checkpoints.output_dir = str(root / "checkpoints")
    config.validation.probe_families = 1
    return config, tokenizer


class StochasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, input_ids, labels, state=None, use_cache=False):
        self.calls.append(input_ids[0, 0].item())
        noise = torch.rand(()) + random.random() + np.random.random()
        return SimpleNamespace(loss=(self.weight * noise - input_ids.float().mean()).square())

    def save_pretrained(self, path, safe_serialization=True):
        torch.save(self.state_dict(), path / "weights.pt")


def reset_rng():
    random.seed(88)
    np.random.seed(88)
    torch.manual_seed(88)


def tiny_training(config):
    model = StochasticModel()
    rows = []
    for index in range(5):
        rows.append({"input_ids": [index + 1, 2, 3], "labels": [-100, 2, 3]})
    dataset = RetrievalDataset(rows, 19, True, {"file": "test-content-hash"})
    loader = DataLoader(dataset, batch_size=1, collate_fn=CausalLMCollator(), generator=torch.Generator().manual_seed(19))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)
    scheduler = build_scheduler(optimizer, config)
    return model, loader, optimizer, scheduler


class AcquisitionTests(unittest.TestCase):
    def test_recipe_is_fresh_and_split_files_do_not_read_test(self):
        with TemporaryDirectory() as directory:
            config, tokenizer = fixture_config(Path(directory))
            Path(directory, "data/test.jsonl").unlink()
            config.validate()
            self.assertEqual(config.model.working_memory_size, 128)
            self.assertEqual(config.model.memory_max_write_grad_norm, 1)
            self.assertEqual(config.training.trainable_parameters, "taal_only")
            self.assertIsNone(config.training.initialize_from_checkpoint)
            self.assertTrue(config.checkpoints.save_training_state)
            loader = create_dataloader(config.data, config.data.train, config.loss, True, 42)
            self.assertEqual(len(list(loader)), 16)
            self.assertEqual(loader.dataset.cursor, 16)
            probes = RetrievalValidation(config, tokenizer, Path(directory) / "probes")
            self.assertEqual(len(probes.rows["train"]), 8)
            self.assertEqual(len(probes.rows["val"]), 8)
            self.assertEqual(len(select_probe_families(loader.dataset.rows, 1)), 8)

    def test_manifest_corruption_and_sequence_overflow_fail_not_filter(self):
        with TemporaryDirectory() as directory:
            config, _ = fixture_config(Path(directory))
            config.data.seq_len = 4
            with self.assertRaisesRegex(ValueError, "silently filter"):
                load_retrieval_splits(config.data)
            config.data.seq_len = 2048
            with Path(config.data.split_files["val"]).open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_retrieval_splits(config.data)

    def test_family_leakage_with_valid_hash_is_rejected(self):
        with TemporaryDirectory() as directory:
            config, _ = fixture_config(Path(directory))
            train_rows, _ = load_retrieval_splits(config.data)
            path = Path(config.data.split_files["val"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["metadata"]["family_id"] = train_rows["train"][0]["metadata"]["family_id"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest_path = Path(config.data.manifest_path)
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["val"]["sha256"] = file_sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_retrieval_splits(config.data)

    def test_resume_matches_uninterrupted_across_epoch_and_partial_accumulation(self):
        with TemporaryDirectory() as directory:
            config = load_config(RECIPE)
            config.training.max_steps = 4
            config.training.gradient_accumulation_steps = 2
            config.training.eval_at_end = False
            config.training.eval_every_steps = 100
            config.checkpoints.output_dir = directory
            reset_rng()
            reference, full_loader, full_optimizer, full_scheduler = tiny_training(config)
            full_progress = train(reference, full_loader, [], full_optimizer, full_scheduler,
                                  config, torch.device("cpu"), lambda _: None, lambda: False)
            expected_rng = (torch.rand(()), random.random(), np.random.random())
            reset_rng()
            partial, loader, optimizer, scheduler = tiny_training(config)
            progress = Progress()
            train(partial, loader, [], optimizer, scheduler, config, torch.device("cpu"),
                  lambda _: None, lambda: len(partial.calls) >= 3, progress=progress)
            self.assertEqual(progress.step, 1)
            self.assertEqual(progress.examples_seen, 2)
            self.assertEqual(loader.dataset.cursor, 2)
            state = TrainingState(optimizer, scheduler, loader)
            checkpoints = ModelCheckpoints(config, "interrupted", state)
            checkpoints.best_metric_value = 0.75
            checkpoints.saved_steps = {1}
            checkpoints.save_final(partial, progress, "signal")
            source = Path(directory) / "interrupted/final"
            resumed, resumed_loader, resumed_optimizer, resumed_scheduler = tiny_training(config)
            resumed.load_state_dict(torch.load(source / "weights.pt", weights_only=True))
            resumed_progress = Progress()
            resumed_state = TrainingState(resumed_optimizer, resumed_scheduler, resumed_loader)
            restored_checkpoints = ModelCheckpoints(config, "resumed", resumed_state)
            resumed_state.restore(source, config, resumed, resumed_progress, restored_checkpoints)
            self.assertEqual(restored_checkpoints.best_metric_value, 0.75)
            self.assertEqual(restored_checkpoints.saved_steps, {1})
            train(resumed, resumed_loader, [], resumed_optimizer, resumed_scheduler, config,
                  torch.device("cpu"), lambda _: None, lambda: False, progress=resumed_progress)
            torch.testing.assert_close(resumed.weight, reference.weight, rtol=0, atol=0)
            self.assertEqual(asdict(resumed_progress), asdict(full_progress))
            self.assertEqual(resumed_loader.dataset.state_dict(), full_loader.dataset.state_dict())
            self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
            torch.testing.assert_close(resumed_loader.generator.get_state(), full_loader.generator.get_state(), rtol=0, atol=0)
            for key, expected in full_optimizer.state_dict()["state"][0].items():
                torch.testing.assert_close(resumed_optimizer.state_dict()["state"][0][key], expected, rtol=0, atol=0)
            self.assertEqual(partial.calls[:2] + resumed.calls, reference.calls)
            torch.testing.assert_close(torch.rand(()), expected_rng[0], rtol=0, atol=0)
            self.assertEqual((random.random(), np.random.random()), expected_rng[1:])
            config.optimizer.lr *= 2
            with self.assertRaisesRegex(ValueError, "recipe"):
                resumed_state.restore(source, config, resumed, resumed_progress, restored_checkpoints)
            config.optimizer.lr /= 2
            with (source / "weights.pt").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resumed_state.restore(source, config, resumed, resumed_progress, restored_checkpoints)

    def test_real_taal_probe_generation_traces_and_restored_mode(self):
        with TemporaryDirectory() as directory:
            config, tokenizer = fixture_config(Path(directory))
            config.validation.probe_max_new_tokens = 1
            model = TaalQwen3ForCausalLM(TaalQwen3Config(
                vocab_size=len(tokenizer), hidden_size=8, intermediate_size=12,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=4, memory_dim=2, memory_depth=2,
                memory_conv_kernel_size=1, memory_chunk_size=1,
                working_memory_size=128, max_persistent_kv_tokens=0,
                num_persistent_tokens=1, memory_max_write_grad_norm=1.0,
            ))
            model.train()
            probes = RetrievalValidation(config, tokenizer, Path(directory) / "run")
            before = torch.get_rng_state().clone()
            metrics = probes(model, Progress(step=1), lambda: False)
            self.assertTrue(model.training)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(metrics["val/probe_questions"], 8)
            self.assertEqual(metrics["val/probe_prefixes"], 4)
            self.assertIn("train/probe_answer_nll", metrics)
            root = Path(directory) / "run/probes/step-1/val"
            self.assertEqual(len(list((root / "token_traces").glob("*.json"))), 8)
            self.assertEqual(len(list((root / "memory_traces/episodes").glob("*.json.gz"))), 8)
            self.assertIsNone(probes(model, Progress(step=2), lambda: True))
            self.assertFalse((Path(directory) / "run/probes/step-2").exists())

    def test_actual_model_checkpoint_reload_preserves_learned_gate(self):
        with TemporaryDirectory() as directory:
            config = load_config(RECIPE)
            config.model.memory_dim = 2
            config.model.memory_conv_kernel_size = 1
            config.model.num_persistent_tokens = 1
            config.checkpoints.output_dir = directory
            model = TaalQwen3ForCausalLM(TaalQwen3Config(
                vocab_size=16, hidden_size=8, intermediate_size=12,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=4, memory_dim=2, memory_depth=2, memory_conv_kernel_size=1,
                memory_chunk_size=1, working_memory_size=128,
                max_persistent_kv_tokens=0, num_persistent_tokens=1,
                memory_max_write_grad_norm=1.0,
            ))
            with torch.no_grad():
                model.model.layers[0].taal.residual_gate.fill_(0.25)
            configure_trainable_parameters(model, "taal_only")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.optimizer.lr)
            scheduler = build_scheduler(optimizer, config)
            _, loader, _, _ = tiny_training(config)
            state = TrainingState(optimizer, scheduler, loader)
            checkpoints = ModelCheckpoints(config, "real", state)
            checkpoints.save_final(model, Progress(), "signal")
            source = Path(directory) / "real/final"
            restored = load_model(config, source)
            configure_trainable_parameters(restored, "taal_only")
            for name, expected in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
            self.assertEqual(restored.model.layers[0].taal.residual_gate.item(), 0.25)

    def test_first_fact_shortcut_does_not_pass_prefix_metric(self):
        with TemporaryDirectory() as directory:
            config, tokenizer = fixture_config(Path(directory))
            config.training.save_traces = False
            probes = RetrievalValidation(config, tokenizer, Path(directory) / "run")
            model = StochasticModel()
            model.config = SimpleNamespace()
            answers = {}
            for rows in probes.rows.values():
                first_answers = {}
                for row in rows:
                    if row["metadata"]["queried_fact_order"] == 0:
                        first_answers[row["group_id"]] = row["answer"]
                for row in rows:
                    answers[row["example_id"]] = first_answers[row["group_id"]]

            class FirstFactBackend:
                capabilities = SimpleNamespace(memory_trace=False)

                def __init__(self, *args):
                    pass

                def require(self, capabilities):
                    pass

                def validate(self, prompts, save_traces):
                    pass

                def validate_saved(self, output, predictions):
                    pass

                def identity(self):
                    return {"test": "first-fact-shortcut"}

                def generate(self, example_id, prompt, output, seed, save_traces):
                    return Prediction(example_id, answers[example_id], [1], "test", len(prompt.token_ids), 1, 0.0, None)

            with patch("training.retrieval_validation.QwenBackend", FirstFactBackend):
                metrics = probes(model, Progress(step=1), lambda: False)
            self.assertEqual(metrics["val/probe_greedy_accuracy"], 0.5)
            self.assertEqual(metrics["val/probe_group_all_correct"], 0.0)

    def test_probe_configuration_rejects_partial_families_and_test_loading(self):
        config = load_config(RECIPE)
        config.data.split_files["test"] = "never-read.jsonl"
        with self.assertRaises(ValueError):
            config.validate()
        config = load_config(RECIPE)
        config.data.train.end = 3
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
