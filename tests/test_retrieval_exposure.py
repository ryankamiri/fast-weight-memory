"""Matched exposure, answer-position parity, trace resume, and queue gating."""

from collections import Counter
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from data.retrieval import RetrievalDataset
from data.retrieval_exposure import balanced_blocks, stratified_rows
from evaluation.shared.retrieval_diagnostics import answer_loss, compare_logits, path_parity, score_answer, selection_counts
from evaluation.storage import atomic_json, file_sha256
from scripts.retrieval_exposure_audit import Allocation, generation_audit, load_checkpoint, token_diagnostics
from scripts.retrieval_exposure_preflight import check_gate
from tests.test_taal_qwen_causal_lm import TaalQwen3CausalLMTests
from tests.test_acquisition_training import fixture_config, tiny_training
from training.config import load_config
from training.checkpoints import ModelCheckpoints
from training.engine import Progress, train
from training.resume import TrainingState


class TinyTokenizer:
    eos_token_id = 39

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]

    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids)


class ExposureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = TaalQwen3CausalLMTests().config(num_hidden_layers=2, memory_max_write_grad_norm=1.0)
        self.model = TaalQwen3ForCausalLM(config)
        with torch.no_grad():
            for layer in self.model.model.layers:
                layer.taal.residual_gate.fill_(0.05)
        self.row = {
            "example_id": "test", "prompt": "1 2 3 4 5 6 7 8 9", "answer": "10 11",
            "prompt_input_ids": list(range(1, 10)), "input_ids": list(range(1, 12)) + [39],
            "labels": [-100] * 9 + [10, 11, 39], "metadata": {}, "group_id": "pair",
        }

    def test_label_eos_loss_and_parity_detect_wrong_predictions(self):
        logits = torch.tensor([[4., 1., 0.], [0., 3., 1.], [0., 0., 2.]])
        labels = torch.tensor([0, 1, 2])
        result = answer_loss(logits, labels)
        self.assertAlmostEqual(result.combined_nll, (result.label_nll * 2 + result.eos_nll) / 3)
        self.assertTrue(compare_logits(logits, logits, labels, result.combined_nll)["passed"])
        changed = logits.clone()
        changed[0, 1] = 10
        self.assertFalse(compare_logits(logits, changed, labels, result.combined_nll)["passed"])
        with self.assertRaises(ValueError):
            answer_loss(logits[:1], labels[:1])

    def test_pair_metrics_exclude_incomplete_consumed_groups(self):
        rows = []
        for assignment in range(2):
            for question in range(2):
                rows.append({"example_id": f"{assignment}-{question}", "answer": str((assignment + question) % 2),
                             "group_id": str(assignment),
                             "metadata": {"family_id": "f", "fact_order": [0, 1],
                                          "query_key": str(question), "queried_fact_order": question}})
        predictions = {row["example_id"]: {"hypothesis": row["answer"]} for row in rows}
        complete = selection_counts(rows, predictions)
        self.assertEqual(complete["both_questions"]["correct"], 2)
        self.assertEqual(complete["both_remapped_answers"]["correct"], 2)
        partial = selection_counts(rows[:3], {row["example_id"]: predictions[row["example_id"]] for row in rows[:3]})
        self.assertEqual(partial["both_questions"], {"correct": 1, "total": 1, "incomplete_groups": 1})

    def test_actual_training_forward_and_blocked_prefill(self):
        self.model.train()
        result = path_parity(self.model, self.row)
        self.assertTrue(result["passed"], result)
        self.assertTrue(self.model.training)
        self.assertLess(result["max_abs_logit_difference"], 1e-5)
        loss = score_answer(self.model, self.row)
        self.assertAlmostEqual(loss["combined_nll"], result["training_loss"], places=6)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_diagnostic_resume_and_native_generation_traces(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            allocation = Allocation()
            self.assertTrue(token_diagnostics(self.model, [self.row], output / "loss", "loss", allocation))
            with patch("scripts.retrieval_exposure_audit.score_answer", side_effect=AssertionError("recomputed")):
                self.assertTrue(token_diagnostics(self.model, [self.row], output / "loss", "loss", allocation))
            with patch("scripts.retrieval_exposure_audit.file_sha256", return_value="fixture"):
                for scale in (1., 0.):
                    path = output / str(scale)
                    self.assertTrue(generation_audit(self.model, TinyTokenizer(), {"checkpoint": "tiny", "tokenizer": "tiny"}, [self.row], path, allocation, scale))
                    self.assertTrue((path / "memory_traces/manifest.json").is_file())
                    self.assertTrue(generation_audit(self.model, TinyTokenizer(), {"checkpoint": "tiny", "tokenizer": "tiny"}, [self.row], path, allocation, scale))
                    self.assertEqual(len((path / "predictions.jsonl").read_text().splitlines()), 1)

    def test_saved_checkpoint_loads_and_exports_resumable_traces(self):
        recipe = Path("training/configs/taal/retrieval_exposure_repeat.yaml")
        config = load_config(recipe)
        original_directory = Path.cwd()
        with TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                recipe.parent.mkdir(parents=True)
                settings = config.to_dict()
                settings["loss"].pop("objective")
                recipe.write_text(yaml.safe_dump(settings))
                backend = Tokenizer(models.WordLevel({str(index): index for index in range(40)}))
                backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="39")
                manifest = Path(config.data.manifest_path)
                atomic_json(manifest, {
                    "tokenizer_backend": json.loads(tokenizer.backend_tokenizer.to_str()),
                    "special_tokens": tokenizer.special_tokens_map,
                })
                atomic_json(Path("data/retrieval_exposure/selection.json"), {})
                dataset = RetrievalDataset([self.row], 42, False,
                                           {"manifest_sha256": file_sha256(manifest)})
                loader = DataLoader(dataset, generator=torch.Generator().manual_seed(42))
                optimizer = torch.optim.AdamW(self.model.parameters())
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
                state = TrainingState(optimizer, scheduler, loader)
                checkpoints = ModelCheckpoints(config, "taal-exposure-repeat-123", state, tokenizer)
                checkpoints.save_final(self.model, Progress(step=512), "fixture")

                output = Path("audit")
                model, loaded_tokenizer, provenance = load_checkpoint("repeat", "123", output, torch.device("cpu"))
                self.assertEqual(provenance["tokenizer"], str(checkpoints.directory / "final"))
                for scale in (1.0, 0.0):
                    target = output / str(scale)
                    self.assertTrue(generation_audit(model, loaded_tokenizer, provenance,
                                                     [self.row], target, Allocation(), scale))
                    traces = target / "memory_traces/manifest.json"
                    self.assertTrue(traces.is_file())
                    before = file_sha256(traces)
                    self.assertTrue(generation_audit(model, loaded_tokenizer, provenance,
                                                     [self.row], target, Allocation(), scale))
                    self.assertEqual(file_sha256(traces), before)
                    self.assertEqual(len((target / "predictions.jsonl").read_text().splitlines()), 1)
            finally:
                os.chdir(original_directory)

    def test_label_balanced_blocks_and_resume_order(self):
        rows = []
        labels = [f"label{i:02}" for i in range(16)]
        for family in range(64):
            pair = labels[(family % 8) * 2:(family % 8) * 2 + 2]
            for assignment in range(2):
                for order in range(2):
                    for question in range(2):
                        rows.append({"example_id": f"{family}-{assignment}-{order}-{question}",
                                     "answer": pair[(question + assignment) % 2],
                                     "metadata": {"family_id": family, "bindings": dict(zip(("x", "y"), pair)),
                                                  "queried_fact_order": (question + order) % 2}})
        blocks = balanced_blocks(rows, labels)
        self.assertEqual(len(blocks), 8)
        broad = sum(blocks, [])
        self.assertEqual(len({row["example_id"] for row in broad}), 512)
        self.assertEqual(blocks[0], broad[:64])
        for block in blocks:
            self.assertEqual(Counter(row["answer"] for row in block), Counter({label: 4 for label in labels}))
        self.assertEqual(len(stratified_rows(broad)), 64)
        data = RetrievalDataset(blocks[0], 42, False, {})
        stream = iter(data)
        first = [next(stream) for _ in range(21)]
        resumed = RetrievalDataset(blocks[0], 42, False, {})
        resumed.load_state_dict(data.state_dict())
        self.assertEqual(first + list(resumed), blocks[0])
        repeated = []
        for epoch in range(8):
            data.set_epoch(epoch)
            repeated.extend(data)
        self.assertEqual(Counter(row["example_id"] for row in repeated), Counter({row["example_id"]: 8 for row in blocks[0]}))

    def test_recipes_differ_only_in_cohort_and_name(self):
        small = load_config("training/configs/taal/retrieval_exposure_repeat.yaml")
        broad = load_config("training/configs/taal/retrieval_exposure_broad.yaml")
        expected = copy.deepcopy(small)
        expected.data = broad.data
        expected.wandb.name = broad.wandb.name
        self.assertEqual(expected, broad)
        self.assertEqual(small.training.max_steps, 512)
        self.assertEqual(small.checkpoints.save_steps, [64, 128, 256, 384, 512])
        self.assertTrue(small.training.save_traces)
        self.assertTrue(small.data.preserve_order)
        self.assertEqual(small.validation.probe_families, 2)

    def test_checkpoint_between_regular_probes_gets_validation(self):
        with TemporaryDirectory() as directory:
            config, _ = fixture_config(Path(directory))
            config.training.max_steps = 3
            config.training.eval_every_steps = 2
            config.checkpoints.save_steps = [1, 3]
            config.validate()
            model, loader, optimizer, scheduler = tiny_training(config)
            evaluated, saved = [], []

            def evaluate(model, progress, should_stop):
                evaluated.append(progress.step)
                return {"val/loss": 1.0}

            train(model, loader, [], optimizer, scheduler, config, torch.device("cpu"),
                  lambda metrics: None, lambda: False,
                  on_validation=lambda model, progress, metrics: saved.append(progress.step),
                  evaluate=evaluate)
            self.assertEqual(evaluated, [1, 2, 3])
            self.assertEqual(saved, evaluated)

    def test_resume_finishes_interrupted_milestone_before_next_update(self):
        with TemporaryDirectory() as directory:
            config, _ = fixture_config(Path(directory))
            config.training.max_steps = 3
            config.training.eval_every_steps = 2
            config.checkpoints.save_steps = [1, 3]
            model, loader, optimizer, scheduler = tiny_training(config)
            progress = Progress()
            stopped = False
            evaluated, saved = [], []

            def evaluate(model, progress, should_stop):
                nonlocal stopped
                evaluated.append((progress.step, len(model.calls)))
                if len(evaluated) == 1:
                    stopped = True
                    return None
                return {"val/loss": 1.0}

            for _ in range(2):
                train(model, loader, [], optimizer, scheduler, config, torch.device("cpu"),
                      lambda metrics: None, lambda: stopped, progress=progress,
                      on_validation=lambda model, progress, metrics: saved.append(progress.step),
                      evaluate=evaluate)
                stopped = False
            self.assertEqual(evaluated, [(1, 1), (1, 1), (2, 2), (3, 3)])
            self.assertEqual(saved, [1, 2, 3])
            self.assertEqual(progress.step, 3)
            self.assertEqual(progress.examples_seen, 3)

    def test_allocation_and_failed_gate_boundaries(self):
        allocation = Allocation()
        with self.assertRaises(RuntimeError):
            allocation.finish(False)
        allocation.signal(signal.SIGUSR1, None)
        with patch.dict(os.environ, SLURM_JOB_ID="123", SLURM_RESTART_COUNT="0"), patch("scripts.retrieval_exposure_audit.subprocess.run") as requeue:
            with self.assertRaises(SystemExit):
                allocation.finish(False)
            requeue.assert_called_once()
        allocation.signal(signal.SIGTERM, None)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            allocation.finish(True)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "complete.json", {"phase": "parity", "arms": ["A", "B", "C"], "passed": True})
            for arm in "ABC":
                atomic_json(root / arm / "parity/summary.json", {"examples": 16, "passed": True})
            check_gate(root)
            atomic_json(root / "C/parity/summary.json", {"examples": 16, "passed": False})
            with self.assertRaises(ValueError):
                check_gate(root)

    def test_eight_job_queue_and_dependencies(self):
        script = Path("scripts/queue_retrieval_exposure.sh").resolve()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            (root / "git").write_text("#!/bin/bash\necho titans-impl\n")
            (root / "squeue").write_text("#!/bin/bash\nexit 0\n")
            (root / "python3").write_text("#!/bin/bash\nexit 0\n")
            (root / "sbatch").write_text('#!/bin/bash\necho "$*" >> "$RECEIPTS"\nwc -l < "$RECEIPTS" | tr -d " "\n')
            for name in ("git", "squeue", "python3", "sbatch"):
                (root / name).chmod(0o755)
            receipts = root / "receipts"
            environment = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"], RECEIPTS=str(receipts))
            command = ["bash", "-c", 'PWD=/projects/llpr/amiri.ry/dev/fast-weight-memory; source "$1" fixture', "test", str(script)]
            subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True)
            lines = receipts.read_text().splitlines()
            self.assertEqual(len(lines), 8)
            self.assertIn("--dependency=afterok:1", lines[4])
            self.assertIn("--dependency=afterok:5", lines[5])
            self.assertIn("--dependency=afterok:1", lines[6])
            self.assertIn("--dependency=afterok:7", lines[7])
            second = subprocess.run(command, cwd=root, env=environment, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(len(receipts.read_text().splitlines()), 8)
        for path in (script, Path("training/sbatch/taal/fs_qwen_retrieval_exposure_short.sbatch"), Path("evaluation/sbatch/taal/fs_qwen_retrieval_exposure_short.sbatch")):
            subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
