"""Single-fact split safety, complete-label scoring and resumable stage boundaries."""

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers
import torch
from transformers import PreTrainedTokenizerFast

from data.balanced_retrieval import LABELS
from data.retrieval import load_retrieval_splits
from data.single_fact import SingleFactConfig, build, prepare
from evaluation.storage import atomic_json
from scripts.retrieval_exposure_audit import Allocation
from scripts.single_fact import generation, pipeline, prepare_recipes
from scripts.single_fact_scoring import SingleFactBenchmark, paired_comparison, predicted_label, wilson
from tests.test_taal_qwen_causal_lm import TaalQwen3CausalLMTests
from tests.test_retrieval_exposure import TinyTokenizer
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from training.config import load_config
from training.train import main as train_main


def fixture_tokenizer():
    vocabulary = {"<eos>": 0}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocabulary[character] = len(vocabulary)
    merges = []
    for label in LABELS:
        prefix = "Ġ"
        for character in label:
            merged = prefix + character
            if merged not in vocabulary:
                vocabulary[merged] = len(vocabulary)
                merges.append((prefix, character))
            prefix = merged
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")


class SingleFactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = fixture_tokenizer()
        cls.config = SingleFactConfig(val_per_regime=16, test_per_regime=16, max_sequence_tokens=4096)
        cls.bundle = build(cls.tokenizer, cls.config)

    def test_record_disjointness_balance_and_exact_token_eviction(self):
        self.assertEqual({k: len(v) for k, v in self.bundle.items()},
                         {"narrow": 32, "mixed": 128, "val": 64, "test": 64, "controls": 64})
        seen = set()
        for split in ("narrow", "val", "test"):
            rows = self.bundle[split]
            keys = {key for row in rows for key in row["metadata"]["bindings"]}
            self.assertFalse(keys & seen)
            seen.update(keys)
            for row in rows:
                self.assertEqual(self.tokenizer.encode(row["prompt"], add_special_tokens=False), row["prompt_input_ids"])
                metadata = row["metadata"]
                self.assertEqual(row["answer"], metadata["bindings"][metadata["query_key"]])
                self.assertGreaterEqual(metadata["query_start_token"] - metadata["fact_token_spans"][0][1], 192)
                self.assertEqual(row["labels"][:len(row["prompt_input_ids"])], [-100] * len(row["prompt_input_ids"]))
        for regime in ("familiar", "filler", "position", "combined"):
            rows = [r for r in self.bundle["test"] if r["metadata"]["regime"] == regime]
            self.assertEqual(Counter(r["answer"] for r in rows), Counter({label: 1 for label in LABELS}))
        self.assertEqual(self.bundle["narrow"], self.bundle["mixed"][:32])
        for offset in (0, 32, 64, 96):
            self.assertEqual([r["answer"] for r in self.bundle["mixed"][offset:offset+32]],
                             [r["answer"] for r in self.bundle["narrow"]])

    def test_control_donors_change_only_the_label(self):
        controls = self.bundle["controls"]
        for left, right in zip(controls[::2], controls[1::2]):
            self.assertEqual(left["pair_group"], right["pair_group"])
            self.assertNotEqual(left["target_token_id"], right["target_token_id"])
            self.assertEqual(len(left["input_ids"]), len(right["input_ids"]))
            changes = [i for i, (a, b) in enumerate(zip(left["input_ids"], right["input_ids"])) if a != b]
            self.assertEqual(len(changes), 1)
            self.assertLess(changes[0], left["final_query_position"] - 128)

    def test_manifest_training_cannot_open_test_or_replace_existing_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(self.tokenizer, self.config, root)
            config = load_config("training/configs/taal/single_fact.yaml")
            config.data.seq_len = 4096
            config.data.manifest_path = str(root / "narrow/manifest.json")
            config.data.split_files = {split: str(root / "narrow" / f"{split}.jsonl") for split in ("train", "val")}
            (root / "test.jsonl").unlink()
            splits, _ = load_retrieval_splits(config.data)
            self.assertEqual(len(splits["train"]), 32)
            with self.assertRaisesRegex(ValueError, "replace"):
                prepare(self.tokenizer, replace(self.config, seed=99), root)

    def test_common_label_contract_handles_punctuation_multitoken_labels_and_conflicts(self):
        for text, label in ((" amber.", "amber"), ('"anchor".', "anchor"), ("birch\n", "birch"),
                            ("amber amber", "amber"), ("The label is amber", None),
                            ("amber or anchor", None), ("ambergris", None), ("", None)):
            self.assertEqual(predicted_label(text), label)
        row = self.bundle["test"][0]
        benchmark = SingleFactBenchmark()
        example = benchmark.prepare(row, self.tokenizer)
        scores = benchmark.score(example, row["answer"] + ".")
        self.assertEqual(scores["label_correct"], 1)
        self.assertEqual(scores["exact_match"], 0)
        self.assertLess(wilson(16, 16)[0], 0.85)

    def test_paired_score_requires_identical_ids_and_preserves_gains_losses(self):
        rows = self.bundle["test"][:4]
        correct = {r["example_id"]: {"hypothesis": r["answer"]} for r in rows}
        wrong = {r["example_id"]: {"hypothesis": ""} for r in rows}
        report = paired_comparison(rows, correct, wrong)
        self.assertEqual((report["gains"], report["losses"], report["accuracy_difference"]), (4, 0, 1))
        wrong.pop(rows[0]["example_id"])
        with self.assertRaises(ValueError):
            paired_comparison(rows, correct, wrong)

    def test_continuations_share_parent_model_optimizer_and_update_budget(self):
        with TemporaryDirectory() as directory:
            recipes = prepare_recipes(Path(directory))
            acquisition, narrow, mixed = [load_config(recipes[name]) for name in ("acquisition", "narrow", "mixed")]
            self.assertIsNone(acquisition.training.initialize_from_checkpoint)
            self.assertEqual(acquisition.training.max_steps, 768)
            self.assertEqual(narrow.training.initialize_from_checkpoint, mixed.training.initialize_from_checkpoint)
            self.assertIsNone(narrow.training.initial_taal_gate)
            self.assertEqual(narrow.training.max_steps, 512)
            self.assertEqual(mixed.training.max_steps, 512)
            self.assertEqual(narrow.model, mixed.model)
            self.assertEqual(narrow.optimizer, mixed.optimizer)
            self.assertEqual(narrow.scheduler, mixed.scheduler)
            self.assertEqual(narrow.optimizer.lr, 3e-4)
            self.assertTrue(narrow.training.save_traces)
            self.assertTrue(narrow.checkpoints.save_training_state)

    def test_resume_of_warm_start_passes_pre_cuda_guard(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = prepare_recipes(root)["narrow"]
            (root / "trainer_state.pt").touch()
            with patch("sys.argv", ["train", "--config", str(recipe), "--resume", str(root)]), \
                 patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                    train_main()

    def test_warning_blocks_dependent_evaluation_and_cancellation_never_requeues(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("u_full", "u_sw128"):
                atomic_json(root / name / "complete.json", {"complete": True})
            with patch("scripts.single_fact.check_inputs"), \
                 patch("scripts.single_fact.needs_training", return_value=True), \
                 patch("scripts.single_fact.run_segment", return_value=True), \
                 patch("scripts.single_fact.audit_checkpoint") as audit, \
                 patch("scripts.retrieval_exposure_audit.subprocess.run") as requeue, \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "2"}):
                with self.assertRaises(SystemExit):
                    pipeline(root, "123", 2, torch.device("cpu"))
                audit.assert_not_called()
                self.assertEqual(requeue.call_args.args[0], ["scontrol", "requeue", "123"])
            allocation = Allocation(16)
            allocation.signal(signal.SIGTERM, None)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                allocation.finish(False)

    def test_shared_generation_records_real_taal_traces_and_resumes(self):
        model = TaalQwen3ForCausalLM(TaalQwen3CausalLMTests().config(memory_max_write_grad_norm=1.0))
        row = {"example_id": "fixture", "prompt": "1 2 3 4 5 6 7 8 9", "answer": "amber",
               "metadata": {"regime": "familiar"}}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.single_fact.file_sha256", return_value="fixture"):
                self.assertTrue(generation(model, TinyTokenizer(), {"checkpoint": "tiny", "tokenizer": "tiny"},
                                           [row], output, Allocation()))
                with patch("evaluation.shared.backends.capture_generation", side_effect=AssertionError("Repeated completed inference")):
                    self.assertTrue(generation(model, TinyTokenizer(), {"checkpoint": "tiny", "tokenizer": "tiny"},
                                               [row], output, Allocation()))
            self.assertEqual(len(list((output / "memory_traces/episodes").glob("*.json.gz"))), 1)


if __name__ == "__main__":
    unittest.main()
