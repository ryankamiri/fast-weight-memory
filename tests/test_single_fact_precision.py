"""Matched recipes, real precision-aware traces and safe dependent submissions."""

import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.storage import atomic_json
from scripts.retrieval_exposure_audit import Allocation
from scripts.single_fact import audit_checkpoint, generation
from scripts.single_fact_precision import pipeline, compare_precisions
from scripts.single_fact_preflight import check_precision_recipe
from scripts.single_fact_scoring import assignment_summary
from tests.test_retrieval_exposure import TinyTokenizer
from tests.test_taal_qwen_causal_lm import TaalQwen3CausalLMTests
from utils.precision import precision_context


class SingleFactPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.row = {
            "example_id": "fixture", "prompt": "1 2 3 4 5 6 7 8 9", "answer": "amber",
            "prompt_input_ids": list(range(1, 10)), "input_ids": list(range(1, 12)) + [39],
            "labels": [-100] * 9 + [10, 11, 39],
            "metadata": {"regime": "familiar", "query_key": "record"},
        }

    def test_matched_recipe_rejects_other_training_changes(self):
        self.assertEqual(check_precision_recipe()["updates"], 768)
        original = yaml.safe_load(Path("training/configs/taal/single_fact_fp32.yaml").read_text())
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "recipe.yaml"
            for section, name, value in (("optimizer", "lr", 1e-4), ("training", "seed", 99),
                                         ("model", "memory_chunk_size", 8),
                                         ("training", "save_traces", False)):
                changed = json.loads(json.dumps(original))
                changed[section][name] = value
                candidate.write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError, "differ only"):
                    check_precision_recipe(candidate=candidate)

    def test_precision_does_not_change_model_initialization(self):
        models = []
        for precision in ("bf16", "fp32"):
            torch.manual_seed(42)
            config = TaalQwen3CausalLMTests().config(execution_precision=precision)
            models.append(TaalQwen3ForCausalLM(config))
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)

    def test_counterfactual_pairs_do_not_reward_constant_labels(self):
        rows = []
        for key in ("a", "b"):
            for label in ("amber", "anchor"):
                rows.append({"example_id": f"{key}-{label}", "answer": label,
                             "metadata": {"regime": "familiar", "query_key": key}})
        constant = {row["example_id"]: {"hypothesis": "amber"} for row in rows}
        result = assignment_summary(rows, constant)
        self.assertEqual((result["pairs"], result["both_correct"], result["same_valid_label"]), (2, 0, 2))
        correct = {row["example_id"]: {"hypothesis": row["answer"]} for row in rows}
        self.assertEqual(assignment_summary(rows, correct)["both_correct"], 2)
        singletons = rows[::2]
        result = assignment_summary(singletons, {row["example_id"]: correct[row["example_id"]] for row in singletons})
        self.assertEqual((result["pairs"], result["incomplete_groups"]), (0, 2))
        with self.assertRaisesRegex(ValueError, "every planned"):
            assignment_summary(rows, {})

    def test_generation_disables_outer_autocast_and_refuses_mixed_precision_resume(self):
        config = TaalQwen3CausalLMTests().config(execution_precision="fp32", memory_max_write_grad_norm=1.0)
        model = TaalQwen3ForCausalLM(config)
        provenance = {"checkpoint": "tiny", "tokenizer": "tiny", "precision": "fp32"}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.single_fact.file_sha256", return_value="fixture"), \
                 torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertTrue(generation(model, TinyTokenizer(), provenance,
                                           [self.row], output, Allocation()))
                self.assertTrue(torch.is_autocast_enabled("cpu"))
            execution = json.loads((output / "execution.json").read_text())
            self.assertFalse(execution["autocast_enabled"])
            self.assertFalse(execution["matmul_tf32"])
            self.assertEqual(execution["float32_matmul_precision"], "highest")
            self.assertEqual(len(list((output / "memory_traces/episodes").glob("*.json.gz"))), 1)
            model.config.execution_precision = "bf16"
            with self.assertRaisesRegex(ValueError, "provenance"):
                generation(model, TinyTokenizer(), provenance, [self.row], output, Allocation())
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                generation(model, TinyTokenizer(), provenance | {"precision": "bf16"}, [self.row], output, Allocation())

    def test_actual_checkpoint_audit_in_both_policies_saves_losses_traces_and_resumes(self):
        config = TaalQwen3CausalLMTests().config(execution_precision="fp32", memory_max_write_grad_norm=1.0)
        model = TaalQwen3ForCausalLM(config)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint)
            for cohort in ("narrow", "val"):
                (root / f"{cohort}.jsonl").write_text(json.dumps(self.row) + "\n")
            atomic_json(root / "manifest.json", {"fixture": True})
            with patch("scripts.single_fact.ROOT", root), \
                 patch("scripts.single_fact.validate_checkpoint", return_value={"progress": {"step": 768}}), \
                 patch("scripts.single_fact.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                 patch("scripts.single_fact.tokenizer_fingerprint", return_value="fixture"), \
                 patch("scripts.single_fact.precision_context", wraps=precision_context) as policy:
                for precision in ("fp32", "bf16"):
                    output = root / precision
                    self.assertTrue(audit_checkpoint("recipe", checkpoint, output, Allocation(),
                                                     torch.device("cpu"), False, precision))
                    self.assertEqual(policy.call_args.args[1], precision)
                    for cohort in ("narrow", "val"):
                        loss = json.loads((output / f"{cohort}-loss/summary.json").read_text())
                        self.assertEqual(loss["examples"], 1)
                        self.assertGreater(loss["label_nll"], 0)
                        self.assertEqual(len(list((output / cohort / "memory_traces/episodes").glob("*.json.gz"))), 1)
                    with patch("evaluation.shared.backends.capture_generation", side_effect=AssertionError("Repeated inference")), \
                         patch("scripts.retrieval_exposure_audit.score_answer", side_effect=AssertionError("Repeated scoring")):
                        self.assertTrue(audit_checkpoint("recipe", checkpoint, output, Allocation(),
                                                         torch.device("cpu"), False, precision))
                with self.assertRaisesRegex(ValueError, "Run settings changed"):
                    audit_checkpoint("recipe", checkpoint, root / "fp32", Allocation(), torch.device("cpu"), False, "bf16")
            self.assertEqual(json.loads((checkpoint / "config.json").read_text())["execution_precision"], "fp32")

    def test_incomplete_training_cannot_publish_a_parent_receipt(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.single_fact_precision.check_inputs"), \
                 patch("scripts.single_fact_precision.file_sha256", return_value="fixture"), \
                 patch("scripts.single_fact_precision.needs_training", return_value=True), \
                 patch("scripts.single_fact_precision.run_segment", side_effect=SystemExit(75)), \
                 patch("scripts.single_fact_precision.audit_checkpoint") as audit:
                with self.assertRaises(SystemExit):
                    pipeline(output, "train", "123", 0, torch.device("cpu"))
                audit.assert_not_called()
            self.assertFalse((output / "jobs/train/complete.json").exists())

    def test_dependent_audit_requires_training_and_records_both_policies(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.single_fact_precision.check_inputs"), \
                 patch("scripts.single_fact_precision.file_sha256", return_value="fixture"), \
                 patch("scripts.single_fact_precision.needs_training", return_value=True), \
                 patch("scripts.single_fact_precision.run_segment") as train, \
                 patch("scripts.single_fact_precision.validate_checkpoint", return_value={"progress": {"step": 768}}), \
                 patch("scripts.single_fact_precision.audit_checkpoint", return_value=True) as audit, \
                 patch("scripts.single_fact_precision.compare_precisions") as compare:
                with self.assertRaisesRegex(ValueError, "receipt is missing"):
                    pipeline(output, "audit", "124", 0, torch.device("cpu"))
                audit.assert_not_called()
                pipeline(output, "train", "123", 0, torch.device("cpu"))
                self.assertEqual(train.call_args.args[-1], 6)
                audit.assert_not_called()
                pipeline(output, "audit", "124", 0, torch.device("cpu"))
                self.assertEqual([call.args[-1] for call in audit.call_args_list], ["fp32", "bf16"])
                compare.assert_called_once_with(output / "audit")
                self.assertTrue((output / "jobs/audit/complete.json").is_file())

    def test_audit_warning_requeues_without_a_completion_receipt(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = f"checkpoints/{output.name}-acquisition/final"
            atomic_json(output / "jobs/train/complete.json", {"complete": True, "step": 768, "checkpoint": checkpoint})
            def interrupted_audit(recipe, checkpoint, output, allocation, device, final, precision):
                allocation.signal(signal.SIGUSR1, None)
                return False
            with patch("scripts.single_fact_precision.check_inputs"), \
                 patch("scripts.single_fact_precision.file_sha256", return_value="fixture"), \
                 patch("scripts.single_fact_precision.audit_checkpoint", side_effect=interrupted_audit), \
                 patch("scripts.retrieval_exposure_audit.subprocess.run") as requeue, \
                 patch.dict(os.environ, SLURM_JOB_ID="124", SLURM_RESTART_COUNT="0"):
                with self.assertRaises(SystemExit):
                    pipeline(output, "audit", "124", 0, torch.device("cpu"))
                self.assertEqual(requeue.call_args.args[0], ["scontrol", "requeue", "124"])
            self.assertFalse((output / "jobs/audit/complete.json").exists())

    def test_precision_comparison_counts_paired_gains_and_losses(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self.row, self.row | {"example_id": "other", "answer": "anchor"}]
            for cohort in ("narrow", "val"):
                (root / f"{cohort}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                for precision, answers in (("fp32", ("amber", "amber")), ("bf16", ("anchor", "anchor"))):
                    path = root / precision / cohort / "predictions.jsonl"
                    path.parent.mkdir(parents=True)
                    path.write_text("".join(json.dumps({"example_id": row["example_id"], "hypothesis": answer}) + "\n"
                                            for row, answer in zip(rows, answers)))
            with patch("scripts.single_fact_precision.ROOT", root):
                compare_precisions(root)
            report = json.loads((root / "precision-comparison.json").read_text())
            self.assertEqual((report["val"]["fp32_gains"], report["val"]["fp32_losses"], report["val"]["changed_label"]), (1, 1, 2))

    def test_launcher_submits_two_jobs_and_preserves_existing_queue(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "output").mkdir()
            commands = root / "bin"
            commands.mkdir()
            (commands / "git").write_text("#!/bin/bash\necho titans-impl\n")
            (commands / "python3").write_text("#!/bin/bash\nexit 0\n")
            (commands / "squeue").write_text('#!/bin/bash\nfor ((i=0;i<${TEST_SHORT_COUNT:-5};i++)); do echo old; done\n')
            (commands / "sbatch").write_text('#!/bin/bash\necho "$*" >> calls.txt\necho $((900 + $(wc -l < calls.txt)))\n')
            for command in commands.iterdir():
                command.chmod(0o755)
            script = root / "queue.sh"
            script.write_text(Path("scripts/queue_single_fact_precision.sh").read_text().replace(
                "/projects/llpr/amiri.ry/dev/fast-weight-memory", str(root)))
            environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}")
            submitted = subprocess.run(["bash", str(script), "fixture"], cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            calls = (root / "calls.txt").read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--dependency", calls[0])
            self.assertIn("--dependency=afterok:901", calls[1])
            self.assertIn("--kill-on-invalid-dep=yes", calls[1])
            self.assertEqual(len((root / "output/single-fact-fp32-fixture/jobs.tsv").read_text().splitlines()), 2)
            for name, count in (("full", "7"), ("fixture", "5")):
                failed = subprocess.run(["bash", str(script), name], cwd=root,
                                        env=dict(environment, TEST_SHORT_COUNT=count), capture_output=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(len((root / "calls.txt").read_text().splitlines()), 2)
            launcher = Path("training/sbatch/taal/fs_qwen_single_fact_precision_short.sbatch").read_text()
            for directive in ("--partition=gpu-short", "--time=02:00:00", "--mem=32G", "--cpus-per-task=4", "--requeue"):
                self.assertIn(directive, launcher)


if __name__ == "__main__":
    unittest.main()
