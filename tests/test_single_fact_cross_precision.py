"""The fourth precision cell reuses a completed BF16 parent without modifying it."""

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

from evaluation.storage import atomic_json
from scripts.single_fact import prepare_recipes
from scripts.single_fact_cross_precision import check_completed_parent, compare_parent, pipeline
from scripts.single_fact_preflight import check_parent_submission


def prepare_parent(parent):
    prepare_recipes(parent, ("acquisition",))
    (parent / "jobs.tsv").write_text(f"acquisition\t123\tgpu\t{parent}\n")
    atomic_json(parent / "jobs/acquisition/job.json", {"job_id": "123"})
    atomic_json(parent / "jobs/acquisition/complete.json", {"complete": True, "save_traces": True})
    atomic_json(parent / "acquisition/audit/complete.json", {"complete": True, "step": 768})


class SingleFactCrossPrecisionTests(unittest.TestCase):
    def test_parent_submission_rejects_wrong_job_path_and_duplicate_receipts(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            prepare_parent(parent)
            check_parent_submission(parent, "123")
            with self.assertRaisesRegex(ValueError, "dependency"):
                check_parent_submission(parent, "999")
            (parent / "jobs.tsv").write_text("acquisition\t123\tgpu\twrong/path\n")
            with self.assertRaisesRegex(ValueError, "dependency"):
                check_parent_submission(parent, "123")
            (parent / "jobs.tsv").write_text(f"acquisition\t123\tgpu\t{parent}\n" * 2)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                check_parent_submission(parent, "123")

    def test_completed_parent_requires_exact_acquisition_and_recipe(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            prepare_parent(parent)
            recipe = check_completed_parent(parent, "123")
            atomic_json(parent / "acquisition/audit/complete.json", {"complete": True, "step": 512})
            with self.assertRaisesRegex(ValueError, "step768"):
                check_completed_parent(parent, "123")
            atomic_json(parent / "acquisition/audit/complete.json", {"complete": True, "step": 768})
            config = yaml.safe_load(recipe.read_text())
            config["optimizer"]["lr"] = 6e-4
            recipe.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "matched BF16"):
                check_completed_parent(parent, "123")

    def test_pipeline_runs_only_fp32_audit_and_keeps_parent_read_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent, output = root / "parent", root / "cross"
            prepare_parent(parent)
            saved = {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
            with patch("scripts.single_fact_cross_precision.check_inputs"), \
                 patch("scripts.single_fact_cross_precision.audit_checkpoint", return_value=True) as audit, \
                 patch("scripts.single_fact_cross_precision.compare_parent") as compare:
                pipeline(parent, "123", output, "124", 0, torch.device("cpu"))
            audit.assert_called_once()
            args = audit.call_args.args
            self.assertEqual(args[:3], (parent / "recipes/acquisition.yaml", Path("checkpoints/parent-acquisition/final"), output / "fp32"))
            self.assertEqual(args[-2:], (False, "fp32"))
            compare.assert_called_once_with(output, parent / "acquisition/audit")
            self.assertEqual(json.loads((output / "complete.json").read_text())["questions"], {"seen": 32, "validation": 128})
            self.assertEqual(saved, {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()})
            with self.assertRaisesRegex(ValueError, "separate output"):
                pipeline(parent, "123", parent / "cross", "124", 0, torch.device("cpu"))

    def test_audit_warning_resumes_and_cancellation_does_not_requeue(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            prepare_parent(parent)
            for signum in (signal.SIGUSR1, signal.SIGTERM):
                output = root / str(signum)
                def interrupted(recipe, checkpoint, output, allocation, device, final, precision):
                    allocation.signal(signum, None)
                    return False
                with patch("scripts.single_fact_cross_precision.check_inputs"), \
                     patch("scripts.single_fact_cross_precision.audit_checkpoint", side_effect=interrupted), \
                     patch("scripts.single_fact_cross_precision.compare_parent") as compare, \
                     patch("scripts.retrieval_exposure_audit.subprocess.run") as requeue, \
                     patch.dict(os.environ, SLURM_JOB_ID="124", SLURM_RESTART_COUNT="0"):
                    if signum == signal.SIGUSR1:
                        with self.assertRaises(SystemExit):
                            pipeline(parent, "123", output, "124", 0, torch.device("cpu"))
                        self.assertEqual(requeue.call_args.args[0], ["scontrol", "requeue", "124"])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "cancelled"):
                            pipeline(parent, "123", output, "124", 0, torch.device("cpu"))
                        requeue.assert_not_called()
                    compare.assert_not_called()
                self.assertFalse((output / "complete.json").exists())
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                pipeline(parent, "123", root / "last", "124", 2, torch.device("cpu"))

    def test_cross_comparison_reuses_bf16_answers_and_checks_checkpoint_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, baseline = root / "cross", root / "parent"
            identity = {"checkpoint_sha256": "same", "progress": {"step": 768}, "source_sha256": "source"}
            atomic_json(output / "fp32/checkpoint.json", identity | {"precision": "fp32"})
            atomic_json(baseline / "checkpoint.json", identity | {"precision": "bf16"})
            row = {"example_id": "sample", "answer": "amber"}
            for cohort in ("narrow", "val"):
                (root / f"{cohort}.jsonl").write_text(json.dumps(row) + "\n")
                for destination, label in ((output / "fp32", "amber"), (baseline, "anchor")):
                    path = destination / cohort / "predictions.jsonl"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({"example_id": "sample", "hypothesis": label}) + "\n")
            saved = {str(path.relative_to(baseline)): path.read_bytes() for path in baseline.rglob("*") if path.is_file()}
            with patch("scripts.single_fact_precision.ROOT", root):
                compare_parent(output, baseline)
            result = json.loads((output / "precision-comparison.json").read_text())
            self.assertEqual(result["val"]["fp32_gains"], 1)
            self.assertIn("BF16-trained", result["val"]["scope"])
            self.assertEqual(saved, {str(path.relative_to(baseline)): path.read_bytes() for path in baseline.rglob("*") if path.is_file()})
            atomic_json(baseline / "checkpoint.json", identity | {"checkpoint_sha256": "different", "precision": "bf16"})
            with self.assertRaisesRegex(ValueError, "same checkpoint"):
                compare_parent(output, baseline)

    def test_launcher_adds_one_dependent_job_at_seven_existing_short_jobs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "output").mkdir()
            commands = root / "bin"
            commands.mkdir()
            (commands / "git").write_text("#!/bin/bash\necho titans-impl\n")
            (commands / "python3").write_text('#!/bin/bash\necho "$*" >> preflight.txt\n')
            (commands / "squeue").write_text('#!/bin/bash\nfor ((i=0;i<${TEST_SHORT_COUNT:-7};i++)); do echo old; done\n')
            (commands / "sbatch").write_text('#!/bin/bash\necho "$*" >> calls.txt\necho 901\n')
            for command in commands.iterdir():
                command.chmod(0o755)
            script = root / "queue.sh"
            script.write_text(Path("scripts/queue_single_fact_cross_precision.sh").read_text().replace(
                "/projects/llpr/amiri.ry/dev/fast-weight-memory", str(root)))
            environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}")
            command = ["bash", str(script), "output/parent", "123", "fixture"]
            submitted = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            calls = (root / "calls.txt").read_text().splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("--dependency=afterok:123", calls[0])
            self.assertIn("--kill-on-invalid-dep=yes", calls[0])
            self.assertTrue(calls[0].endswith("output/single-fact-cross-fixture output/parent 123"))
            self.assertIn("--parent-output output/parent --parent-job 123", (root / "preflight.txt").read_text())
            for batch, count in (("full", "8"), ("fixture", "7")):
                failed = subprocess.run(command[:-1] + [batch], cwd=root,
                                        env=dict(environment, TEST_SHORT_COUNT=count), capture_output=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(len((root / "calls.txt").read_text().splitlines()), 1)
            launcher = Path("training/sbatch/taal/fs_qwen_single_fact_cross_precision_short.sbatch").read_text()
            for directive in ("--partition=gpu-short", "--time=02:00:00", "--mem=32G", "--cpus-per-task=4", "--requeue"):
                self.assertIn(directive, launcher)


if __name__ == "__main__":
    unittest.main()
