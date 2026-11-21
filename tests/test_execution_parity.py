"""Actual tiny-model execution, instrumentation, precision restoration and launch contracts."""

import copy
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal.execution_parity import (
    PRECISIONS, compare_execution, execute_path, execution_precision,
)
from scripts.retrieval_execution_parity import (
    EXAMPLE_IDS, completed_case, compare_kv, diagnostic_rows, run_case, run_diagnostic,
)
from scripts.retrieval_exposure_audit import Allocation
from tests.test_retrieval_exposure import TinyTokenizer
from tests.test_taal_qwen_causal_lm import TaalQwen3CausalLMTests


class ExecutionParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = TaalQwen3CausalLMTests().config(
            num_hidden_layers=2, memory_max_write_grad_norm=1.0,
            memory_conv_kernel_size=4, max_persistent_kv_tokens=0,
        )
        self.model = TaalQwen3ForCausalLM(config)
        with torch.no_grad():
            for layer in self.model.model.layers:
                layer.taal.residual_gate.fill_(0.1)
        self.row = {
            "example_id": "fixture", "prompt_input_ids": list(range(1, 10)),
            "input_ids": list(range(1, 12)) + [39],
            "labels": [-100] * 9 + [10, 11, 39],
            "prompt": "1 2 3 4 5 6 7 8 9", "answer": "10 11", "metadata": {}, "group_id": "pair",
        }

    def test_all_precisions_align_paths_and_keep_weights_unchanged(self):
        before = copy.deepcopy(self.model.state_dict())
        for precision in PRECISIONS:
            with self.subTest(precision=precision):
                reference = execute_path(self.model, self.row, "full_eval", precision)
                labels = torch.tensor([10, 11, 39])
                self.assertEqual(len(reference.features), 42)
                self.assertTrue(all(value.shape[0] == 12 for value in reference.features.values()))
                full_cache = None
                for path in ("full_train", "full_cache", "blocked", "blocked_alternate"):
                    actual = execute_path(self.model, self.row, path, precision)
                    comparison = compare_execution(reference, actual, labels)
                    self.assertTrue(comparison["metrics"]["passed"], comparison["metrics"])
                    self.assertEqual(actual.tokens_seen, 12)
                    if path == "full_cache":
                        full_cache = actual
                    if path.startswith("blocked"):
                        self.assertEqual(len(compare_kv(full_cache.kv, actual.kv)), 2)
                unobserved = execute_path(self.model, self.row, "blocked", precision, record=False)
                torch.testing.assert_close(actual.logits, unobserved.logits, atol=1e-5, rtol=1e-4)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(self.model.training)

    def test_precision_and_hooks_restore_after_failure(self):
        device = torch.device("cpu")
        prior = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled())
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with execution_precision(device, "fp32_math") as settings:
                self.assertFalse(settings["autocast"])
                self.assertFalse(settings["matmul_tf32"])
                self.assertFalse(settings["sdpa_enabled"]["flash"])
                raise RuntimeError("fixture")
        restored = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled())
        self.assertEqual(prior, restored)

        broken = copy.deepcopy(self.row)
        broken["input_ids"][0] = 999
        broken["prompt_input_ids"][0] = 999
        with self.assertRaises(IndexError):
            execute_path(self.model, broken, "full_eval", "fp32_math")
        self.assertTrue(self.model.training)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.model.modules()))

    def test_kv128_memory128_and_persistent8_cross_multiple_blocks(self):
        config = TaalQwen3CausalLMTests().config(
            num_hidden_layers=1, working_memory_size=128, max_persistent_kv_tokens=0,
            memory_dim=128, memory_depth=2, memory_conv_kernel_size=4,
            memory_max_write_grad_norm=1.0, num_persistent_tokens=8,
        )
        model = TaalQwen3ForCausalLM(config)
        ids = torch.randint(0, 39, (279,)).tolist() + [39]
        row = dict(self.row, input_ids=ids, prompt_input_ids=ids[:-2],
                   labels=[-100] * 278 + ids[-2:])
        reference = execute_path(model, row, "full_cache", "fp32_math")
        for path in ("full_train", "blocked", "blocked_alternate"):
            actual = execute_path(model, row, path, "fp32_math")
            comparison = compare_execution(reference, actual, torch.tensor(ids[-2:]))
            self.assertTrue(comparison["metrics"]["passed"], comparison["metrics"])
            if path.startswith("blocked"):
                kv = compare_kv(reference.kv, actual.kv)
                # Retain W-1 past tokens; the next incoming token occupies the final slot.
                self.assertEqual(kv[0]["positions"], list(range(153, 280)))

    def test_first_difference_and_final_state_checks_are_not_silent(self):
        result = execute_path(self.model, self.row, "full_eval", "fp32_math")
        changed = copy.deepcopy(result)
        key = sorted(changed.features)[3]
        changed.features[key][5, 0] += 1.0
        comparison = compare_execution(result, changed, torch.tensor([10, 11, 39]))
        first = comparison["metrics"]["first_tolerance_difference"]
        self.assertEqual((first["stage"], first["position"]), (key, 5))
        self.assertEqual(comparison["snapshots"]["first_tolerance"]["actual"].shape,
                         changed.features[key][5].shape)
        changed.tokens_seen -= 1
        with self.assertRaisesRegex(ValueError, "token counts"):
            compare_execution(result, changed, torch.tensor([10, 11, 39]))

    def test_case_writes_inspectable_artifacts_and_detects_corruption(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_case(self.model, TinyTokenizer(), self.row, "fp32_math", 0.0, root)
            self.assertTrue(all(check["passed"] for check in summary["observer_checks"].values()))
            self.assertTrue(summary["save_traces"])
            self.assertTrue(completed_case(root))
            logits = torch.load(root / "full_eval-logits.pt", weights_only=True)
            self.assertEqual(logits["labels"].tolist(), [10, 11, 39])
            self.assertEqual(logits["logits"].shape, (3, 40))
            self.assertEqual(len(summary["comparisons"]), 4)
            (root / "blocked-stages.json.gz").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "changed diagnostic"):
                completed_case(root)

    @patch("scripts.retrieval_exposure_audit.file_sha256", return_value="fixture")
    def test_runner_matrix_and_greedy_trace_export_resume(self, file_hash):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {"checkpoint": "tiny", "tokenizer": "tiny"}
            allocation = Allocation()
            self.assertTrue(run_diagnostic(self.model, TinyTokenizer(), provenance,
                                           [self.row], root, allocation))
            summaries = list((root / "fixture").glob("*/reads-*/summary.json"))
            self.assertEqual(len(summaries), 6)
            self.assertTrue((root / "generation/memory_traces/manifest.json").is_file())
            self.assertTrue(run_diagnostic(self.model, TinyTokenizer(), provenance,
                                           [self.row], root, allocation))
            self.assertEqual(len((root / "generation/predictions.jsonl").read_text().splitlines()), 1)

    def test_fixed_examples_match_across_arms_and_missing_example_fails(self):
        rows = [dict(self.row, example_id=example_id) for example_id in EXAMPLE_IDS]
        with patch("scripts.retrieval_execution_parity.checked_rows", return_value=list(reversed(rows))):
            for arm in "ABC":
                self.assertEqual(diagnostic_rows(arm), rows)
        with patch("scripts.retrieval_execution_parity.checked_rows", return_value=rows[:1]):
            with self.assertRaises(ValueError):
                diagnostic_rows("A")

    def test_three_independent_jobs_no_training_and_duplicate_protection(self):
        script = Path("scripts/queue_execution_parity.sh").resolve()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            commands = {
                "git": "echo titans-impl", "squeue": "exit 0", "python3": "exit 0",
                "sbatch": 'echo "$*" >> "$RECEIPTS"; wc -l < "$RECEIPTS" | tr -d " "',
            }
            for name, command in commands.items():
                path = root / name
                path.write_text("#!/bin/bash\n" + command + "\n")
                path.chmod(0o755)
            receipts = root / "receipts"
            environment = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"], RECEIPTS=str(receipts))
            command = ["bash", "-c", 'PWD=/projects/llpr/amiri.ry/dev/fast-weight-memory; source "$1" fixture', "test", str(script)]
            subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True)
            lines = receipts.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertFalse(any("--dependency" in line or "training/" in line for line in lines))
            for arm, line in zip("ABC", lines):
                self.assertIn(f"--job-name=taal-path-{arm}", line)
                self.assertIn(f"fixture/{arm}", line)
            second = subprocess.run(command, cwd=root, env=environment, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(len(receipts.read_text().splitlines()), 3)
        for path in (script, Path("evaluation/sbatch/taal/fs_qwen_execution_parity_short.sbatch")):
            subprocess.run(["bash", "-n", str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
