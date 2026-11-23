"""Execute the proposed controls, objective and resumable pipeline on tiny Qwen."""

import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen3ForCausalLM

from architectures.shared.qwen.causal_lm import causal_lm_loss
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.titans.neural_memory import NeuralMemory
from architectures.ttcd.qwen.causal_lm import TTCDQwen3ForCausalLM
from evaluation.shared.backends import QwenBackend
from evaluation.storage import atomic_json
from scripts.retrieval_backward_parity import backward_case, run_diagnostics, selected_families
from scripts.retrieval_exposure_audit import Allocation, generation_audit
from scripts.retrieval_parallel import run_pipeline
from scripts.retrieval_parallel_audit import ARMS, baseline_config, recipe_path, validation_rows
from tests.test_retrieval_exposure import TinyTokenizer
from tests.test_taal_qwen_causal_lm import TaalQwen3CausalLMTests
from training.config import load_config
from training.train import configure_trainable_parameters, verify_loading
from utils.precision import precision_context


class ParallelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.row = {"example_id": "fixture", "prompt_input_ids": list(range(1, 10)),
                    "input_ids": list(range(1, 12)) + [39], "labels": [-100] * 9 + [10, 11, 39],
                    "prompt": "1 2 3 4 5 6 7 8 9", "answer": "10 11", "metadata": {}, "group_id": "pair"}

    def test_eos_weight_changes_only_eos_coefficient_in_loss_and_gradient(self):
        head = torch.nn.Linear(5, 13, bias=False)
        values = torch.randn(10, 5, requires_grad=True)
        labels = torch.tensor([[-100, -100, 4, 5, 12, 3], [-100, 8, 12, -100, 6, 12]])
        targets = labels[:, 1:].reshape(-1)
        for weight in (1.0, 0.1, 0.0):
            actual = causal_lm_loss(head, values, labels, 12, weight)
            token_loss = F.cross_entropy(head(values), targets, reduction="none", ignore_index=-100)
            expected = (token_loss * torch.where(targets == 12, weight, 1)).sum() / (targets != -100).sum()
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual, (values, head.weight), retain_graph=True)
            expected_grad = torch.autograd.grad(expected, (values, head.weight), retain_graph=True)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b)

    def test_static_adapter_has_no_memory_but_has_trainable_gradients_and_reload(self):
        config = TaalQwen3CausalLMTests().config(adapter_type="static", num_persistent_tokens=0,
                    static_adapter_width=9, execution_precision="fp32", max_persistent_kv_tokens=0)
        model = TaalQwen3ForCausalLM(config)
        self.assertFalse(any(isinstance(module, NeuralMemory) for module in model.modules()))
        parameters = configure_trainable_parameters(model, "taal_only")
        self.assertEqual(sum(p.numel() for p in parameters), 2 * 16 * 9 + 1)
        model.model.layers[0].taal.residual_gate.data.fill_(0.1)
        ids = torch.tensor([self.row["input_ids"]])
        result = model(ids, labels=ids)
        result.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
        self.assertGreater(float(model.model.layers[0].taal.projection_in.weight.grad.norm()), 0)
        self.assertEqual(result.state.memory_states, {})
        model.eval()
        with torch.inference_mode():
            full = model(ids).logits
            state, parts = None, []
            for i in range(ids.shape[1]):
                out = model(ids[:, i:i+1], state=state, use_cache=True)
                state = out.state
                parts.append(out.logits)
            torch.testing.assert_close(full, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-4)
        backend = QwenBackend(model, TinyTokenizer(), {}, {"do_sample": False, "max_new_tokens": 2, "eos_token_id": 39})
        self.assertFalse(backend.capabilities.memory_trace)
        with tempfile.TemporaryDirectory() as directory:
            base_directory = Path(directory) / "base"
            Qwen3ForCausalLM(config).save_pretrained(base_directory)
            fresh, loading = TaalQwen3ForCausalLM.from_pretrained(base_directory, config=config, output_loading_info=True)
            verify_loading(fresh, loading)
            self.assertEqual(float(fresh.model.layers[0].taal.residual_gate), 0.0)
            model.save_pretrained(directory)
            restored = TaalQwen3ForCausalLM.from_pretrained(directory)
            self.assertEqual(restored.config.execution_precision, "fp32")
            torch.testing.assert_close(model(ids).logits, restored(ids).logits)
            root = Path(directory) / "generation"
            self.assertTrue(generation_audit(restored, TinyTokenizer(), {}, [self.row], root, Allocation()))
            self.assertTrue(list((root / "token_traces").glob("*.json")))
            self.assertFalse((root / "memory_traces").exists())

    def test_untouched_full_matches_original_and_swa_masks_old_tokens(self):
        base = TaalQwen3CausalLMTests().config()
        original = Qwen3ForCausalLM(base).eval()
        full = TTCDQwen3ForCausalLM(baseline_config(base, 64)).eval()
        full.load_state_dict(original.state_dict(), strict=True)
        ids = torch.tensor([self.row["input_ids"]])
        with torch.inference_mode():
            torch.testing.assert_close(original(ids).logits, full(ids).logits, atol=1e-5, rtol=1e-4)
        swa = TTCDQwen3ForCausalLM(baseline_config(base, 4)).eval()
        swa.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(swa.config.fast_weight_layers, [])
        mask = swa.model._prepare_masks(None, torch.randn(1, 12, 16), torch.arange(12), torch.arange(12))["teacher"]
        self.assertTrue((mask[0, 0, -1, :8] < -1e20).all())
        self.assertTrue((mask[0, 0, -1, 8:] == 0).all())
        with torch.inference_mode():
            expected = swa(ids).logits
            state, parts = None, []
            for i in range(12):
                out = swa(ids[:, i:i+1], state=state, use_cache=True)
                state = out.state
                parts.append(out.logits)
            torch.testing.assert_close(expected, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-4)
            self.assertLessEqual(state.past_key_values.layers[0].keys.shape[-2], 4)

    def test_backward_and_manual_derivative_comparisons_execute_without_mutation(self):
        config = TaalQwen3CausalLMTests().config(memory_max_write_grad_norm=1.0)
        model = TaalQwen3ForCausalLM(config)
        configure_trainable_parameters(model, "taal_only")
        model.model.layers[0].taal.residual_gate.data.fill_(0.1)
        with tempfile.TemporaryDirectory() as directory:
            for precision in ("fp32_auto", "bf16_auto"):
                backward_case(model, self.row, precision, Path(directory))
                report = json.loads((Path(directory) / "backward.json").read_text())
                self.assertEqual(set(report["comparisons"]), {"label", "eos", "combined"})
                for derivative in report["paths"]["full"]["manual_derivative"]:
                    for values in derivative["parameters"].values():
                        self.assertLess(values["max_abs"], 1e-5)
                if precision == "fp32_auto":
                    for metrics in report["comparisons"]["combined"]["raw"].values():
                        self.assertLess(metrics["max_abs"], 1e-5)

    def test_configs_are_matched_and_cohorts_are_complete(self):
        configs = {arm: load_config(recipe_path(arm)) for arm in ARMS}
        for config in configs.values():
            self.assertEqual(config.training.max_steps, 512)
            self.assertEqual(config.optimizer.lr, 3e-4)
            self.assertEqual(config.scheduler.final_lr, 3e-4)
            self.assertEqual(config.checkpoints.save_steps, [64, 128, 256, 384, 512])
            self.assertTrue(config.training.save_traces)
        reference = configs["r3"].to_dict()
        fp32 = configs["r3_fp32"].to_dict()
        reference.pop("wandb"); fp32.pop("wandb")
        fp32["model"].pop("execution_precision")
        self.assertEqual(reference, fp32)
        eos = configs["r3_eos"].to_dict(); eos.pop("wandb"); eos["loss"].pop("eos_weight")
        self.assertEqual(reference, eos)
        self.assertEqual(len(validation_rows()), 512)
        self.assertEqual(len(selected_families()), 16)

    def test_diagnostic_runs_saved_optimizer_state_and_resumes_cross_precision_pair(self):
        config = TaalQwen3CausalLMTests().config(memory_max_write_grad_norm=1.0)
        model = TaalQwen3ForCausalLM(config)
        parameters = configure_trainable_parameters(model, "taal_only")
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"optimizer": optimizer.state_dict(),
                        "trainable_parameters": [n for n, p in model.named_parameters() if p.requires_grad]},
                       root / "trainer_state.pt")
            def load_fixture(*args):
                return copy.deepcopy(model), TinyTokenizer(), {"checkpoint": str(root), "tokenizer": "tiny"}
            with patch("scripts.retrieval_backward_parity.load_checkpoint", side_effect=load_fixture), \
                 patch("scripts.retrieval_backward_parity.selected_families", return_value=[self.row]), \
                 patch("scripts.retrieval_backward_parity.diagnostic_rows", return_value=[]):
                output = root / "audit"
                self.assertTrue(run_diagnostics(output, Allocation(), torch.device("cpu")))
                for arm in ("C", "repeat"):
                    report = json.loads((output / arm / "fixture/cross-precision-backward.json").read_text())
                    self.assertEqual(set(report["comparisons"]), {"full", "blocked"})
                with patch("scripts.retrieval_backward_parity.backward_case") as backward:
                    self.assertTrue(run_diagnostics(output, Allocation(), torch.device("cpu")))
                    backward.assert_not_called()

    def test_strict_fp32_disables_autocast_and_restores_settings(self):
        old = torch.backends.cuda.matmul.allow_tf32
        with precision_context(torch.device("cpu"), "fp32"):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old)

    def test_pipeline_does_not_repeat_completed_training_or_ignore_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.retrieval_parallel.needs_training", return_value=False), \
                 patch("scripts.retrieval_parallel.run_segment") as train, \
                 patch("scripts.retrieval_parallel.audit_trained", return_value=True) as audit:
                run_pipeline("r3", output, "123", 2, torch.device("cpu"))
                train.assert_not_called(); audit.assert_called_once()
                self.assertTrue((output / "complete.json").is_file())
            with patch("scripts.retrieval_parallel.needs_training", return_value=True), \
                 patch("scripts.retrieval_parallel.run_segment", return_value=True), \
                 patch("scripts.retrieval_parallel.audit_trained") as audit, \
                 patch("scripts.retrieval_exposure_audit.subprocess.run") as command, \
                 patch.dict("os.environ", {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0"}):
                with self.assertRaises(SystemExit):
                    run_pipeline("r3", output, "123", 0, torch.device("cpu"))
                audit.assert_not_called()
                self.assertEqual(command.call_args.args[0], ["scontrol", "requeue", "123"])

    def test_shell_launchers_parse(self):
        for path in ("scripts/queue_retrieval_parallel.sh", "training/sbatch/taal/fs_qwen_retrieval_parallel_short.sbatch"):
            subprocess.run(["bash", "-n", path], check=True)


if __name__ == "__main__":
    unittest.main()
