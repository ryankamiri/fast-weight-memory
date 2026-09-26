import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from training.config import (
    BridgeMemoryDataConfig,
    BridgeMemoryLossConfig,
    BridgeMemoryRecordRange,
    CausalLMDataConfig,
    CausalLMLossConfig,
    RecordRange,
    TaalModelConfig,
    TaalValidationConfig,
    TrainingConfig,
    TTCDModelConfig,
    load_config,
)
from training.engine import build_scheduler, perplexity, train, validate
from training.train import configure_trainable_parameters, load_model, verify_loading
from architectures.ttcd.qwen.mlp import TTCDQwen3MLP
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config


class TinyModel(nn.Module):
    """Scalar model with the same forward contract, for exercising the trainer."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, input_ids, labels, state=None, use_cache=False):
        self.calls.append((state, use_cache, self.training, torch.is_grad_enabled()))
        loss = (self.weight - input_ids.float().mean()).square()
        return SimpleNamespace(loss=loss)


class TinyBridgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward_bridge_memory(
        self, input_ids, delayed_labels, all_token_loss_weight,
        delayed_answer_loss_weight, logits_to_keep, state=None, use_cache=False,
    ):
        loss = self.weight.square()
        logits = torch.zeros(input_ids.shape[0], logits_to_keep, 32)
        targets = delayed_labels[:, -1]
        logits[torch.arange(input_ids.shape[0]), -2, targets] = 4
        return SimpleNamespace(
            loss=loss,
            all_token_loss=None,
            delayed_answer_loss=loss,
            logits=logits,
        )


class TinyLoader:
    def __init__(self, batches):
        self.batches = batches
        self.epochs = []
        self.dataset = self

    def __iter__(self):
        return iter(self.batches)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def batch(value, size=1, length=4):
    ids = torch.full((size, length), value, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


class TrainingConfigTests(unittest.TestCase):
    def test_2k_chunk_changes_only_chunk_size_and_run_name(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        original = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_1k.yaml")
        larger_chunk = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_2k.yaml")
        self.assertEqual(larger_chunk.model.chunk_size, 2048)
        self.assertEqual(larger_chunk.data.batch_size, 1)
        original.model.chunk_size = 2048
        original.wandb.name = "qwen3-0.6b-cpt-fw-swa-4k-2k-2k-64k"
        self.assertEqual(larger_chunk, original)

    def test_native_4k_swa_matches_fw_training_recipe(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        swa = load_config(folder / "qwen3_0_6b_cpt_swa_4k.yaml")
        fw = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_1k.yaml")
        self.assertEqual(swa.model.teacher_window_size, 4096)
        self.assertEqual(swa.model.fast_weight_layers, [])
        self.assertEqual(swa.data.batch_size, 1)
        self.assertEqual(swa.validation.fast_weight_read_scales, [1.0])
        fw.model.fast_weight_layers = []
        fw.validation.fast_weight_read_scales = [1.0]
        fw.wandb.name = "qwen3-0.6b-cpt-swa-4k-64k"
        self.assertEqual(swa, fw)

    def test_2k_chunk_batch_two_variant(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        original = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_2k.yaml")
        batch_two = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_2k_bs2.yaml")
        self.assertEqual(batch_two.data.batch_size, 2)
        original.data.batch_size = 2
        original.wandb.name += "-bs2"
        self.assertEqual(batch_two, original)

    def test_window_decomposition_configs_change_only_named_windows(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        teacher_8k = load_config(folder / "qwen3_0_6b_cpt_fw_swa_8k_2k_2k.yaml")
        expected_teacher_8k = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_2k.yaml")
        expected_teacher_8k.model.teacher_window_size = 8192
        expected_teacher_8k.wandb.name = "qwen3-0.6b-cpt-fw-swa-8k-2k-2k-64k"
        self.assertEqual(teacher_8k, expected_teacher_8k)

        student_4k = load_config(folder / "qwen3_0_6b_cpt_fw_swa_8k_4k_2k.yaml")
        expected_student_4k = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_2k.yaml")
        expected_student_4k.model.teacher_window_size = 8192
        expected_student_4k.model.student_window_size = 4096
        expected_student_4k.wandb.name = "qwen3-0.6b-cpt-fw-swa-8k-4k-2k-64k"
        self.assertEqual(student_4k, expected_student_4k)

    def test_small_window_configs_change_one_variable_per_transition(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        recipes = [
            ("qwen3_0_6b_cpt_fw_swa_4k_2k_1k.yaml", 4096, 2048, 1024),
            ("qwen3_0_6b_cpt_fw_swa_4k_1k_1k.yaml", 4096, 1024, 1024),
            ("qwen3_0_6b_cpt_fw_swa_2k_1k_1k.yaml", 2048, 1024, 1024),
            ("qwen3_0_6b_cpt_fw_swa_2k_1k_512.yaml", 2048, 1024, 512),
        ]
        configs = [load_config(folder / filename) for filename, *_ in recipes]

        for config, (_, teacher, student, chunk) in zip(configs, recipes):
            self.assertEqual(
                (config.model.teacher_window_size, config.model.student_window_size, config.model.chunk_size),
                (teacher, student, chunk),
            )
            self.assertEqual(config.data.batch_size, 1)
            self.assertEqual(config.training.gradient_accumulation_steps, 16)

        for previous, current in zip(configs, configs[1:]):
            previous_values = previous.to_dict()
            current_values = current.to_dict()
            previous_values["wandb"]["name"] = current_values["wandb"]["name"]
            changed_model_fields = {
                key for key in previous_values["model"]
                if previous_values["model"][key] != current_values["model"][key]
            }
            self.assertEqual(len(changed_model_fields), 1)
            for key in changed_model_fields:
                previous_values["model"][key] = current_values["model"][key]
            self.assertEqual(previous_values, current_values)

    def test_yaml_defaults(self):
        config = load_config(Path(__file__).resolve().parents[1] / "training/configs/ttcd/qwen3_0_6b.yaml")
        expected = TrainingConfig()
        expected.training.eval_every_steps = 10
        expected.training.log_every_steps = 1
        expected.wandb.entity = "ryanamiri05-northeastern-university"
        expected.wandb.name = "qwen3-0.6b-cpt-64k"
        self.assertIsInstance(config.model, TTCDModelConfig)
        self.assertEqual(config, expected)
        self.assertNotIn("mode", config.to_dict()["wandb"])
        self.assertIsInstance(config.optimizer.lr, float)
        self.assertEqual(config.training.max_steps, 10000)

    def test_overlapping_ranges_and_invalid_schedule(self):
        config = TrainingConfig()
        config.data.train = RecordRange(63)
        with self.assertRaisesRegex(ValueError, "overlap"):
            config.validate()
        config.data.train = RecordRange(64)
        config.scheduler.schedule_steps = config.scheduler.warmup_steps
        with self.assertRaisesRegex(ValueError, "warmup"):
            config.validate()

    def test_invalid_read_scales(self):
        for scales in ([], [0.0], [1.0, 1.0], [1.0, -0.5], [1.0, float("nan")], [True, 0.0]):
            with self.subTest(scales=scales):
                config = TrainingConfig()
                config.validation.fast_weight_read_scales = scales
                with self.assertRaises(ValueError):
                    config.validate()

    def test_bridge_loss_and_overlap_configuration(self):
        config = TrainingConfig()
        config.data = BridgeMemoryDataConfig(
            dataset_config="t4096-s2048-c1024",
            batch_size=1,
            seq_len=7695,
            train=BridgeMemoryRecordRange(0, 64, split="test", conditions=["bridge"]),
            val=BridgeMemoryRecordRange(0, 64, split="test", conditions=["bridge"]),
            allow_train_val_overlap=True,
        )
        config.loss = BridgeMemoryLossConfig(all_tokens_weight=0.0, delayed_answer_weight=1.0)
        config.training.trainable_parameters = "ttcd_only"
        config.validate()

        config.data.allow_train_val_overlap = False
        with self.assertRaisesRegex(ValueError, "overlap"):
            config.validate()
        config.data.allow_train_val_overlap = True
        config.data.batch_size = 2
        config.validate()

    def test_confirmation_experiment_configs(self):
        folder = Path(__file__).resolve().parents[1] / "training/configs/ttcd"
        bounded = load_config(folder / "qwen3_0_6b_cpt_fw_swa_4k_2k_1k.yaml")
        full_teacher = load_config(folder / "qwen3_0_6b_cpt_fw_full_65k_2k_1k.yaml")

        expected = bounded.to_dict()
        expected["model"]["teacher_window_size"] = 65536
        expected["wandb"]["name"] = "qwen3-0.6b-cpt-fw-full-65k-2k-1k"
        self.assertEqual(full_teacher.to_dict(), expected)

        overfit = load_config(folder / "qwen3_0_6b_bridge_overfit.yaml")
        self.assertEqual(overfit.data.record_format, "bridge_memory")
        self.assertIsInstance(overfit.data, BridgeMemoryDataConfig)
        self.assertIsInstance(overfit.loss, BridgeMemoryLossConfig)
        self.assertTrue(overfit.data.allow_train_val_overlap)
        self.assertEqual(overfit.loss, BridgeMemoryLossConfig(0.0, 1.0))
        self.assertEqual(overfit.training.trainable_parameters, "ttcd_only")
        self.assertEqual(overfit.checkpoints.selection_mode, "max")

        curriculum = load_config(folder / "qwen3_0_6b_bridge_curriculum.yaml")
        self.assertEqual(curriculum.loss, BridgeMemoryLossConfig(1.0, 1.0))
        self.assertEqual((curriculum.data.train.start, curriculum.data.train.end), (0, 96))
        self.assertEqual((curriculum.data.val.start, curriculum.data.val.end), (96, 128))
        self.assertFalse(curriculum.data.allow_train_val_overlap)
        self.assertEqual(curriculum.training.trainable_parameters, "all")

        self.assertIsInstance(full_teacher.data, CausalLMDataConfig)
        self.assertIsInstance(full_teacher.loss, CausalLMLossConfig)

    def test_taal_delayed_recall_overfit_config(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml"
        )
        config = load_config(path)

        self.assertIsInstance(config.model, TaalModelConfig)
        self.assertIsInstance(config.validation, TaalValidationConfig)
        self.assertEqual(config.model.architecture, "taal")
        self.assertEqual(config.model.revision, "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
        self.assertEqual(config.model.working_memory_size, 2048)
        self.assertEqual(config.model.memory_dim, 128)
        self.assertEqual(config.model.memory_chunk_size, 64)
        self.assertEqual(config.data.train.conditions, ["no_bridge"])
        self.assertEqual((config.data.train.start, config.data.train.end), (0, 64))
        self.assertEqual((config.data.val.start, config.data.val.end), (0, 32))
        self.assertEqual(config.data.batch_size, 1)
        self.assertEqual(config.loss, BridgeMemoryLossConfig(0.0, 1.0))
        self.assertEqual(config.training.gradient_accumulation_steps, 8)
        self.assertEqual(config.training.max_steps, 500)
        self.assertEqual(config.scheduler.schedule_steps, 500)
        self.assertEqual(config.scheduler.warmup_steps, 20)
        self.assertEqual(config.training.eval_every_steps, 25)
        self.assertEqual(config.training.trainable_parameters, "taal_only")
        self.assertEqual(config.validation.memory_read_scales, [1.0, 0.5, 0.0])

    def test_unknown_yaml_setting_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text("training:\n  unknown_setting: 5\n")
            with self.assertRaises(TypeError):
                load_config(path)

    def test_scheduler_warmup_decay_and_hold(self):
        config = TrainingConfig()
        config.scheduler.warmup_steps = 2
        config.scheduler.schedule_steps = 6
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.optimizer.lr)
        scheduler = build_scheduler(optimizer, config)
        rates = []
        for _ in range(9):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], config.optimizer.lr / 2)
        self.assertAlmostEqual(rates[1], config.optimizer.lr)
        self.assertGreater(rates[2], rates[3])
        for rate in rates[5:]:
            self.assertAlmostEqual(rate, config.scheduler.final_lr)


class TrainingLoopTests(unittest.TestCase):
    def test_paired_validation_and_restore(self):
        model = TinyModel()
        model.mlp = TTCDQwen3MLP(Qwen3Config(hidden_size=8, intermediate_size=12),
                              is_fast_weight_layer=True)
        def forward(input_ids, labels, state=None, use_cache=False):
            self.assertIsNone(state)
            self.assertFalse(use_cache)
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return SimpleNamespace(loss=torch.tensor(3.0 - model.mlp.fast_weight_read_scale))
        model.forward = forward
        model.mlp.fast_weight_read_scale = 0.25
        loader = TinyLoader([batch(1), batch(2)])
        metrics = validate(model, loader, torch.device("cpu"), read_scales=[1.0, 0.5, 0.0])
        self.assertEqual(metrics["val/loss"], 2.0)
        self.assertEqual(metrics["val/loss_fw_read_scale_0.5"], 2.5)
        self.assertEqual(metrics["val/fw_read_loss_improvement_scale_0.5"], 0.5)
        self.assertAlmostEqual(metrics["val/perplexity_fw_read_scale_0.5"], math.exp(2.5))
        self.assertEqual(metrics["val/loss_fw_read_scale_0"], 3.0)
        self.assertEqual(metrics["val/fw_read_loss_improvement_scale_1"], 1.0)
        self.assertAlmostEqual(metrics["val/perplexity_fw_read_scale_0"], math.exp(3))
        self.assertTrue(model.training)
        self.assertEqual(model.mlp.fast_weight_read_scale, 0.25)
        with patch("training.engine._validate_pass", side_effect=[metrics, RuntimeError("failed")]):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                validate(model, loader, torch.device("cpu"), read_scales=[1.0, 0.5, 0.0])
        self.assertEqual(model.mlp.fast_weight_read_scale, 0.25)
        with patch("training.engine._validate_pass", side_effect=[metrics, None]):
            self.assertIsNone(validate(model, loader, torch.device("cpu"),
                                       read_scales=[1.0, 0.5, 0.0]))
        self.assertEqual(model.mlp.fast_weight_read_scale, 0.25)

    def test_bridge_validation_reports_delayed_loss_and_candidate_accuracy(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 7]]),
            "delayed_labels": torch.tensor([[-100, -100, -100, 7]]),
            "target_token_ids": torch.tensor([7]),
            "candidate_token_ids": torch.tensor([[7, 8, 9]]),
            "conditions": ["bridge"],
            "query_variants": ["exact"],
        }
        metrics = validate(
            TinyBridgeModel(), TinyLoader([batch]), torch.device("cpu"),
            read_scales=[1.0, 0.5, 0.0],
            loss_config=BridgeMemoryLossConfig(all_tokens_weight=0.0, delayed_answer_weight=1.0),
        )
        self.assertEqual(metrics["val/loss"], 0.25)
        self.assertEqual(metrics["val/delayed_answer_loss"], 0.25)
        self.assertEqual(metrics["val/bridge_exact_candidate_accuracy"], 1.0)
        self.assertEqual(metrics["val/all_vocabulary_top_1_accuracy"], 1.0)

    def test_taal_validation_reports_all_memory_read_scales(self):
        model = TaalQwen3ForCausalLM(TaalQwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            working_memory_size=4,
            memory_dim=4,
            memory_depth=2,
            memory_conv_kernel_size=2,
            num_persistent_tokens=2,
        ))
        input_ids = torch.tensor([[1, 2, 3, 7]])
        batch = {
            "input_ids": input_ids,
            "delayed_labels": torch.tensor([[-100, -100, -100, 7]]),
            "target_token_ids": torch.tensor([7]),
            "candidate_token_ids": torch.tensor([[7, 8, 9]]),
            "conditions": ["no_bridge"],
            "query_variants": ["exact"],
        }
        metrics = validate(
            model,
            TinyLoader([batch]),
            torch.device("cpu"),
            read_scales=[1.0, 0.5, 0.0],
            loss_config=BridgeMemoryLossConfig(
                all_tokens_weight=0.0,
                delayed_answer_weight=1.0,
            ),
        )

        for scale in (1, 0.5, 0):
            self.assertIn(
                f"val/no_bridge_exact_candidate_accuracy_memory_read_scale_{scale:g}",
                metrics,
            )
        self.assertIn("val/memory_read_loss_improvement_scale_1", metrics)

    def test_baseline_validation_does_not_repeat(self):
        model = TinyModel()
        validate(model, TinyLoader([batch(1)]), torch.device("cpu"),
                 read_scales=[1.0, 0.5, 0.0])
        self.assertEqual(len(model.calls), 1)

    def test_max_steps_stops_without_external_signal_and_validates_once(self):
        model = TinyModel()
        config = self.config()
        config.training.max_steps = 2
        progress, logs, scheduler = self.run_loop(
            model, TinyLoader([batch(1)] * 10), config, successful_steps=99,
        )
        self.assertEqual(progress.step, 2)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(sum(training for _, _, training, _ in model.calls), 4)
        self.assertEqual([item["train/step"] for item in logs if "val/loss" in item], [2])

    def test_max_steps_must_be_positive(self):
        for value in (0, -1, 1.5, True):
            config = self.config()
            config.training.max_steps = value
            with self.assertRaisesRegex(ValueError, "max_steps"):
                config.validate()

    def test_real_qwen_load_and_checkpointed_training(self):
        native = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4,
        ))
        config = self.config()
        config.model.fast_weight_layers = [0]
        config.model.teacher_window_size = 4
        config.model.student_window_size = 2
        config.model.chunk_size = 2
        with TemporaryDirectory() as folder:
            native.save_pretrained(folder)
            config.model.model_id = folder
            model = load_model(config)
        for key, value in native.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value)
        self.assertTrue(model.is_gradient_checkpointing)
        progress, logs, _ = self.run_loop(model, TinyLoader([batch(1), batch(2)]), config, 1)
        self.assertEqual(progress.step, 1)
        self.assertTrue(math.isfinite(logs[-1]["val/loss"]))

    def test_real_qwen_loads_as_taal_with_only_memory_weights_added(self):
        native = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        ))
        config = self.config()
        config.model = TaalModelConfig(
            working_memory_size=4,
            max_persistent_kv_tokens=2,
            memory_dim=4,
            memory_depth=2,
            memory_conv_kernel_size=2,
            num_persistent_tokens=2,
        )
        config.validation = TaalValidationConfig()
        config.training.gradient_checkpointing = True
        config.training.trainable_parameters = "taal_only"
        with TemporaryDirectory() as folder:
            native.save_pretrained(folder)
            config.model.model_id = folder
            model = load_model(config)

        self.assertIsInstance(model, TaalQwen3ForCausalLM)
        self.assertTrue(model.is_gradient_checkpointing)
        for key, value in native.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value)

    def config(self):
        config = TrainingConfig()
        config.training.gradient_accumulation_steps = 2
        config.training.log_every_steps = 1
        config.training.eval_every_steps = 2
        config.training.max_grad_norm = 100
        return config

    def run_loop(self, model, loader, config, successful_steps=2):
        config.training.max_steps = min(config.training.max_steps, successful_steps)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        logs = []
        updates = 0
        step = optimizer.step

        def counted_step():
            nonlocal updates
            step()
            updates += 1

        with patch.object(optimizer, "step", side_effect=counted_step):
            progress = train(
                model, loader, TinyLoader([batch(2)]), optimizer, scheduler,
                config, torch.device("cpu"), logs.append,
                lambda: False,
            )
        return progress, logs, scheduler

    def test_accumulation_averages_microbatch_gradients(self):
        model = TinyModel()
        config = self.config()
        config.training.eval_at_end = False
        loader = TinyLoader([batch(1), batch(3, size=2)])
        progress, logs, scheduler = self.run_loop(model, loader, config, successful_steps=1)
        # Each microbatch contributes equally, as in TTCD: (-1 + -5) / 2.
        expected = 0.5 - 0.01 * (-3)
        self.assertAlmostEqual(model.weight.item(), expected, places=6)
        self.assertAlmostEqual(logs[0]["train/loss"], (0.25 + 2 * 6.25) / 3)
        self.assertEqual(progress.tokens_seen, 12)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(all(state is None and not cache for state, cache, _, _ in model.calls))

    def test_validation_cadence_epochs_and_final_validation(self):
        model = TinyModel()
        loader = TinyLoader([batch(1)])
        progress, logs, _ = self.run_loop(model, loader, self.config(), successful_steps=3)
        self.assertEqual(progress.step, 3)
        self.assertEqual(loader.epochs, list(range(6)))
        self.assertEqual([item["train/step"] for item in logs if "val/loss" in item], [2, 3])
        self.assertTrue(model.calls[0][2])  # No initial validation.
        for state, cache, training, grad_enabled in model.calls:
            self.assertIsNone(state)
            self.assertFalse(cache)
            if not training:
                self.assertFalse(grad_enabled)
        self.assertTrue(model.training)

    def test_nonfinite_loss_discards_previous_accumulation(self):
        model = TinyModel()
        original = model.forward
        calls = 0

        def forward(**kwargs):
            nonlocal calls
            calls += 1
            result = original(**kwargs)
            if calls == 2:
                result.loss = result.loss * float("nan")
            return result

        config = self.config()
        config.training.eval_at_end = False
        with patch.object(model, "forward", side_effect=forward):
            progress, _, scheduler = self.run_loop(
                model, TinyLoader([batch(9), batch(9), batch(1), batch(1)]), config, 1,
            )
        self.assertAlmostEqual(model.weight.item(), 0.51, places=6)
        self.assertEqual(progress.nan_losses, 1)
        self.assertEqual(progress.skipped_updates, 1)
        self.assertEqual(progress.step, 1)
        self.assertEqual(scheduler.last_epoch, 1)

    def test_nonfinite_gradient_skips_optimizer_and_scheduler(self):
        for invalid in (float("nan"), float("inf")):
            model = TinyModel()
            calls = 0

            def corrupt_first_gradient(gradient):
                nonlocal calls
                calls += 1
                return gradient * invalid if calls == 1 else gradient

            hook = model.weight.register_hook(corrupt_first_gradient)
            config = self.config()
            config.training.gradient_accumulation_steps = 1
            config.training.eval_at_end = False
            config.training.max_steps = 1
            progress, _, scheduler = self.run_loop(model, TinyLoader([batch(1), batch(1)]), config, 99)
            hook.remove()
            self.assertAlmostEqual(model.weight.item(), 0.51, places=6)
            self.assertEqual(progress.skipped_updates, 1)
            self.assertEqual(progress.nan_gradients + progress.infinite_gradients, 1)
            self.assertEqual(scheduler.last_epoch, 1)

    def test_validation_weights_tokens_and_restores_mode(self):
        model = TinyModel()
        result = validate(model, TinyLoader([batch(1), batch(3, size=2)]), torch.device("cpu"))
        self.assertAlmostEqual(result["val/loss"], (0.25 + 2 * 6.25) / 3)
        self.assertEqual(result["val/records"], 3)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "no records"):
            validate(model, TinyLoader([]), torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertTrue(math.isnan(perplexity(float("nan"))))

    def test_loading_allows_only_added_parameters(self):
        mlp = SimpleNamespace(W_proj=nn.Parameter(torch.ones(2, 2)), beta_proj=None,
                              teacher_conv=None, student_conv=None)
        model = SimpleNamespace(
            config=TTCDQwen3Config(
                num_hidden_layers=1,
                fast_weight_layers=[0],
            ),
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
        )
        verify_loading(model, {"missing_keys": ["model.layers.0.mlp.W_proj"]})
        for info in ({"missing_keys": ["model.embed_tokens.weight"]},
                     {"unexpected_keys": ["other.weight"]},
                     {"mismatched_keys": ["lm_head.weight"]},
                     {"error_msgs": ["bad checkpoint"]}):
            with self.assertRaisesRegex(ValueError, "verification"):
                verify_loading(model, info)

    def test_fast_weight_only_parameter_scope(self):
        config = self.config()
        config.model.fast_weight_layers = [0]
        model = TTCDQwen3MLP(Qwen3Config(hidden_size=8, intermediate_size=12),
                           is_fast_weight_layer=True)
        wrapper = SimpleNamespace(
            config=SimpleNamespace(fast_weight_layers=[0]),
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=model)]),
            parameters=model.parameters,
        )
        parameters = configure_trainable_parameters(wrapper, "ttcd_only")
        trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertEqual(trainable_names, {
            "W_proj", "beta_proj", "teacher_conv.weight", "student_conv.weight",
        })
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 64 + 8 + 120)

    def test_taal_only_parameter_scope(self):
        model = TaalQwen3ForCausalLM(TaalQwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            working_memory_size=4,
            memory_dim=4,
            memory_depth=2,
            memory_conv_kernel_size=2,
            num_persistent_tokens=2,
        ))
        parameters = configure_trainable_parameters(model, "taal_only")
        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(all(".taal." in name for name in trainable_names))
        self.assertEqual(
            sum(parameter.numel() for parameter in parameters),
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if ".taal." in name
            ),
        )


if __name__ == "__main__":
    unittest.main()
