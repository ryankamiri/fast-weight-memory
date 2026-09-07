import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from training.checkpoints import ModelCheckpoints
from training.config import TrainingConfig
from training.engine import Progress, train, validate
from test_training import TinyLoader, TinyModel, batch


class CheckpointTests(unittest.TestCase):
    def model(self):
        return FWQwen3ForCausalLM(FWQwen3Config(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, fast_weight_layers=[0], teacher_window_size=4,
            student_window_size=2, chunk_size=2,
        ))

    def test_best_replacement_final_and_reload(self):
        with TemporaryDirectory() as directory:
            config = TrainingConfig()
            config.checkpoints.output_dir = directory
            saves = ModelCheckpoints(config, "run")
            model = self.model()
            progress = Progress(step=1)
            saves.on_validation(model, progress, {"val/loss": 3.0})
            best = Path(directory) / "run/best"
            with patch.object(saves, "save", wraps=saves.save) as write:
                for loss in (4.0, 3.0, float("nan"), float("inf")):
                    saves.on_validation(model, progress, {"val/loss": loss})
                write.assert_not_called()
            progress.step = 2
            with torch.no_grad():
                model.lm_head.weight.add_(0.5)
            saves.on_validation(model, progress, {"val/loss": 2.0})
            saves.save_final(model, progress, "signal")
            self.assertEqual(sorted(p.name for p in best.parent.iterdir()), ["best", "final"])
            for name in ("best", "final"):
                restored = FWQwen3ForCausalLM.from_pretrained(best.parent / name)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)
            metadata = json.loads((best / "training_metadata.json").read_text())
            self.assertEqual(metadata["progress"]["step"], 2)
            self.assertEqual(metadata["val_loss"], 2.0)

    def test_disabled_saves_and_failed_replacement_preserve_best(self):
        with TemporaryDirectory() as directory:
            config = TrainingConfig()
            config.checkpoints.output_dir = directory
            saves = ModelCheckpoints(config, "run")
            model = self.model()
            saves.on_validation(model, Progress(), {"val/loss": 3.0})
            with patch.object(model, "save_pretrained", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    saves.on_validation(model, Progress(step=1), {"val/loss": 2.0})
            self.assertEqual(saves.best_loss, 3.0)
            self.assertEqual([p.name for p in saves.directory.iterdir()], ["best"])
            config.checkpoints.save_best = config.checkpoints.save_final = False
            with patch.object(saves, "save") as write:
                saves.on_validation(model, Progress(), {"val/loss": 1.0})
                saves.save_final(model, Progress(), "completed")
                write.assert_not_called()

    def test_stop_during_validation_returns_no_partial_score(self):
        model = TinyModel()
        stopped = False

        def batches():
            nonlocal stopped
            yield batch(1)
            stopped = True
            yield batch(2)

        self.assertIsNone(validate(model, batches(), torch.device("cpu"), lambda: stopped))
        self.assertTrue(model.training)

    def test_stop_discards_partial_gradients_and_skips_final_validation(self):
        model = TinyModel()
        config = TrainingConfig()
        progress = Progress()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        callback = unittest.mock.Mock()
        train(model, TinyLoader([batch(1)]), TinyLoader([batch(2)]), optimizer,
              scheduler, config, torch.device("cpu"), lambda _: None,
              lambda: len(model.calls) > 0, on_validation=callback, progress=progress)
        self.assertEqual(progress.step, 0)
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(model.weight.grad)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
