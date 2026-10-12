"""Optional exact-training-set convergence gate; disabled in existing recipes."""

import math

from training.config import LoopConfig


def reached_validation_target(metrics: dict[str, float], settings: LoopConfig) -> bool:
    if settings.early_stop_val_loss is None:
        return False
    loss = metrics["val/loss"]
    accuracy = metrics[settings.early_stop_accuracy_metric]
    return math.isfinite(loss) and loss < settings.early_stop_val_loss and accuracy == 1.0
