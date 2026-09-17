"""Keep one best and one final HF model snapshot per run, without optimizer state."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import tempfile


class ModelCheckpoints:
    def __init__(self, config, run_id):
        self.config = config
        self.directory = Path(config.checkpoints.output_dir) / run_id
        self.best_metric_value: float | None = None

    def save(self, model, progress, name, *, val_loss=None, selection_value=None, reason=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / name
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.directory))
        backup = self.directory / f".{name}-previous"
        
        try:
            model.save_pretrained(temporary, safe_serialization=True)
            metadata = {
                "progress": asdict(progress), "val_loss": val_loss, "reason": reason,
                "selection_metric": self.config.checkpoints.selection_metric,
                "selection_value": selection_value,
                "training_config": self.config.to_dict(),
                "contents": "Model/config only. No optimizer, scheduler, RNG, KV cache, or session fast weights.",
            }
            (temporary / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            # Keep the previous snapshot intact until the new one is fully written.
            if backup.exists():
                raise FileExistsError(f"Recover or remove interrupted checkpoint backup first: {backup}")
            if target.exists():
                target.rename(backup)
            try:
                temporary.rename(target)
            except BaseException:
                if backup.exists():
                    backup.rename(target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Saved {name} model at optimizer step {progress.step}: {target}", flush=True)

    def on_validation(self, model, progress, metrics):
        metric = self.config.checkpoints.selection_metric
        if metric not in metrics:
            raise ValueError(f"Checkpoint selection metric was not logged: {metric}")
        value = metrics[metric]
        mode = self.config.checkpoints.selection_mode
        improved = (
            self.best_metric_value is None
            or (mode == "min" and value < self.best_metric_value)
            or (mode == "max" and value > self.best_metric_value)
        )
        if self.config.checkpoints.save_best and math.isfinite(value) and improved:
            self.save(
                model, progress, "best", val_loss=metrics.get("val/loss"),
                selection_value=value,
                reason=f"best {metric} ({mode})",
            )
            self.best_metric_value = value

    def save_final(self, model, progress, reason):
        if self.config.checkpoints.save_final:
            self.save(model, progress, "final", reason=reason)
