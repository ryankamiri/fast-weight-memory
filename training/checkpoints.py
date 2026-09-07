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
        self.best_loss = math.inf

    def save(self, model, progress, name, *, val_loss=None, reason=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / name
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.directory))
        backup = self.directory / f".{name}-previous"
        
        try:
            model.save_pretrained(temporary, safe_serialization=True)
            metadata = {
                "progress": asdict(progress), "val_loss": val_loss, "reason": reason,
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
        loss = metrics["val/loss"]
        if self.config.checkpoints.save_best and math.isfinite(loss) and loss < self.best_loss:
            self.save(model, progress, "best", val_loss=loss, reason="lowest validation loss")
            self.best_loss = loss

    def save_final(self, model, progress, reason):
        if self.config.checkpoints.save_final:
            self.save(model, progress, "final", reason=reason)
