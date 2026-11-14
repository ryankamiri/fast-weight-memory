"""Restore training at complete update boundaries, including episode order and RNG."""

import copy
import json
from dataclasses import asdict
import random

import numpy as np
import torch

from evaluation.storage import file_sha256


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs from saved training")

        torch.cuda.set_rng_state_all(state["cuda"])


def recipe_identity(config):
    values = config.to_dict()
    values.pop("wandb")
    values["checkpoints"].pop("output_dir")
    # A continuation can extend the update budget, but cannot change the LR schedule.
    values["training"].pop("max_steps")
    return values


class UpdateBoundary:
    def __init__(self, loader, progress):
        self.loader = loader
        self.progress = progress
        self.commit()

    def commit(self):
        self.data = copy.deepcopy(self.loader.dataset.state_dict())
        self.counters = asdict(self.progress)
        self.rng = rng_state()
        self.loader_rng = self.loader.generator.get_state()

    def rewind(self):
        self.loader.dataset.load_state_dict(self.data)
        for name, value in self.counters.items():
            setattr(self.progress, name, value)
        restore_rng(self.rng)
        self.loader.generator.set_state(self.loader_rng)


class TrainingState:
    def __init__(self, optimizer, scheduler, loader):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = loader
        self.safe_to_save = True

    def capture(self, config, model, progress, checkpoints):
        if not self.safe_to_save:
            raise ValueError("Cannot save resumable state after a failed training operation")

        return {
            "schema": "training-resume/v1",
            "recipe": recipe_identity(config),
            "progress": asdict(progress),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng_state(),
            "data": self.loader.dataset.state_dict(),
            "loader_rng": self.loader.generator.get_state(),
            "best_metric_value": checkpoints.best_metric_value,
            "saved_steps": sorted(checkpoints.saved_steps),
            "trainable_parameters": [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
        }

    def restore(self, source, config, model, progress, checkpoints):
        # Only load this project's trusted local checkpoints; pickle is not an interchange format.
        metadata = json.loads((source / "training_metadata.json").read_text())
        files = metadata.get("files", {})
        if "trainer_state.pt" not in files:
            raise ValueError("Full checkpoint has no integrity manifest")

        for filename, expected in files.items():
            if "/" in filename or "\\" in filename or file_sha256(source / filename) != expected:
                raise ValueError("Resume checkpoint file checksum mismatch")

        state = torch.load(source / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state.get("schema") != "training-resume/v1" or state["recipe"] != recipe_identity(config):
            raise ValueError("Resume recipe differs from the saved training recipe")

        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if names != state["trainable_parameters"]:
            raise ValueError("Resume trainable parameter order changed")

        if state["progress"]["step"] > config.training.max_steps:
            raise ValueError("Resume checkpoint exceeds requested update budget")

        self.loader.dataset.load_state_dict(state["data"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.loader.generator.set_state(state["loader_rng"])
        for name, value in state["progress"].items():
            setattr(progress, name, value)
        checkpoints.best_metric_value = state["best_metric_value"]
        checkpoints.saved_steps = set(state["saved_steps"])
        restore_rng(state["rng"])
