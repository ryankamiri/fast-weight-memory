"""Typed experiment settings loaded from a single YAML file."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import yaml


@dataclass
class RecordRange:
    start: int = 0
    end: int | None = None


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-0.6B-Base"
    revision: str | None = None
    fast_weight_layers: list[int] = field(default_factory=lambda: [0, 7, 14, 21])
    teacher_window_size: int = 8192
    student_window_size: int = 4096
    max_persistent_tokens: int = 512
    chunk_size: int = 4096
    fast_weight_lr: float = 0.3
    use_projection: bool = True
    use_conv: bool = True
    conv_kernel_size: int = 5
    dynamic_beta: bool = True
    normalize_student_features: bool = False


@dataclass
class DataConfig:
    dataset_id: str = "ryankamiri/prolong-qwen"
    revision: str | None = None
    train: RecordRange = field(default_factory=lambda: RecordRange(64))
    val: RecordRange = field(default_factory=lambda: RecordRange(0, 64))
    batch_size: int = 1
    seq_len: int = 65536
    num_workers: int = 2
    shuffle_buffer_size: int = 128


@dataclass
class OptimizerConfig:
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class SchedulerConfig:
    warmup_steps: int = 20
    schedule_steps: int = 10000
    final_lr: float = 1e-6


@dataclass
class LoopConfig:
    seed: int = 42
    max_steps: int = 10000
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    eval_every_steps: int = 100
    eval_at_end: bool = True
    log_every_steps: int = 10


@dataclass
class ValidationConfig:
    compare_without_fast_weight_reads: bool = True


@dataclass
class CheckpointConfig:
    output_dir: str = "checkpoints"
    save_best: bool = True
    save_final: bool = True


@dataclass
class WandbConfig:
    project: str = "fast-weight-memory"
    entity: str | None = None
    name: str | None = None


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: LoopConfig = field(default_factory=LoopConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if type(self.validation.compare_without_fast_weight_reads) is not bool:
            raise ValueError("compare_without_fast_weight_reads must be a boolean")
        if not isinstance(self.checkpoints.output_dir, str) or not self.checkpoints.output_dir.strip():
            raise ValueError("checkpoints.output_dir must be a nonempty path")
        for flag in (self.checkpoints.save_best, self.checkpoints.save_final):
            if type(flag) is not bool:
                raise ValueError("Checkpoint saving flags must be booleans")
        positive_integers = {
            "max_steps": self.training.max_steps,
            "batch_size": self.data.batch_size,
            "seq_len": self.data.seq_len,
            "shuffle_buffer_size": self.data.shuffle_buffer_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "eval_every_steps": self.training.eval_every_steps,
            "log_every_steps": self.training.log_every_steps,
            "schedule_steps": self.scheduler.schedule_steps,
        }
        for name, value in positive_integers.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.data.seq_len < 2:
            raise ValueError("seq_len must allow at least one next-token prediction")
        for name, value in (("num_workers", self.data.num_workers),
                            ("max_persistent_tokens", self.model.max_persistent_tokens),
                            ("seed", self.training.seed),
                            ("warmup_steps", self.scheduler.warmup_steps)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.scheduler.warmup_steps >= self.scheduler.schedule_steps:
            raise ValueError("warmup_steps must be smaller than schedule_steps")
        for name, value in (("lr", self.optimizer.lr), ("eps", self.optimizer.eps),
                            ("max_grad_norm", self.training.max_grad_norm)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for value in (self.optimizer.beta1, self.optimizer.beta2):
            if not 0 <= value < 1:
                raise ValueError("AdamW betas must be in [0, 1)")
        if not math.isfinite(self.optimizer.weight_decay) or self.optimizer.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.scheduler.final_lr <= self.optimizer.lr:
            raise ValueError("final_lr must be between zero and peak lr")
        for bounds in (self.data.train, self.data.val):
            if type(bounds.start) is not int or bounds.start < 0:
                raise ValueError("Record range start must be a nonnegative integer")
            if bounds.end is not None and (type(bounds.end) is not int or bounds.end <= bounds.start):
                raise ValueError("Record range end must be greater than start")
        train_end = self.data.train.end if self.data.train.end is not None else math.inf
        val_end = self.data.val.end if self.data.val.end is not None else math.inf
        if max(self.data.train.start, self.data.val.start) < min(train_end, val_end):
            raise ValueError("Training and validation record ranges must not overlap")


def load_config(path: str | Path) -> TrainingConfig:
    with Path(path).open() as source:
        sections = yaml.safe_load(source)
    if not isinstance(sections, dict):
        raise ValueError("Training YAML must contain a mapping")
    section_types = {
        "model": ModelConfig, "data": DataConfig, "optimizer": OptimizerConfig,
        "scheduler": SchedulerConfig, "training": LoopConfig, "wandb": WandbConfig,
        "checkpoints": CheckpointConfig,
        "validation": ValidationConfig,
    }
    unknown = sections.keys() - section_types.keys()
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    values = {}
    for name, section_type in section_types.items():
        options = dict(sections.get(name, {}))
        if name == "data":
            for split in ("train", "val"):
                if split in options:
                    options[split] = RecordRange(**options[split])
        values[name] = section_type(**options)
    config = TrainingConfig(**values)
    config.validate()
    return config
