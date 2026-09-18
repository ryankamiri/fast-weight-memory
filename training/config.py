"""Typed experiment settings loaded from a single YAML file."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class RecordRange:
    start: int = 0
    end: int | None = None
    split: str = "train"


@dataclass
class BridgeMemoryRecordRange(RecordRange):
    conditions: list[str] | None = None
    query_variants: list[str] | None = None


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
    fast_weight_read_scale: float = 1.0
    use_projection: bool = True
    use_conv: bool = True
    conv_kernel_size: int = 5
    dynamic_beta: bool = True
    normalize_student_features: bool = False


@dataclass
class DataConfig(ABC):
    dataset_id: str = "ryankamiri/prolong-qwen"
    revision: str | None = None
    allow_train_val_overlap: bool = False
    batch_size: int = 1
    seq_len: int = 65536
    num_workers: int = 2
    shuffle_buffer_size: int = 128

    @property
    @abstractmethod
    def record_format(self) -> str:
        """Dataset row schema selected by this concrete config type."""


@dataclass
class CausalLMDataConfig(DataConfig):
    train: RecordRange = field(default_factory=lambda: RecordRange(64))
    val: RecordRange = field(default_factory=lambda: RecordRange(0, 64))

    @property
    def record_format(self) -> Literal["causal_lm"]:
        return "causal_lm"


@dataclass
class BridgeMemoryDataConfig(DataConfig):
    dataset_id: str = "ryankamiri/ttcd-bridge-memory"
    dataset_config: str = ""
    train: BridgeMemoryRecordRange = field(default_factory=BridgeMemoryRecordRange)
    val: BridgeMemoryRecordRange = field(default_factory=BridgeMemoryRecordRange)

    @property
    def record_format(self) -> Literal["bridge_memory"]:
        return "bridge_memory"


@dataclass
class LossConfig(ABC):
    all_tokens_weight: float = 1.0

    @property
    @abstractmethod
    def objective(self) -> str:
        """Training objective selected by this concrete config type."""


@dataclass
class CausalLMLossConfig(LossConfig):
    @property
    def objective(self) -> Literal["causal_lm"]:
        return "causal_lm"


@dataclass
class BridgeMemoryLossConfig(LossConfig):
    delayed_answer_weight: float = 0.0

    @property
    def objective(self) -> Literal["bridge_memory"]:
        return "bridge_memory"


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
    trainable_parameters: str = "all"


@dataclass
class ValidationConfig:
    fast_weight_read_scales: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.0])


@dataclass
class CheckpointConfig:
    output_dir: str = "checkpoints"
    save_best: bool = True
    save_final: bool = True
    selection_metric: str = "val/loss"
    selection_mode: str = "min"


@dataclass
class WandbConfig:
    project: str = "fast-weight-memory"
    entity: str | None = None
    name: str | None = None


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: CausalLMDataConfig | BridgeMemoryDataConfig = field(default_factory=CausalLMDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: CausalLMLossConfig | BridgeMemoryLossConfig = field(default_factory=CausalLMLossConfig)
    training: LoopConfig = field(default_factory=LoopConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self):
        values = asdict(self)
        values["data"]["record_format"] = self.data.record_format
        values["loss"]["objective"] = self.loss.objective
        return values

    def validate(self):
        if isinstance(self.data, CausalLMDataConfig) != isinstance(self.loss, CausalLMLossConfig):
            raise ValueError("Causal-LM data requires a causal-LM loss config")
        if isinstance(self.data, BridgeMemoryDataConfig) != isinstance(self.loss, BridgeMemoryLossConfig):
            raise ValueError("Bridge-memory data requires a bridge-memory loss config")
        if self.training.trainable_parameters not in {"all", "fast_weight_only"}:
            raise ValueError("training.trainable_parameters must be all or fast_weight_only")
        if self.training.trainable_parameters == "fast_weight_only" and not self.model.fast_weight_layers:
            raise ValueError("fast_weight_only training requires fast-weight layers")
        weights = [("all_tokens_weight", self.loss.all_tokens_weight)]
        if isinstance(self.loss, BridgeMemoryLossConfig):
            weights.append(("delayed_answer_weight", self.loss.delayed_answer_weight))
        for name, value in weights:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if isinstance(self.loss, CausalLMLossConfig) and self.loss.all_tokens_weight == 0:
            raise ValueError("Causal-LM all_tokens_weight must be positive")
        if isinstance(self.data, BridgeMemoryDataConfig):
            if not self.data.dataset_config.strip():
                raise ValueError("bridge_memory data requires data.dataset_config")
            if self.data.batch_size != 1:
                raise ValueError("Variable-length bridge_memory records require batch_size=1")
            if self.loss.delayed_answer_weight == 0:
                raise ValueError("bridge_memory training requires delayed_answer_weight > 0")
        scales = self.validation.fast_weight_read_scales
        if not isinstance(scales, list) or not scales:
            raise ValueError("fast_weight_read_scales must be a nonempty list")
        for scale in [self.model.fast_weight_read_scale, *scales]:
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
                raise ValueError("Fast-weight read scales must be finite nonnegative numbers")
        if 1.0 not in scales or len(set(scales)) != len(scales):
            raise ValueError("Validation read scales must include 1.0 and have no duplicates")
        if not isinstance(self.checkpoints.output_dir, str) or not self.checkpoints.output_dir.strip():
            raise ValueError("checkpoints.output_dir must be a nonempty path")
        if not isinstance(self.checkpoints.selection_metric, str) or not self.checkpoints.selection_metric.startswith("val/"):
            raise ValueError("checkpoints.selection_metric must name a validation metric")
        if self.checkpoints.selection_mode not in {"min", "max"}:
            raise ValueError("checkpoints.selection_mode must be min or max")
        for name, flag in (
            ("allow_train_val_overlap", self.data.allow_train_val_overlap),
            ("gradient_checkpointing", self.training.gradient_checkpointing),
            ("eval_at_end", self.training.eval_at_end),
            ("save_best", self.checkpoints.save_best),
            ("save_final", self.checkpoints.save_final),
        ):
            if type(flag) is not bool:
                raise ValueError(f"{name} must be a boolean")
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
            if not isinstance(bounds.split, str) or not bounds.split:
                raise ValueError("Dataset split names must be nonempty strings")
            if isinstance(bounds, BridgeMemoryRecordRange):
                for name, values in (
                    ("conditions", bounds.conditions), ("query_variants", bounds.query_variants),
                ):
                    if values is not None and (
                        not isinstance(values, list) or not values
                        or any(not isinstance(value, str) or not value for value in values)
                        or len(values) != len(set(values))
                    ):
                        raise ValueError(f"{name} must be null or a nonempty list of unique strings")
        if not self.data.allow_train_val_overlap and self.data.train.split == self.data.val.split:
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
        "model": ModelConfig, "optimizer": OptimizerConfig,
        "scheduler": SchedulerConfig,
        "training": LoopConfig, "wandb": WandbConfig,
        "checkpoints": CheckpointConfig,
        "validation": ValidationConfig,
    }
    unknown = sections.keys() - (section_types.keys() | {"data", "loss"})
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    data_options = dict(sections.get("data", {}))
    record_format = data_options.pop("record_format", "causal_lm")
    if record_format == "causal_lm":
        data_type = CausalLMDataConfig
        range_type = RecordRange
        loss_type = CausalLMLossConfig
    elif record_format == "bridge_memory":
        data_type = BridgeMemoryDataConfig
        range_type = BridgeMemoryRecordRange
        loss_type = BridgeMemoryLossConfig
    else:
        raise ValueError("data.record_format must be causal_lm or bridge_memory")
    for split in ("train", "val"):
        if split in data_options:
            data_options[split] = range_type(**data_options[split])

    values = {
        "data": data_type(**data_options),
        "loss": loss_type(**dict(sections.get("loss", {}))),
    }
    for name, section_type in section_types.items():
        options = dict(sections.get(name, {}))
        values[name] = section_type(**options)
    config = TrainingConfig(**values)
    config.validate()
    return config
