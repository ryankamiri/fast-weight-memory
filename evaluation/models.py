"""Checkpoint configuration and exact loading for evaluation backends."""

import math

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.ttcd.qwen.causal_lm import TTCDQwen3ForCausalLM
from architectures.ttcd.qwen.configuration import TTCDQwen3Config


def validate_read_scale(scale, name="memory_read_scale"):
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def configure_model(config, mode, settings=None):
    if mode == "taal":
        if not isinstance(config, TaalQwen3Config):
            raise ValueError("taal mode requires a TaaL checkpoint")

        settings = settings or {}
        unsupported = set(settings) - {"memory_read_scale", "retain_task_prefix"}
        if unsupported:
            raise ValueError(
                f"TaaL evaluation preserves checkpoint architecture; unsupported settings: {sorted(unsupported)}"
            )

        scale = settings.get("memory_read_scale", 1.0)
        validate_read_scale(scale)

        if type(settings.get("retain_task_prefix", False)) is not bool:
            raise ValueError("retain_task_prefix must be boolean")

        return config
    if isinstance(config, TaalQwen3Config):
        raise ValueError("TaaL checkpoints require taal mode")

    if mode not in {"full", "swa", "fw_swa"}:
        raise ValueError("mode must be full, swa, or fw_swa")
    if bool(config.fast_weight_layers) != (mode == "fw_swa"):
        raise ValueError("Checkpoint fast-weight layers do not match evaluation mode")
    if settings:
        overrides = {}
        for name in ("teacher_window_size", "student_window_size", "chunk_size"):
            if name in settings:
                overrides[name] = settings[name]
        config = TTCDQwen3Config(**{**config.to_dict(), **overrides})
    return config


def load_model(source, config, fast_weight_read_scale=1.0, dtype=torch.bfloat16):
    validate_read_scale(fast_weight_read_scale, "fast_weight_read_scale")

    model_class = TTCDQwen3ForCausalLM
    if isinstance(config, TaalQwen3Config):
        model_class = TaalQwen3ForCausalLM
    else:
        config.fast_weight_read_scale = float(fast_weight_read_scale)
    model, loading = model_class.from_pretrained(
        source,
        config=config,
        dtype=dtype,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Checkpoint did not load exactly: {loading}")
    return model
