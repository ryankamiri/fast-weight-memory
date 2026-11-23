"""Explicit execution policy shared by training and numerical diagnostics."""

from contextlib import contextmanager, nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

PRECISIONS = ("bf16_auto", "bf16_math", "fp32_auto", "fp32_math")


@contextmanager
def execution_precision(device, precision):
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")

    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        # FP32 controls must not silently retain TF32 matrix multiplication.
        if precision.startswith("fp32"):
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        attention = sdpa_kernel(SDPBackend.MATH) if precision.endswith("math") else nullcontext()
        with attention, torch.autocast(device.type, dtype=torch.bfloat16,
                                      enabled=precision.startswith("bf16")):
            yield {
                "precision": precision,
                "autocast": torch.is_autocast_enabled(device.type),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "sdpa_enabled": {
                    "math": torch.backends.cuda.math_sdp_enabled(),
                    "flash": torch.backends.cuda.flash_sdp_enabled(),
                    "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                },
            }
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def precision_context(device, policy="bf16"):
    if policy not in ("bf16", "fp32"):
        raise ValueError(f"Unknown training precision: {policy}")
    if device.type == "cpu" and policy == "bf16":
        return nullcontext()
    return execution_precision(device, policy + "_auto")
