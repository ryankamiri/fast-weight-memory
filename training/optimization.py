"""Explicit optimizer groups and observable clipping for matched TaaL experiments."""

from dataclasses import dataclass
import math

import torch


@dataclass
class GradientClippingResult:
    total_norm: float
    metrics: dict[str, float]


def build_optimizer(model, config):
    memory = []
    gates = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.endswith(".taal.residual_gate"):
            gates.append(parameter)
        else:
            memory.append(parameter)
    settings = config.optimizer
    if settings.gate_lr_multiplier == 1.0:
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=settings.lr,
            betas=(settings.beta1, settings.beta2),
            eps=settings.eps,
            weight_decay=settings.weight_decay,
        )
    groups = [{"params": memory, "lr": settings.lr, "name": "memory"}]
    if gates:
        groups.append(
            {
                "params": gates,
                "lr": settings.lr * settings.gate_lr_multiplier,
                "name": "gates",
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=(settings.beta1, settings.beta2),
        eps=settings.eps,
        weight_decay=settings.weight_decay,
    )


def clip_training_gradients(model, config):
    memory = []
    gates = []
    metrics = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.endswith(".taal.residual_gate"):
            gates.append(parameter)
            if parameter.grad is not None:
                metrics[f"gate_gradient_before/{name}"] = (
                    parameter.grad.detach().float().item()
                )
        else:
            memory.append(parameter)
    if config.optimizer.gate_max_grad_norm is None:
        total = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.training.max_grad_norm,
        ).item()
    else:
        memory_norm = torch.nn.utils.clip_grad_norm_(
            memory, config.training.max_grad_norm
        ).item()
        gate_norm = torch.nn.utils.clip_grad_norm_(
            gates, config.optimizer.gate_max_grad_norm
        ).item()
        total = math.hypot(memory_norm, gate_norm)
        metrics["optimizer/memory_grad_norm"] = memory_norm
        metrics["optimizer/gate_grad_norm"] = gate_norm
    for name, parameter in model.named_parameters():
        if name.endswith(".taal.residual_gate") and parameter.grad is not None:
            metrics[f"gate_gradient_after/{name}"] = (
                parameter.grad.detach().float().item()
            )
    return GradientClippingResult(total, metrics)
